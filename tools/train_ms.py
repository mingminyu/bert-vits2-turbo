import glob
import os
import torch
import shutil
from tqdm import tqdm
from torch import distributed as dist
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

from utils import commons, util
from utils.data_utils import TextAudioSpeakerLoader, TextAudioSpeakerCollate, DistributedBucketSampler
from utils.models import SynthesizerTrn, MultiPeriodDiscriminator, DurationDiscriminator
from utils.losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from utils.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols
from utils.config import Vits2Config, DataConfig

# import logging
# import json
# import argparse
# import itertools
# import math
# from torch import nn, optim
# import logging
# import torch.multiprocessing as mp

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')
global_step = 0


# def train():
#     """Assume Single Node Multi GPUs Training Only"""
#     assert torch.cuda.is_available(), "CPU training is not allowed."
#
#     n_gpus = torch.cuda.device_count()
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '65280'
#
#     hps = util.get_hparams()
#     if not hps.cont:
#         shutil.copy('./pretrained_models/vits2_base_model/D_0.pth', './logs/OUTPUT_MODEL/D_0.pth')
# #         shutil.copy('./pretrained_models/vits2_base_model/G_0.pth', './logs/OUTPUT_MODEL/G_0.pth')
# #         shutil.copy('./pretrained_models/vits2_base_model/DUR_0.pth', './logs/OUTPUT_MODEL/DUR_0.pth')
#     mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


def run(config: Vits2Config, n_gpus: int = 1, cont=False):
    train_ms_cfg = config.train_ms_cfg
    train_cfg = config.train_cfg
    data_cfg = config.data_cfg
    model_cfg = config.model_cfg
    rank = 0
    # setup environment variables
    os.environ['MASTER_ADDR'] = train_ms_cfg.env.master_addr
    os.environ['MASTER_PORT'] = train_ms_cfg.env.master_port

    if not cont:
        _ = [
            shutil.copy(file, train_ms_cfg.save_model_path)
            for file in glob.glob(f"{train_ms_cfg.base_model_path}/*.pth")
        ]


    global global_step
    # if rank == 0:
    logger = util.get_logger(train_ms_cfg.save_model_path)
    # logger.info(hps)
    util.check_git_hash(train_ms_cfg.save_model_path)
    writer = SummaryWriter(log_dir=train_ms_cfg.save_model_path)
    writer_eval = SummaryWriter(log_dir=os.path.join(train_ms_cfg.save_model_path, "eval"))

    dist.init_process_group(
        backend='gloo' if os.name == 'nt' else 'nccl',
        init_method='env://', world_size=n_gpus, rank=rank
    )
    # torch.manual_seed(hps.train.seed)
    torch.manual_seed(train_cfg.seed)
    torch.cuda.set_device(rank)

    # train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
    train_dataset = TextAudioSpeakerLoader(data_cfg.training_files, data_cfg)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        train_cfg.batch_size,
        [32, 300, 400, 500, 600, 700, 800, 900, 1000],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True)
    collate_fn = TextAudioSpeakerCollate()
    train_loader = DataLoader(train_dataset, num_workers=2, shuffle=False, pin_memory=True,
                              collate_fn=collate_fn, batch_sampler=train_sampler)
    # if rank == 0:
        # eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data)
    eval_dataset = TextAudioSpeakerLoader(data_cfg.validation_files, data_cfg)
    eval_loader = DataLoader(
        eval_dataset, num_workers=0, shuffle=False, batch_size=1, pin_memory=True,
        drop_last=False, collate_fn=collate_fn
    )

    # if "use_noise_scaled_mas" in hps.model.keys() and hps.model.use_noise_scaled_mas == True:
    if model_cfg.use_noise_scaled_mas is True:
        print("Using noise scaled MAS for VITS2")
        # use_noise_scaled_mas = True
        mas_noise_scale_initial = 0.01
        noise_scale_delta = 2e-6
    else:
        print("Using normal MAS for VITS1")
        # use_noise_scaled_mas = False
        mas_noise_scale_initial = 0.0
        noise_scale_delta = 0.0

    # if "use_duration_discriminator" in hps.model.keys() and hps.model.use_duration_discriminator == True:
    if model_cfg.use_duration_discriminator is True:
        print("Using duration discriminator for VITS2")
        use_duration_discriminator = True
        net_dur_disc = DurationDiscriminator(
            model_cfg.hidden_channels,
            model_cfg.hidden_channels,
            3,
            0.1,
            gin_channels=model_cfg.gin_channels if data_cfg.n_speakers != 0 else 0,
        ).cuda(rank)

    if model_cfg.use_spk_conditioned_encoder is True:
        if data_cfg.n_speakers == 0:
            raise ValueError("n_speakers must be > 0 when using spk conditioned encoder to train multi-speaker model")
        # use_spk_conditioned_encoder = True
    else:
        print("Using normal encoder for VITS1")
        # use_spk_conditioned_encoder = False

    net_g = SynthesizerTrn(
        len(symbols),
        data_cfg.filter_length // 2 + 1,
        train_cfg.segment_size // data_cfg.hop_length,
        n_speakers=data_cfg.n_speakers,
        mas_noise_scale_initial=mas_noise_scale_initial,
        noise_scale_delta=noise_scale_delta,
        **model_cfg.dict()
    ).cuda(rank)

    # freeze_enc = getattr(hps.model, "freeze_enc", False)
    freeze_enc = False
    if freeze_enc:
        print("freeze encoder !!!")
        for param in net_g.enc_p.parameters():
            param.requires_grad = False

    net_d = MultiPeriodDiscriminator(model_cfg.use_spectral_norm).cuda(rank)
    optim_g = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net_g.parameters()),
        train_cfg.learning_rate,
        betas=train_cfg.betas,
        eps=train_cfg.eps)
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        train_cfg.learning_rate,
        betas=train_cfg.betas,
        eps=train_cfg.eps)

    if net_dur_disc is not None:
        optim_dur_disc = torch.optim.AdamW(
            net_dur_disc.parameters(),
            train_cfg.learning_rate,
            betas=train_cfg.betas,
            eps=train_cfg.eps)
    else:
        optim_dur_disc = None

    net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
    net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)
    if net_dur_disc is not None:
        net_dur_disc = DDP(net_dur_disc, device_ids=[rank], find_unused_parameters=True)

    pretrain_dir = None
    if pretrain_dir is None:
        try:
            if net_dur_disc is not None:
                _, optim_dur_disc, _, epoch_str = util.load_checkpoint(
                    util.latest_checkpoint_path(train_ms_cfg.save_model_path, "DUR_*.pth"), net_dur_disc, optim_dur_disc,
                    skip_optimizer=not cont)
            _, optim_g, _, epoch_str = util.load_checkpoint(util.latest_checkpoint_path(train_ms_cfg.save_model_path, "G_*.pth"),
                                                             net_g,
                                                             optim_g, skip_optimizer=not cont)
            _, optim_d, _, epoch_str = util.load_checkpoint(util.latest_checkpoint_path(train_ms_cfg.save_model_path, "D_*.pth"),
                                                             net_d,
                                                             optim_d, skip_optimizer=not cont)

            epoch_str = max(epoch_str, 1)
            global_step = (epoch_str - 1) * len(train_loader)
        except Exception as e:
            print(e)
            epoch_str = 1
            global_step = 0
    else:
        _, _, _, epoch_str = util.load_checkpoint(util.latest_checkpoint_path(pretrain_dir, "G_*.pth"), net_g,
                                                   optim_g, True)
        _, _, _, epoch_str = util.load_checkpoint(util.latest_checkpoint_path(pretrain_dir, "D_*.pth"), net_d,
                                                   optim_d, True)

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=train_cfg.lr_decay, last_epoch=epoch_str - 2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=train_cfg.lr_decay, last_epoch=epoch_str - 2)

    if net_dur_disc is not None:
        scheduler_dur_disc = torch.optim.lr_scheduler.ExponentialLR(optim_dur_disc, gamma=train_cfg.lr_decay,
                                                                    last_epoch=epoch_str - 2)
    else:
        scheduler_dur_disc = None
    scaler = GradScaler(enabled=train_cfg.fp16_run)

    for epoch in range(epoch_str, train_cfg.epochs + 1):
        if rank == 0:
            train_and_evaluate(rank, epoch, config, [net_g, net_d, net_dur_disc], [optim_g, optim_d, optim_dur_disc],
                               [scheduler_g, scheduler_d, scheduler_dur_disc], scaler, [train_loader, eval_loader],
                               logger, [writer, writer_eval])
        else:
            train_and_evaluate(rank, epoch, config, [net_g, net_d, net_dur_disc], [optim_g, optim_d, optim_dur_disc],
                               [scheduler_g, scheduler_d, scheduler_dur_disc], scaler, [train_loader, None], None, None)
        scheduler_g.step()
        scheduler_d.step()
        if net_dur_disc is not None:
            scheduler_dur_disc.step()


def train_and_evaluate(rank, epoch, config, nets, optims, schedulers, scaler, loaders, logger, writers):
    net_g, net_d, net_dur_disc = nets
    optim_g, optim_d, optim_dur_disc = optims
    scheduler_g, scheduler_d, scheduler_dur_disc = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_ms_cfg = config.train_ms_cfg
    train_cfg = config.train_cfg
    data_cfg = config.data_cfg
    model_cfg = config.model_cfg

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()
    if net_dur_disc is not None:
        net_dur_disc.train()
    for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers, tone, language, bert) in tqdm(
            enumerate(train_loader)):

        if net_g.module.use_noise_scaled_mas:
            current_mas_noise_scale = net_g.module.mas_noise_scale_initial - net_g.module.noise_scale_delta * global_step
            net_g.module.current_mas_noise_scale = max(current_mas_noise_scale, 0.0)
        x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
        spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
        y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)
        speakers = speakers.cuda(rank, non_blocking=True)
        tone = tone.cuda(rank, non_blocking=True)
        language = language.cuda(rank, non_blocking=True)
        bert = bert.cuda(rank, non_blocking=True)

        with autocast(enabled=train_cfg.fp16_run):
            y_hat, l_length, attn, ids_slice, x_mask, z_mask, \
                (z, z_p, m_p, logs_p, m_q, logs_q), (hidden_x, logw, logw_) = net_g(x, x_lengths, spec, spec_lengths,
                                                                                    speakers, tone, language, bert)
            mel = spec_to_mel_torch(
                spec,
                data_cfg.filter_length,
                data_cfg.n_mel_channels,
                data_cfg.sampling_rate,
                data_cfg.mel_fmin,
                data_cfg.mel_fmax)
            y_mel = commons.slice_segments(mel, ids_slice, train_cfg.segment_size // data_cfg.hop_length)
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                data_cfg.filter_length,
                data_cfg.n_mel_channels,
                data_cfg.sampling_rate,
                data_cfg.hop_length,
                data_cfg.win_length,
                data_cfg.mel_fmin,
                data_cfg.mel_fmax
            )

            y = commons.slice_segments(y, ids_slice * data_cfg.hop_length, train_cfg.segment_size)  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
                loss_disc_all = loss_disc
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(hidden_x.detach(), x_mask.detach(), logw.detach(),
                                                        logw_.detach())
                with autocast(enabled=False):
                    # TODO: I think need to mean using the mask, but for now, just mean all
                    loss_dur_disc, losses_dur_disc_r, losses_dur_disc_g = discriminator_loss(y_dur_hat_r, y_dur_hat_g)
                    loss_dur_disc_all = loss_dur_disc

                optim_dur_disc.zero_grad()
                scaler.scale(loss_dur_disc_all).backward()
                scaler.unscale_(optim_dur_disc)
                grad_norm_dur_disc = commons.clip_grad_value_(net_dur_disc.parameters(), None)
                scaler.step(optim_dur_disc)

        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=train_cfg.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(hidden_x, x_mask, logw, logw_)
            with autocast(enabled=False):
                loss_dur = torch.sum(l_length.float())
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * train_cfg.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * train_cfg.c_kl

                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
                if net_dur_disc is not None:
                    loss_dur_gen, losses_dur_gen = generator_loss(y_dur_hat_g)
                    loss_gen_all += loss_dur_gen

        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if global_step % train_cfg.log_interval == 0:
                lr = optim_g.param_groups[0]['lr']
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]
                logger.info('Train Epoch: {} [{:.0f}%]'.format(
                    epoch,
                    100. * batch_idx / len(train_loader)))
                logger.info([x.item() for x in losses] + [global_step, lr])

                scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr,
                               "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}
                scalar_dict.update(
                    {"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/dur": loss_dur, "loss/g/kl": loss_kl})
                scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
                scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
                scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})

                image_dict = {
                    "slice/mel_org": util.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                    "slice/mel_gen": util.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
                    "all/mel": util.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
                    "all/attn": util.plot_alignment_to_numpy(attn[0, 0].data.cpu().numpy())
                }
                util.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict)

            if global_step % train_cfg.eval_interval == 0:
                evaluate(data_cfg, net_g, eval_loader, writer_eval)
                util.save_checkpoint(net_g, optim_g, train_cfg.learning_rate, epoch,
                                      os.path.join(train_ms_cfg.save_model_path, "G_{}.pth".format(global_step)))
                util.save_checkpoint(net_d, optim_d, train_cfg.learning_rate, epoch,
                                      os.path.join(train_ms_cfg.save_model_path, "D_{}.pth".format(global_step)))
                if net_dur_disc is not None:
                    util.save_checkpoint(net_dur_disc, optim_dur_disc, train_cfg.learning_rate, epoch,
                                          os.path.join(train_ms_cfg.save_model_path, "DUR_{}.pth".format(global_step)))
                # keep_ckpts = getattr(hps.train, 'keep_ckpts', 5)
                keep_ckpts = train_cfg.keep_ckpts
                if keep_ckpts > 0:
                    util.clean_checkpoints(path_to_models=train_ms_cfg.save_model_path, n_ckpts_to_keep=keep_ckpts,
                                           sort_by_time=True)

        global_step += 1

    if rank == 0:
        logger.info('====> Epoch: {}'.format(epoch))


def evaluate(data_cfg: DataConfig, generator, eval_loader, writer_eval):
    generator.eval()
    image_dict = {}
    audio_dict = {}
    print("Evaluating ...")

    with torch.no_grad():
        for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers, tone, language, bert) in enumerate(
                eval_loader):
            x, x_lengths = x.cuda(), x_lengths.cuda()
            spec, spec_lengths = spec.cuda(), spec_lengths.cuda()
            y, y_lengths = y.cuda(), y_lengths.cuda()
            speakers = speakers.cuda()
            bert = bert.cuda()
            tone = tone.cuda()
            language = language.cuda()
            for use_sdp in [True, False]:
                y_hat, attn, mask, *_ = generator.module.infer(x, x_lengths, speakers, tone, language, bert, y=spec,
                                                               max_len=1000, sdp_ratio=0.0 if not use_sdp else 1.0)
                y_hat_lengths = mask.sum([1, 2]).long() * data_cfg.hop_length

                mel = spec_to_mel_torch(
                    spec,
                    data_cfg.filter_length,
                    data_cfg.n_mel_channels,
                    data_cfg.sampling_rate,
                    data_cfg.mel_fmin,
                    data_cfg.mel_fmax)
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1).float(),
                    data_cfg.filter_length,
                    data_cfg.n_mel_channels,
                    data_cfg.sampling_rate,
                    data_cfg.hop_length,
                    data_cfg.win_length,
                    data_cfg.mel_fmin,
                    data_cfg.mel_fmax
                )
                image_dict.update({
                    f"gen/mel_{batch_idx}": util.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
                })
                audio_dict.update({
                    f"gen/audio_{batch_idx}_{use_sdp}": y_hat[0, :, :y_hat_lengths[0]]
                })
                image_dict.update({f"gt/mel_{batch_idx}": util.plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
                audio_dict.update({f"gt/audio_{batch_idx}": y[0, :, :y_lengths[0]]})

    util.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=data_cfg.sampling_rate
    )
    generator.train()

