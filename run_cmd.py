import glob
import torch
import whisper
import argparse
from tqdm import tqdm
from loguru import logger
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from utils.audio import denoise_audio, split_audio_vad_asr, split_audio_channel
from utils.config import Vits2Config
from tools.init_project import create_project_dirs, download_project_pretrained_models
from tools.gen_samples import generate_training_samples
from tools.gen_bert import process_line
from tools.train_ms import run
from tools.infer import generate_tts_audio



def stage1_init_project(
    download_pretrained_models: bool = False,
    whisper_size: str = "medium"
):
    create_project_dirs()

    if download_pretrained_models:
        download_project_pretrained_models(whisper_size)


def stage2_split_audio_channel(
        audio_path: str,
        spk_id: str,
        keep_channel: int
):
    split_audio_channel(
        audio_filepath=audio_path,
        spk_id=spk_id,
        keep_channel=keep_channel
    )


def stage3_denoise_audio(
        audio_path: str,
        spk_id: str,
):
    denoise_model_name = "speech_dfsmn_ans_psm_48k_causal"
    logger.info(f"加载音频降噪模型: {denoise_model_name}")
    denoise_ans = pipeline(
        Tasks.acoustic_noise_suppression,
        model=f'./pretrained_models/damo/{denoise_model_name}'
    )

    denoise_audio_path = denoise_audio(
        denoise_ans=denoise_ans,
        audio_filepath=audio_path,
        spk_id=spk_id,
    )
    logger.info(f"音频降噪完成，路径为: {denoise_audio_path}")


def stage4_split_audio(
        audio_path: str,
        spk_id: str,
        whisper_model_size: str = "medium"
):
    """切分音频"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"加载 Whisper ASR 模型: {whisper_model_size}")
    whisper_model = whisper.load_model(
        whisper_model_size, device=device,
        download_root="./pretrained_models/whisper"
    )

    vad_model_name = "speech_fsmn_vad_zh-cn-16k-common-pytorch"
    logger.info(f"加载静音检测模型: {vad_model_name}")
    vad_ans = pipeline(
        task=Tasks.voice_activity_detection,
        model='./pretrained_models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch',
        model_revision=None,
    )

    split_audio_vad_asr(
        audio_path=audio_path,
        spk_id=spk_id,
        whisper_model=whisper_model,
        vad_ans=vad_ans,
    )


def stage5_generate_training_samples(config: Vits2Config):
    """生成训练样本"""
    generate_training_samples(config.gen_samples_cfg)


def stage6_generate_bert_files(config: Vits2Config):
    """生成 BERT 训练文件"""
    lines = []
    with open(config.gen_samples_cfg.train_path, encoding='utf-8') as f:
        lines.extend(f.readlines())
    
    for line in tqdm(lines):
        process_line(line, config.data_cfg.add_blank)


def stage8_generate_audio(sid: str, config: Vits2Config, model_step: str = None):
    """生成音频"""
    text = "先生，怎么逾期那么久了，还不还钱啊"
    audio_path = generate_tts_audio(
        text=text,
        sid=sid,
        audio_save_filename=f"{sid}_tts.wav",
        config=config,
        model_path=f"{config.train_ms_cfg.save_model_path}/G_{model_step}.pth"
    )
    return audio_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", '-s', default=1, type=int)
    parser.add_argument("--spk_id", '-sid', default="", type=str)
    parser.add_argument("--model_step", default="", type=str)
    parser.add_argument(
        '--download_pretrained_models', '-dpm', action="store_true",
        default=False, help="是否下载预训练模型"
    )
    parser.add_argument("--whisper_size", '-ws', default="medium", type=str)
    args = parser.parse_args()

    proj_cfg = Vits2Config(
        yaml_cfg_path="config/config.yml", json_cfg_path="config/config.json"
    )

    if args.stage == 1:
        stage1_init_project(args.download_pretrained_models, args.whisper_size)
    elif args.stage == 2:
        upload_files = glob.glob("audio/upload/*/*.wav")
        for upload_file in tqdm(upload_files):
            sid = upload_file.split("/")[-2]
            stage2_split_audio_channel(upload_file, sid, 1)

    elif args.stage == 3:
        split_files = glob.glob("audio/split/*/*.wav")
        for split_file in tqdm(split_files):
            sid = split_file.split("/")[-2]
            stage3_denoise_audio(split_file, sid)

    elif args.stage == 4:
        denoise_files = glob.glob("audio/denoise/*/*.wav")
        for denoise_file in tqdm(denoise_files):
            sid = denoise_file.split("/")[-2]
            stage4_split_audio(denoise_file, sid, args.whisper_size)

    elif args.stage == 5:
        stage5_generate_training_samples(proj_cfg)
    elif args.stage == 6:
        stage6_generate_bert_files(proj_cfg)
    elif args.stage == 7:
        run(config=proj_cfg)
    elif args.stage == 8:
        stage8_generate_audio(sid=args.spk_id, config=proj_cfg, model_step=args.model_step)
    else:
        raise ValueError("`--stage` parameter must be in 1~8, default 1.")
