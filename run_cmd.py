import torch
import whisper
import argparse
from tqdm import tqdm
from loguru import logger
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from utils.audio import denoise_audio, split_audio_vad_asr
from utils.config import Vits2Config
from tools.init_project import create_project_dirs, download_pretrained_models
from tools.gen_samples import generate_training_samples
from tools.gen_bert import process_line
from tools.train_ms import run
from tools.infer import generate_tts_audio



def step1_init_project(
    download_project_pretrained_models: bool = False,
    whisper_size: str = "medium"
):
    create_project_dirs()

    if download_project_pretrained_models:
        download_pretrained_models(whisper_size)


def stage3_denoise_audio():
    denoise_model_name = "speech_dfsmn_ans_psm_48k_causal"
    logger.info(f"加载音频降噪模型: {denoise_model_name}")
    denoise_ans = pipeline(
        Tasks.acoustic_noise_suppression,
        model=f'./pretrained_models/damo/{denoise_model_name}'
    )

    denoise_audio_path = denoise_audio(
        denoise_ans=denoise_ans,
        audio_filepath="audio/examples/nana_speech.wav",
        spk_id="nana",
    )
    logger.info(f"音频降噪完成，路径为: {denoise_audio_path}")


def stage4_split_audio(whisper_model_size: str = "medium"):
    """切分音频"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"加载 Whisper ASR 模型: {whisper_model_size}")
    whisper_model = whisper.load_model(
        whisper_model_size, device=device,
        download_root="./pretrained_models/whisper"
    )

    split_audio_vad_asr(
        audio_filepath="audio/denoise/nana/nana_speech.wav",
        spk_id="nana",
        whisper_model=whisper_model
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
        audio_save_filename="nana_tts.wav",
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
        '--download_project_pretrained_models', '-dppm', action="store_true",
        default=False, help="是否下载预训练模型"
    )
    parser.add_argument("--whisper_size", '-ws', default="medium", type=str)
    args = parser.parse_args()

    proj_cfg = Vits2Config(yaml_cfg_path="config/config.yml", json_cfg_path="config/config.json")

    if args.stage == 1:
        step1_init_project(args.download_project_pretrained_models, args.whisper_size)
    elif args.stage == 2:
        ...  # 分离双声道
    elif args.stage == 3:
        stage3_denoise_audio()
    elif args.stage == 4:
        stage4_split_audio(args.whisper_size)
    elif args.stage == 5:
        stage5_generate_training_samples(proj_cfg)
    elif args.stage == 6:
        stage6_generate_bert_files(proj_cfg)
    elif args.stage == 7:
        run(config=proj_cfg)
    elif args.stage == 8:
        stage8_generate_audio(sid=args.sid, config=proj_cfg, model_step=args.model_step)
    else:
        raise ValueError("`--stage` parameter must be in 1~8, default 1.")
