import torch
import whisper
import argparse
from loguru import logger
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from utils.audio import denoise_audio, split_audio_vad_asr
from tools.init_project import create_project_dirs


def stage2_denoise_audio():
    denoise_model_name = "speech_dfsmn_ans_psm_48k_causal"
    logger.info(f"加载音频降噪模型: {denoise_model_name}")
    denoise_ans = pipeline(
        Tasks.acoustic_noise_suppression,
        model=f'./pretrained_models/{denoise_model_name}'
    )

    denoise_audio_path = denoise_audio(
        denoise_ans=denoise_ans,
        audio_filepath="audio/examples/nana_speech.wav",
        spk_id="nana",
    )
    logger.info(f"音频降噪完成，路径为: {denoise_audio_path}")


def stage3_split_audio(whisper_model_size: str = "medium"):
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
        whisper_model=whisper_model,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default=1, type=int)
    parser.add_argument("--whisper_size", default="medium")
    args = parser.parse_args()

    if args.stage == 1:
        create_project_dirs()
    elif args.stage == 2:
        ...  # 分离双声道
    elif args.stage == 3:
        stage2_denoise_audio()
    elif args.stage == 4:
        stage3_split_audio(args.whisper_size)
