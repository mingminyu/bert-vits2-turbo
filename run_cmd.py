import torch
import whisper
import argparse
from loguru import logger
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from utils.audio import denoise_audio, split_audio_vad_asr
from tools.init_project import create_project_dirs
from tools.preprocess_text import generate_training_samples
from tools.gen_bert import process_line
from tools.train_ms import train


def stage3_denoise_audio():
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
        stage3_denoise_audio()
    elif args.stage == 4:
        stage4_split_audio(args.whisper_size)
    elif args.stage == 5:
        generate_training_samples()
    elif args.stage == 6:
        from multiprocessing import Pool
        from tqdm import tqdm

        lines = []
        with open("data/train/train_samples.csv", encoding='utf-8') as f:
            lines.extend(f.readlines())

        with Pool(processes=2) as pool:  # A100 40GB suitable config,if coom,please decrease the processess number.
            for _ in tqdm(pool.imap_unordered(process_line, lines)):
                pass
    elif args.stage == 7:
        train()


