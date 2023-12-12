import os
import glob
import shutil
import warnings
import librosa
import whisper
import torch
import numpy as np
import pandas as pd
from loguru import logger
from scipy.io import wavfile
from typing import Literal, List, Dict, Union
from modelscope.pipelines import Pipeline
from whisper.model import Whisper



def split_audi_channel(
    audio_filepath: str,
    save_channel: Literal[0, 1, 2],
    spk_id: str
) -> str:
    """切分双声道音频"""
    if not os.path.exists(audio_filepath):
        raise FileNotFoundError(f"Error: {audio_filepath} 录音文件不存在")

    if not os.path.exists(f"audio/split/{spk_id}"):
        os.mkdir(f"audio/split/{spk_id}")

    save_path = os.path.join(f"audio/split/{spk_id}", os.path.basename(audio_filepath))

    if save_channel == 2:
        shutil.copy(audio_filepath, f"audio/split/{spk_id}")
        return save_path

    sr, wav_data = wavfile.read(audio_filepath)
    role_wav_data = [item[save_channel] for item in wav_data]
    save_path = os.path.join(f"audio/split/{spk_id}", os.path.basename(audio_filepath))
    wavfile.write(save_path, sr, np.array(role_wav_data))
    return save_path


def denoise_audio(
    denoise_ans: Pipeline,
    audio_filepath: str,
    spk_id: str
) -> str:
    """音频降噪"""
    if not os.path.exists(audio_filepath):
        raise FileNotFoundError(f"Error: {audio_filepath} 录音文件不存在")

    if not os.path.exists(f"audio/denoise/{spk_id}"):
        os.mkdir(f"audio/denoise/{spk_id}")

    save_path = os.path.join(f"audio/denoise/{spk_id}", os.path.basename(audio_filepath))
    denoise_ans(input=audio_filepath, output_path=save_path)
    return save_path


def split_audio_asr(
    audio_filepath: str,
    spk_id: str,
    whisper_model: Whisper,
    target_sr: int = 44100
):
    """使用 ASR 切分音频，因为模型训练所需音频的采样率未 44100，所以这里切分后的音频都重采样为 44100"""
    asr_res = whisper.transcribe(
        model=whisper_model,
        audio=audio_filepath,
        initial_prompt="以下是普通话的转写文本",
        task="transcribe", beam_size=5
    )

    if asr_res["language"] != "zh":
        logger.warning("Warning: 检测到音频非中文")
        return None

    wav, sr = librosa.load(audio_filepath, sr=None, offset=0, duration=None, mono=True)
    wav, _ = librosa.effects.trim(wav, top_db=20)
    peak = np.abs(wav).max()

    if peak > 1.0:
        wav = 0.98 * wav / peak

    # 对音频重置采样为 target_sr
    wav2 = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    wav2 /= max(wav2.max(), -wav2.min())

    save_dir = os.path.join("audio", "segments", spk_id)
    audio_filename = os.path.basename(audio_filepath).split('.')[0]

    _ = [os.remove(file) for file in glob.glob(f"audio/segments/{spk_id}/{audio_filename}*.wav")]

    audios_segments_info = []
    for seg in asr_res["segments"]:
        start_time, end_time = seg["start"], seg["end"]
        seg_duration = seg["end"] - seg["start"]
        seg_text = seg["text"]
        logger.info(
            f"\ntime: {start_time}-{end_time}, duration: {seg_duration}"
            f"\ntext: {seg_text}"
        )
        audio_seg = wav2[int(start_time * target_sr):int(end_time * target_sr)]
        audio_idx = len(os.listdir(save_dir)) + 1
        out_filepath = f"{save_dir}/{audio_filename}_{audio_idx}.wav"
        wavfile.write(
            out_filepath, rate=target_sr,
            data=(audio_seg * np.iinfo(np.int16).max).astype(np.int16)
        )

        audios_segments_info.append([out_filepath, seg_text, seg_duration])

    df_audio_asr = pd.DataFrame(audios_segments_info, columns=["audio_id", "text", "duration"])
    df_audio_asr.to_csv(f"data/asr/{audio_filename}.csv", index=False, encoding="utf8")
    audios_segments = [audios_segment_info[0] for audios_segment_info in audios_segments_info]
    return audios_segments


def split_audio_vad_asr(
    audio_filepath: str,
    spk_id: str,
    whisper_model: Whisper,
    vad_ans: Pipeline = None,
    resample_rate: int = 44100
) -> None:
    """切分音频"""
    # TODO: 完成 VAD 部分
    if not os.path.exists(audio_filepath):
        raise FileNotFoundError(f"Error: {audio_filepath} 录音文件不存在")

    if not os.path.exists(f"audio/vad/{spk_id}"):
        os.mkdir(f"audio/vad/{spk_id}")

    audio_duration = librosa.get_duration(path=audio_filepath)

    # 如果音频时长 <= 180s，直接使用 Whisper ASR 进行音频切分
    if audio_duration <= 180:
        split_audio_asr(audio_filepath, spk_id, whisper_model)
    else:
        ...

