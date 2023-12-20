import os
import glob
import shutil
import librosa
import whisper
import numpy as np
import pandas as pd
from loguru import logger
from scipy.io import wavfile
from typing import List, Union
from modelscope.pipelines import Pipeline
from whisper.model import Whisper



def split_audio_channel(
    audio_filepath: str,
    keep_channel: int,
    spk_id: str
) -> str:
    """切分双声道音频"""
    if not os.path.exists(audio_filepath):
        raise FileNotFoundError(f"Error: {audio_filepath} 录音文件不存在")

    if not os.path.exists(f"audio/split/{spk_id}"):
        os.mkdir(f"audio/split/{spk_id}")

    save_path = os.path.join(f"audio/split/{spk_id}", os.path.basename(audio_filepath))

    if keep_channel == 2:
        shutil.copy(audio_filepath, f"audio/split/{spk_id}")
        return save_path

    try:
        sr, wav_data = wavfile.read(audio_filepath)
    except ValueError:
        wav_data, sr = librosa.load(audio_filepath)

    if len(wav_data.shape) == 1:
        shutil.copy(audio_filepath, f"audio/split/{spk_id}")
        return save_path

    role_wav_data = [item[keep_channel] for item in wav_data]
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
) -> Union[None, List[str]]:
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

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    _ = [os.remove(file) for file in glob.glob(f"audio/segments/{spk_id}/{audio_filename}*.wav")]

    audios_segments_info = []
    for seg in asr_res["segments"]:
        start_time, end_time = seg["start"], seg["end"]
        seg_duration = seg["end"] - seg["start"]
        seg_text = seg["text"]

        if seg_duration < 1.5 or seg_duration > 8 or len(seg_text) < 5:
            continue

        logger.info(
            f"\ntime: {round(start_time, 2)}-{round(end_time, 2)}, duration: {round(seg_duration, 2)}"
            f"\ntext: {seg_text}"
        )
        audio_seg = wav2[int(start_time * target_sr):int(end_time * target_sr)]
        audio_idx = len(os.listdir(save_dir)) + 1
        out_filepath = f"{save_dir}/{audio_filename}_{audio_idx}.wav"
        wavfile.write(
            out_filepath, rate=target_sr,
            data=(audio_seg * np.iinfo(np.int16).max).astype(np.int16)
        )

        audios_segments_info.append([out_filepath, spk_id, "ZH", seg_text, round(seg_duration, 2)])

    df_audio_asr = pd.DataFrame(
        audios_segments_info, columns=["audio_id", "spk_id", "lang", "text", "duration"]
    )

    if not os.path.exists(f"data/asr/{spk_id}"):
        os.makedirs(f"data/asr/{spk_id}")

    df_audio_asr.to_csv(f"data/asr/{spk_id}/{audio_filename}.csv",
                        index=False, encoding="utf8", sep="|")
    audios_segments = [audios_segment_info[0] for audios_segment_info in audios_segments_info]
    return audios_segments


def split_audio_vad_asr(
    audio_path: str,
    spk_id: str,
    whisper_model: Whisper,
    vad_ans: Pipeline = None,
    resample_rate: int = 44100
) -> None:
    """切分音频"""
    # TODO: 完成 VAD 部分
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Error: {audio_path} 录音文件不存在")

    if not os.path.exists(f"audio/vad/{spk_id}"):
        os.mkdir(f"audio/vad/{spk_id}")

    audio_duration = librosa.get_duration(path=audio_path)

    # 如果音频时长 <= 180s，直接使用 Whisper ASR 进行音频切分
    if audio_duration <= 180:
        split_audio_asr(audio_path, spk_id, whisper_model)
    else:
        wav_data, sr = librosa.load(audio_path)
        wav_data, _ = librosa.effects.trim(wav_data, top_db=20)
        peak = np.abs(wav_data).max()

        if peak > 1.0:
            wav_data = 0.98 * wav_data / peak

        # vad 算法采用的采样率为 16000
        wav_data = librosa.resample(wav_data, orig_sr=sr, target_sr=16000)
        wav_data /= max(wav_data.max(), -wav_data.min())

        # 有些音频经过 VAD 检测后，没有检测出静音，需要使用 whisper 再次切分
        vad_result = vad_ans(audio_in=wav_data)
        audio_duration = wav_data.shape[0] / 16
        logger.info(f"{audio_path} duration: {int(audio_duration)}")

        wav_filename = os.path.basename(audio_path).split('.')[0]
        vad_save_dir = f"audio/vad/{spk_id}"

        # 如果已经有 vad 生成文件，则清除
        vad_wavs_exist = glob.glob(f"{vad_save_dir}/{wav_filename}*.wav")
        _ = [os.remove(vad_wav_exist) for vad_wav_exist in vad_wavs_exist]

        cnt = 0
        for idx, seg in enumerate(vad_result["text"]):
            if (seg[1] - seg[0]) >= 3000:
                duration = (seg[1] - seg[0]) / 1000
                seg_start, seg_end = int(seg[0] * 16), int(seg[1] * 16)

                output_path = os.path.join(vad_save_dir, f"{wav_filename}_{cnt}.wav")
                logger.info(f"{output_path} duration: {int(duration)}")
                wav_seg_data = librosa.resample(wav_data[seg_start:seg_end], orig_sr=16000, target_sr=44100)
                wav_seg_data = (wav_seg_data * np.iinfo(np.int16).max).astype(np.int16)

                # 再次重采样至 44100
                wavfile.write(output_path, rate=44100, data=wav_seg_data)
                # 再经过 whisper 切分
                split_audio_asr(output_path, spk_id, whisper_model)
                cnt += 1

