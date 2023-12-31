import os
import glob
import shutil
import librosa
import whisper
import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel
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


class AsrSegmentResult(BaseModel):
    start: float
    end: float
    text: str


def split_audio_asr(
    audio_filepath: str,
    spk_id: str,
    whisper_model: Whisper,
    target_sr: int = 44100
) -> Union[None, List[str]]:
    """使用 ASR 切分音频，因为模型训练所需音频的采样率未 44100，所以这里切分后的音频都重采样为 44100"""
    def __reformat_asr_segments(asr_segments: List[AsrSegmentResult]) -> List[AsrSegmentResult]:
        """重新格式化的操作:
        满足以下条件，则该 segment 向后拼接
        1. 如果当前 segment 的文本长度 < 5 或音频时长 < 1s，则向后拼接
        """
        segments = asr_segments.copy()
        reformat_segments = []
        for idx, seg_res in enumerate(segments):
            # 判断字符长度，如果小于 5，则往后拼接
            if (idx < len(segments) - 1) and (
                len(seg_res.text) < 5 or (seg_res.end - seg_res.start) < 1
            ):
                segments[idx + 1].start = seg_res.start
                segments[idx + 1].text = seg_res.text + " " + segments[idx + 1].text
            else:
                reformat_segments.append(seg_res)

        return reformat_segments

    asr_res = whisper.transcribe(
        model=whisper_model,
        audio=audio_filepath,
        initial_prompt="以下是普通话的转写文本",
        task="transcribe", beam_size=5
    )

    if asr_res["language"] != "zh":
        logger.warning("Warning: 检测到音频非中文")
        return None

    asr_res = [AsrSegmentResult(**item) for item in asr_res["segments"]]
    asr_res_reformatted = __reformat_asr_segments(asr_res)

    save_dir = os.path.join("audio", "segments", spk_id)
    audio_filename = os.path.basename(audio_filepath).split('.')[0]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    _ = [os.remove(file) for file in glob.glob(f"audio/segments/{spk_id}/{audio_filename}*.wav")]
    wav, sr = librosa.load(audio_filepath, sr=None, offset=0, duration=None, mono=True)

    audios_segments_info = []
    for seg in asr_res_reformatted:
        seg_duration = round(seg.end - seg.start, 2)

        logger.info(
            f"\ntime: {round(seg.start, 2)}-{round(seg.end, 2)}, duration: {seg_duration}"
            f"\ntext: {seg.text}"
        )
        audio_seg = wav[int(seg.start * target_sr):int(seg.end * target_sr)]
        audio_idx = len(os.listdir(save_dir)) + 1
        out_filepath = f"{save_dir}/{audio_filename}_{audio_idx}.wav"
        wavfile.write(out_filepath, rate=target_sr, data=audio_seg)
        audios_segments_info.append([out_filepath, spk_id, "ZH", seg.text, seg_duration])

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
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Error: {audio_path} 录音文件不存在")

    if not os.path.exists(f"audio/vad/{spk_id}"):
        os.mkdir(f"audio/vad/{spk_id}")

    audio_duration = librosa.get_duration(path=audio_path)
    wav, sr = librosa.load(audio_path, sr=None, offset=0, duration=None, mono=True)
    wav2 = librosa.resample(wav, orig_sr=sr, target_sr=resample_rate)
    wavfile.write(audio_path, rate=resample_rate, data=wav2)

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
