import os
import torch
from typing import Literal, Optional
from scipy.io import wavfile
from datetime import datetime

from utils import commons, util
from utils.models import SynthesizerTrn
from text.symbols import symbols
from text.cleaner import clean_text, cleaned_text_to_sequence
from text import get_bert
from utils.config import Vits2Config, DataConfig



def get_net_g(model_path: str, config: Vits2Config, device: str = "cuda:0"):
    """当前版本模型 net_g"""
    data_cfg, train_cfg, model_cfg = config.data_cfg, config.train_cfg, config.model_cfg
    net_g = SynthesizerTrn(
        len(symbols),
        data_cfg.filter_length // 2 + 1,
        train_cfg.segment_size // data_cfg.hop_length,
        n_speakers=data_cfg.n_speakers,
        **model_cfg.dict(),
    ).to(device)
    _ = net_g.eval()
    _ = util.load_checkpoint(model_path, net_g, None, skip_optimizer=True)
    return net_g


def get_text(text: str, lang: str, add_blank: bool = True):
    # 在此处实现当前版本的get_text
    norm_text, phone, tone, word2ph = clean_text(text, lang)
    phone, tone, language = cleaned_text_to_sequence(phone, tone, lang)

    if lang != "ZH":
        raise ValueError("language_str should be ZH, JP or EN")

    if add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)

        word2ph = [
            2 * int(word2ph_) + 1 if idx == 0 else int(word2ph_) * 2
            for idx, word2ph_ in enumerate(word2ph)
        ]
    bert = get_bert(norm_text, word2ph, lang)
    del word2ph
    assert bert.shape[-1] == len(phone), f"Bert seq len {bert.shape[-1]} != {len(phone)}"

    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)
    return bert, phone, tone, language


def infer(
    text: str,
    sdp_ratio: float,
    noise_scale: float,
    noise_scale_w: float,
    length_scale: float,
    sid: str,
    language: str,
    data_cfg: DataConfig,
    net_g: SynthesizerTrn,
    device: str = "cuda:0"
):
    # 在此处实现当前版本的推理
    bert, phones, tones, lang_ids = get_text(text, language, add_blank=data_cfg.add_blank)

    with torch.no_grad():
        x_tst = phones.to(device).unsqueeze(0)
        tones = tones.to(device).unsqueeze(0)
        lang_ids = lang_ids.to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        del phones
        speakers = torch.LongTensor([data_cfg.spk2id[sid]]).to(device)
        audio = (
            net_g.infer(
                x_tst,
                x_tst_lengths,
                speakers,
                tones,
                lang_ids,
                bert,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )[0][0, 0].data.cpu().float().numpy()
        )

        del x_tst, tones, lang_ids, bert, x_tst_lengths, speakers
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return audio


def generate_tts_audio(
        text: str,
        sid: str,
        audio_save_filename: str,
        config: Vits2Config,
        net_g: Optional[SynthesizerTrn] = None,
        sdp_ratio: Optional[float] = 0.2,
        noise_scale: Optional[float] = 0.5,
        noise_scale_w: Optional[float] = 1.0,
        length_scale: Optional[float] = 1.0,
        language: Optional[str] = "ZH",
        device: Literal['cuda:0', 'cpu'] = "cuda:0",
        model_path: str = None,
) -> str:
    """生成 TTS 合成语音"""
    if not os.path.exists("audio/tts"):
        os.mkdir("audio/tts")

    if not os.path.exists(f"audio/tts/{sid}"):
        os.mkdir(f"audio/tts/{sid}")

    if os.path.exists(os.path.join(f"audio/tts/{sid}", audio_save_filename)):
        print("当前文件夹下已经存在同名文件")
        audio_basename = os.path.basename(audio_save_filename).split(".wav")[0]
        audio_save_filename = f"{audio_basename}_{datetime.now():%Y%m%d%H%M%S}.wav"

    if model_path is not None:
        net_g = get_net_g(model_path, config, device)

    if net_g is None:
        raise ValueError("请传入 TTS 模型")

    data_cfg = config.data_cfg
    with torch.no_grad():
        audio = infer(
            text, sdp_ratio, noise_scale, noise_scale_w, length_scale, sid,
            language, config.data_cfg, net_g, device
        )

        audio_save_path = os.path.join(f"audio/tts/{sid}", audio_save_filename)
        wavfile.write(audio_save_path, data_cfg.sampling_rate, audio)
        print(f"TTS 合成语音成功，文件路径: {audio_save_path}")

    return audio_save_path
