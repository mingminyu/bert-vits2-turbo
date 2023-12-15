import os
import whisper
import wget
from loguru import logger


def create_project_dirs(spk_id: str = "") -> None:
    """创建项目文件夹"""
    fst_dirs = ["audio", "pretrained_models", "config", "data", "models"]

    # 创建项目一级目录
    for fst_dir in fst_dirs:
        if not os.path.exists(fst_dir):
            os.mkdir(fst_dir)

    # 创建 audio 二级子目录
    audio_down_dirs = ["upload", "split", "denoise", "segments", "vad", "tts"]
    for audio_down_dir in audio_down_dirs:
        audio_down_dir_ = os.path.join("audio", audio_down_dir, spk_id)
        if not os.path.exists(audio_down_dir_):
            os.makedirs(audio_down_dir_)

    # 创建 data 二级子目录
    data_down_dirs = ["asr", "trans", "train", "cleaned"]
    for data_down_dir in data_down_dirs:
        data_down_dir_ = os.path.join("data", data_down_dir, spk_id)
        if not os.path.exists(data_down_dir_):
            os.makedirs(data_down_dir_)

    if not os.path.exists("pretrained_models/whisper"):
        os.makedirs("pretrained_models/whisper")


def download_pretrained_models(
    whisper_size: str = "medium",
):
    """下载预训练模型
    只适用于可以联网环境下
    """
    # 下载 chinese-roberta-wwm-ext-large
    logger.info("下载模型: chinese-roberta-wwm-ext-large")

    roberta_model_urls = [
        "https://huggingface.co/hfl/chinese-roberta-wwm-ext-large/resolve/main/tf_model.h5",
        "https://huggingface.co/hfl/chinese-roberta-wwm-ext-large/resolve/main/flax_model.msgpack",
        "https://huggingface.co/hfl/chinese-roberta-wwm-ext-large/resolve/main/pytorch_model.bin",
        "https://huggingface.co/hfl/chinese-roberta-wwm-ext-large/resolve/main/config.json"
    ]
    _ = [
        wget.download(url, "pretrained_models/chinese-roberta-wwm-ext-large")
        for url in roberta_model_urls
    ]
    logger.info("下载完成: chinese-roberta-wwm-ext-large")

    # 下载 speech_dfsmn_ans_psm_48k_causal
    from modelscope import snapshot_download

    logger.info("下载模型: speech_dfsmn_ans_psm_48k_causal")
    snapshot_download(
        model_id="damo/speech_dfsmn_ans_psm_48k_causal",
        cache_dir="./pretrained_models"
    )
    logger.info("下载完成: speech_dfsmn_ans_psm_48k_causal")

    # 下载 speech_fsmn_vad_zh-cn-16k-common-pytorch
    logger.info("下载模型: speech_fsmn_vad_zh-cn-16k-common-pytorch")
    snapshot_download(
        model_id="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        cache_dir="./pretrained_models"
    )
    logger.info("下载完成: speech_fsmn_vad_zh-cn-16k-common-pytorch")

    # 下载 Whisper 模型
    logger.info(f"下载模型: whisper {whisper_size}")

    if not os.path.exists(f"pretrained_models/whisper/{whisper_size}.pt"):
        _ = whisper.load_model(
            whisper_size, download_root="pretrained_models/whisper"
        )
    logger.info(f"下载完成: whisper {whisper_size}")
    logger.warning("下载完请重启下 notebook，释放掉机器资源")

    # 下载底模
    if not os.path.exists(f"pretrained_models/vits2_base_model"):
        os.makedirs("pretrained_models/vits2_base_model")

    vits_base_model_files = [
        "https://huggingface.co/Erythrocyte/bert-vits2_base_model/resolve/main/DUR_0.pth",
        "https://huggingface.co/Erythrocyte/bert-vits2_base_model/resolve/main/D_0.pth",
        "https://huggingface.co/Erythrocyte/bert-vits2_base_model/resolve/main/G_0.pth",
    ]
    logger.info(f"下载 Vits2 底模")
    _ = [
        wget.download(url, "pretrained_models/vits2_base_model")
        for url in vits_base_model_files
    ]
    logger.info(f"下载 Vits2 底模完成")
