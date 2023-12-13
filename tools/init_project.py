import os


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
