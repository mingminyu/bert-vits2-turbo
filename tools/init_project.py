import os
import fire


def create_project_dirs(spk_id: str = "") -> None:
    """创建项目文件夹"""
    fst_dirs = ["audio", "pretrained_models", "config", "data", "models"]

    # 创建项目一级目录
    for fst_dir in fst_dirs:
        if not os.path.exists(fst_dir):
            os.mkdir(fst_dir)

    audio_down_dirs = ["upload", "split", "denoise", "bert", "vad", "tts"]
    for audio_down_dir in audio_down_dirs:
        audio_down_dir_ = os.path.join("audio", audio_down_dir, spk_id)
        if not os.path.exists(audio_down_dir_):
            os.makedirs(audio_down_dir_)


def download_project_pretrained_models(model_id: str = None):
    """下载预训练模型"""
    # TODO: 待完成



def run(
        create_dirs: bool = True,
        spk_id: str = "",
        download_pretrained_models: bool = False,
        pretrained_model_id: str = None,
):
    # 创建项目文件夹
    if create_project_dirs:
        create_project_dirs(spk_id)

    # 下载预训练模型
    if download_pretrained_models:
        download_project_pretrained_models(pretrained_model_id)


if __name__ == '__main__':
    fire.Fire(run)





