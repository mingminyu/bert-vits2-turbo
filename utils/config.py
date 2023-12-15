import os
import yaml
import json
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, computed_field


class ResampleConfig(BaseModel):
    """Resample 配置"""
    sampling_rate: int = Field(default=44100, description="采样率")
    in_dir: str = Field(default=None, description="原始音频的路径")
    out_dir: str = Field(default=None, description="重采样后输出音频的路径")


class GenSamplesConfig(BaseModel):
    """Gen Samples 配置"""
    asr_dir: str = Field(default="data/asr", description="ASR转写数据集文件")
    trans_path: str = Field(default="data/trans", description="转写数据集文件")
    cleaned_path: str = Field(default="data/cleaned/samples_cleaned.csv", description="清洗后的转写数据文件")
    train_path: str = Field(default="data/train/train_samples.csv", description="训练数据集文件")
    val_path: str = Field(default="data/train/val_samples.csv", description="验证数据集文件")
    val_per_spk: int = Field(default=4, descriptionscription="每个人验证多少个样本")
    max_val_total: int = Field(default=8, description="验证集最大数量")
    config_path: str = Field(default="config/config.json", description="模型配置文件路径")


class GenBertConfig(BaseModel):
    """Gen Bert 配置"""
    config_path: str = Field(default="config/config.json", description="模型配置文件路径")
    n_jobs: int = Field(default=2, description="并行数量")
    device: str = Field(default="cpu", description="cpu/cuda")
    # 由于服务器一台机器只有 1 个 GPU，这里默认关闭，相应代码也做进一步删除
    use_multi_device: bool = Field(default=False, description="使用多卡推理")


class EnvVar(BaseModel):
    master_addr: str
    master_port: int


class TrainMSConfig(BaseModel):
    """train_ms 配置"""
    env: List[EnvVar]
    base_model_dir: str
    model_dir: str = Field(default="models", description="训练模型保存路径")
    config_path: str = Field(default="config/config.json", description="模型配置文件路径")


class Vits2Config:
    __slots__ = (
        'resample_cfg', 'gen_samples_cfg', 'bert_gen_cfg', 'train_ms_cfg',
        'train_cfg', 'model_cfg', 'data_cfg',
    )

    def __init__(
            self,
            yaml_cfg_path: str = "config/config.yaml",
            json_cfg_path: str = "config/config.json",
    ):
        if not os.path.isfile(yaml_cfg_path):
            raise FileNotFoundError(f"{yaml_cfg_path} 不存在")

        if not os.path.isfile(json_cfg_path):
            raise FileNotFoundError(f"{json_cfg_path} 不存在")

        with open(yaml_cfg_path, 'r') as f:
            yaml_cfg = yaml.safe_load(f.read())

        with open(json_cfg_path, 'r') as f:
            json_cfg = json.loads(f.read())

        # 项目配置信息
        self.resample_cfg = ResampleConfig(**yaml_cfg["resample"])
        self.gen_samples_cfg = GenSamplesConfig(**yaml_cfg["gen_samples"])
        self.bert_gen_cfg = GenBertConfig(**yaml_cfg["gen_bert"])
        self.train_ms_cfg = TrainMSConfig(**yaml_cfg["train_ms"])

        # 模型训练配置
        self.train_cfg = TrainConfig(**json_cfg["train"])
        self.data_cfg = DataConfig(**json_cfg["data"])
        self.model_cfg = ModelConfig(**json_cfg["model"])


class ModelConfig(BaseModel):
    """对应 config/config.json 文件中 model 部分"""
    use_spk_conditioned_encoder: bool = Field(default=True, description="?")
    use_noise_scaled_mas: bool = Field(default=True, description="")
    use_mel_posterior_encoder: bool = Field(default=False, description="")
    use_duration_discriminator: bool = Field(default=True, description="")
    inter_channels: int = Field(default=192, description="")
    hidden_channels: int = Field(default=192, description="")
    filter_channels: int = Field(default=768, description="")
    n_heads: int = Field(default=2, description="")
    n_layers: int = Field(default=6, description="")
    kernel_size: int = Field(default=3, description="")
    p_dropout: int = Field(default=0.1, description="")
    resblock: int = Field(default="1", description="")
    resblock_kernel_sizes: list[int] = Field(default=[3, 7, 11], description="")
    resblock_dilation_sizes: list[int] = Field(
        default=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        description="")
    upsample_rates: list[int] = Field(default=[8, 8, 2, 2, 2], description="")
    upsample_initial_channel: int = Field(default=512, description="")
    upsample_kernel_sizes: list[int] = Field(default=[16, 16, 8, 2, 2], description="")
    n_layers_q: int = Field(default=3, description="?")
    use_spectral_norm: bool = Field(default=False, description="?")
    gin_channels: int = Field(default=256, description="?")


class TrainConfig(BaseModel):
    """对应 config/config.json 文件中 train 部分"""
    log_interval: int = Field(default=200, description="每隔多少个 step 保存模型")
    eval_interval: int = Field(default=1000, description="?")
    seed: int = Field(default=52, description="随机数种子")
    epochs: int = Field(default=1000, description="训练多少个 epoch")
    learning_rate: float = Field(default=3e-4, description="学习率")
    betas: list[float] = Field(default=[0.8, 0.99], description="")
    eps: float = Field(default=1e-9, description="?")
    batch_size: int = Field(default=12, description="每个批次的训练数据量")
    fp16_run: bool = Field(default=False, description="是否开启 fp16 训练加速")
    lr_decay: float = Field(default=0.999875, description="?")
    segment_size: int = Field(default=16384, description="?")
    init_lr_ratio: int = Field(default=1, description="?")
    warmup_epochs: int = Field(default=0, description="从指定 epoch 热启动")
    c_mel: int = Field(default=45, description="?")
    c_kl: float = Field(default=1.0, description="?")
    keep_ckpts: int = Field(default=5, description="保存近多少个历史模型版本")


class DataConfig(BaseModel):
    """对应 config/config.json 文件中 data 部分"""
    use_mel_posterior_encoder: bool = Field(default=False, description="?")
    training_files: str = Field(default="data/train/train_samples.csv", description="训练数据集路径")
    validation_files: str = Field(default="data/train/val_samples.csv", description="验证数据集路径")
    max_wav_value: float = Field(default=32768.0, description="")
    sampling_rating: int = Field(default=44100, description="采样率")
    filter_length: int = Field(default=2048, description="?")
    hop_length: int = Field(default=512, description="?")
    win_length: int = Field(default=2048, description="?")
    n_mel_channels: int = Field(default=128, description="?")
    mel_fmin: float = Field(default=0.0, description="?")
    mel_fmax: float = Field(default=None, description="?")
    add_blank: bool = Field(default=True, description="?")
    n_speakers: int = Field(default=1, description="角色数量")
    spk2id: Dict[str, int] = Field(default={}, description="角色ID映射表")


