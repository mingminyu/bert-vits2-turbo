import os
import glob
import json
import pandas as pd
from sklearn.model_selection import train_test_split

from text.cleaner import clean_text


def generate_training_samples(
    asr_dir: str = "data/asr",
    val_per_spk: int = 4,
    max_val_total: int = 8
):
    """生成训练样本"""
    df = pd.concat(
        [pd.read_csv(file) for file in glob.glob(f"{asr_dir}/*/*.csv")],
        ignore_index=True
    )
    df["text"] = df["text"].str.strip()
    df["cleaned"] = df[["text", "lang"]].apply(lambda s: clean_text(s[0], s[1]), axis=1)

    df["norm_text"] = df['cleaned'].map(lambda s: s[0])
    df["phones"] = df['cleaned'].map(lambda s: " ".join(s[1]))
    df["tones"] = df['cleaned'].map(lambda s: " ".join(map(str, s[2])))
    df["word2ph"] = df['cleaned'].map(lambda s: " ".join(map(str, s[3])))

    if not os.path.exists("data/cleaned"):
        os.makedirs("data/cleaned")

    cleaned_save_path = os.path.join("data/cleaned", "samples_cleaned.csv")
    save_cols = ["audio_id", "spk_id", "lang", "norm_text", "phones", "tones", "word2ph"]
    df[save_cols].to_csv(
        cleaned_save_path,
        index=False, sep="|", header=False, encoding='utf8'
    )

    df_train, df_val = train_test_split(
        df[save_cols], test_size=val_per_spk * df['spk_id'].nunique(), shuffle=True,
        random_state=42, stratify=df['spk_id']
    )

    # TODO: 源代码里面直接使用全部数据作为训练集，其实不太合理
    df[save_cols].to_csv("data/train/train_samples.csv", index=False, sep="|", header=False, encoding='utf8')
    df_val.to_csv("data/train/val_samples.csv", index=False, sep="|", header=False, encoding='utf8')

    config_path = "config/config.json"
    with open(config_path, 'r', encoding='utf8') as f:
        model_config = json.loads(f.read())

    model_config['data']['training_files'] = "data/train/train_samples.csv"
    model_config['data']['validation_files'] = "data/train/val_samples.csv"
    model_config['data']['spk2id'] = dict(zip(df['spk_id'].unique(), range(df['spk_id'].nunique())))
    model_config['data']['n_speakers'] = df['spk_id'].nunique()

    with open(config_path, "w", encoding="utf-8") as fw:
        json.dump(model_config, fw, indent=2, ensure_ascii=False)
