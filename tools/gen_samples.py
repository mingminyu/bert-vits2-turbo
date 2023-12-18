import os
import glob
import json
import pandas as pd
from sklearn.model_selection import train_test_split

from text.cleaner import clean_text
from utils.config import GenSamplesConfig

"""
对应到原项目中 preprocess_text.py 脚本，因为这个模块仅仅是生成训练样本的，所以我这里做了函数名都做了修改
"""


def generate_training_samples(
    gen_samples_config: GenSamplesConfig
):
    """生成训练样本"""
    df = pd.concat(
        [pd.read_csv(file) for file in glob.glob(f"{gen_samples_config.asr_dir}/*/*.csv")],
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

    save_cols = ["audio_id", "spk_id", "lang", "norm_text", "phones", "tones", "word2ph"]
    df[save_cols].to_csv(
        gen_samples_config.cleaned_path,
        index=False, sep="|", header=False, encoding='utf8'
    )

    test_size = gen_samples_config.val_per_spk * df['spk_id'].nunique()
    df_train, df_val = train_test_split(
        df[save_cols], test_size=test_size, shuffle=True,
        random_state=42, stratify=df['spk_id']
    )

    # 在 BERT-VITS2 1.0 版本中，源代码直接使用全部数据作为训练集，其实不太合理
    # 在 BERT-VITS2 2.x 版本中，这个问题得到的修正
    df[save_cols].to_csv(gen_samples_config.train_path, index=False, sep="|", header=False, encoding='utf8')
    df_val.to_csv(gen_samples_config.val_path, index=False, sep="|", header=False, encoding='utf8')

    with open(gen_samples_config.config_path, 'r', encoding='utf8') as f:
        model_config = json.loads(f.read())

    model_config['data']['training_files'] = gen_samples_config.train_path
    model_config['data']['validation_files'] = gen_samples_config.val_path
    model_config['data']['spk2id'] = dict(zip(df['spk_id'].unique(), range(df['spk_id'].nunique())))
    model_config['data']['n_speakers'] = df['spk_id'].nunique()

    with open(gen_samples_config.config_path, "w", encoding="utf-8") as fw:
        json.dump(model_config, fw, indent=2, ensure_ascii=False)
