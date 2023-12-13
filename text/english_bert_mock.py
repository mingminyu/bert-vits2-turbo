import torch


def get_bert_feature(norm_text: str, word2ph):
    """这里 norm_text 并未使用，只是为了和 chinese_bert.py 中 get_bert_feature 保持一致，方便传参
    """
    return torch.zeros(1024, sum(word2ph))
