import os
import torch
from utils import commons
from text import cleaned_text_to_sequence, get_bert
# from utils import util
# from torch.utils.data import DataLoader
# from multiprocessing import Pool
# from utils.data_utils import TextAudioSpeakerLoader, TextAudioSpeakerCollate
# from tqdm import tqdm
# import warnings




def process_line(line: str, add_blank: bool = True):
    wav_path, spk, language_str, text, phones, tone, word2ph = line.strip().split("|")
    phone = phones.split(" ")
    tone = [int(i) for i in tone.split(" ")]
    word2ph = [int(i) for i in word2ph.split(" ")]
    w2pho = [i for i in word2ph]
    word2ph = [i for i in word2ph]
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    if add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)

        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1

    bert_path = wav_path.replace(".wav", ".bert.pt")

    if not os.path.exists(bert_path):
        bert = get_bert(text, word2ph, language_str)
        assert bert.shape[-1] == len(phone)
        torch.save(bert, bert_path)
