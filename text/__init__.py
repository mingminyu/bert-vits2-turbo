from text.symbols import *


_symbol_to_id = {s: i for i, s in enumerate(symbols)}


def cleaned_text_to_sequence(cleaned_text, tones, language: str):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    """
    phones = [_symbol_to_id[symbol] for symbol in cleaned_text]
    tone_start = language_tone_start_map[language]
    tones = [i + tone_start for i in tones]
    lang_id = language_id_map[language]
    lang_ids = [lang_id for i in phones]
    return phones, tones, lang_ids


def get_bert(norm_text, word2ph, language):
    from text.chinese_bert import get_bert_feature as zh_bert
    from text.english_bert_mock import get_bert_feature as en_bert
    lang_bert_func_map = {'ZH': zh_bert, 'EN': en_bert}
    bert = lang_bert_func_map[language](norm_text, word2ph)
    return bert
