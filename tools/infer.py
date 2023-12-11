import os
import torch
from typing import Literal, Optional
from scipy.io import wavfile
from datetime import datetime

from utils import commons, util



def get_net_g(model_path: str, device: str, hps: HParams):
    """当前版本模型 net_g"""
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to(device)
    _ = net_g.eval()
    _ = util.load_checkpoint(model_path, net_g, None, skip_optimizer=True)
    return net_g



if __name__ == '__main__':
    ...
