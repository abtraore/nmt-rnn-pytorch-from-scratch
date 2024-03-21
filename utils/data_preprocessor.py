from pathlib import Path
import numpy as np
import unicodedata
import re

import torch


MAX_VOCAB_SIZE = 12000

np.random.seed(1234)
torch.manual_seed(1234)

PATH = "data/por-eng/por.txt"


def load_data(path: Path, encoding: str = "utf-8"):

    path = Path(path)
    text = path.read_text(encoding=encoding)

    lines = text.splitlines()
    pairs = [line.split("\t") for line in lines]

    context = np.array([context for _, context, _ in pairs])
    target = np.array([target for target, _, _ in pairs])

    return context, target


# TODO: Check if it is correct
def clean_data(data: np.ndarray):

    normalize_fn = lambda x: unicodedata.normalize("NFKD", x)
    v_normalize_fn = np.vectorize(normalize_fn)

    lower_fn = lambda x: x.lower()
    v_lower_fn = np.vectorize(lower_fn)

    strip_fn = lambda x: x.strip()
    v_strip_fn = np.vectorize(strip_fn)

    regex_1_fn = lambda x: re.sub("[^ a-z.?!,¿]", "", x)
    v_regex_1_fn = np.vectorize(regex_1_fn)

    regex_2_fn = lambda x: re.sub("[.?!,¿]", r" \g<0> ", x)
    v_regex_2_fn = np.vectorize(regex_2_fn)

    surround_fn = lambda x: f"[SOS] {x} [EOS]"
    v_surround_fn = np.vectorize(surround_fn)

    clean_data = v_normalize_fn(data)
    clean_data = v_lower_fn(clean_data)
    clean_data = v_regex_1_fn(clean_data)
    clean_data = v_regex_2_fn(clean_data)
    clean_data = v_strip_fn(clean_data)
    clean_data = v_surround_fn(clean_data)

    return clean_data


def make_maps(data):

    stoi = {}
    stoi["[SOS]"] = 0
    stoi["[EOS]"] = 1
    stoi["[UNK]"] = 3
    stoi[" "] = 4

    idx = 0
    for _, line in enumerate(data):
        clean_words = [words.strip() for words in line.split()]

        for word in clean_words:

            if word not in stoi:
                stoi[word] = idx + 5
                idx += 1

    itos = {v: k for k, v in stoi.items()}

    stoi = dict(list(stoi.items())[:MAX_VOCAB_SIZE])
    itos = dict(list(itos.items())[:MAX_VOCAB_SIZE])

    return stoi, itos


def get_encoder_decoder(stoi, itos):

    encoder = lambda x: stoi[x]
    decoder = lambda x: itos[x]

    return encoder, decoder
