import numpy as np

from utils.datasets import EnToPorDataset
from utils.models import Translator
from utils.metrics import jaccard_similarity

import torch
from torch.utils.data import random_split

from utils.data_preprocessor import (
    get_encoder_decoder,
    load_data,
    make_maps,
    clean_data,
)
from utils.sampling import decode, generate_sample, average_overlap
from utils.configs import EnglishToPortugueseConfig

device = "cuda" if torch.cuda.is_available() else "cpu"

UNITS = EnglishToPortugueseConfig.units

dataset = EnToPorDataset("data/por-eng/por.txt")
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

model = Translator(UNITS, dataset.en_vocab_size, dataset.por_vocab_size)
model.load_state_dict(torch.load("translator.pt"))
model = model.to(device)
model.train(False)

por_data, en_data = load_data("data/por-eng/por.txt")
por_stoi, por_itos = make_maps(por_data)
en_stoi, en_itos = make_maps(en_data)

por_encoder, por_decoder = get_encoder_decoder(por_stoi, por_itos)
en_encoder, en_decoder = get_encoder_decoder(en_stoi, en_itos)

train_dataset, val_dataset = random_split(
    dataset, [train_size, val_size], generator=torch.Generator().manual_seed(1234)
)

text = "Birds fly south in the winter."
text = clean_data(np.array([[text]])).item()
text = torch.tensor(list(map(en_encoder, text.split()))).unsqueeze(0).to(device)

samples, logits = generate_sample(
    model=model, text=text, temperature=0.4, n_samples=30, device=device
)

scores = average_overlap(samples=samples, similarity_fn=jaccard_similarity)

best_translation_index = max(scores, key=scores.get)

final_translation = decode(por_decoder, samples[best_translation_index])

print(f"Translation: {final_translation}")
print(f"Logit: {logits[best_translation_index]}")
