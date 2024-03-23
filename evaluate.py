from tqdm import tqdm

from utils.datasets import EnToPorDataset
from utils.models import Translator
from torchmetrics.text import BLEUScore
from torchmetrics.text.rouge import ROUGEScore

import torch
from torch.utils.data import DataLoader, random_split

from utils.configs import EnglishToPortugueseConfig

from utils.sampling import translate, decode
from utils.data_preprocessor import (
    get_encoder_decoder,
    load_data,
    make_maps,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

UNITS = EnglishToPortugueseConfig.units
BATCH_SIZE = 1

dataset = EnToPorDataset("data/por-eng/por.txt")
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset, [train_size, val_size], generator=torch.Generator().manual_seed(1234)
)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


por_data, en_data = load_data("data/por-eng/por.txt")
por_stoi, por_itos = make_maps(por_data)
en_stoi, en_itos = make_maps(en_data)

por_encoder, por_decoder = get_encoder_decoder(por_stoi, por_itos)
en_encoder, en_decoder = get_encoder_decoder(en_stoi, en_itos)


model = Translator(UNITS, dataset.en_vocab_size, dataset.por_vocab_size)
model.load_state_dict(torch.load("translator.pt"))
model = model.to(device)
model.train(False)


bleu = BLEUScore(n_gram=2)
rouge = ROUGEScore()

for data in tqdm(val_loader):

    context, _, target_out = data
    context = context.to(device)

    tokens, logits = translate(model=model, text=context, device=device)

    r = rouge(decode(por_decoder, tokens), [decode(por_decoder, target_out.numpy()[0])])
    b = bleu(decode(por_decoder, tokens), [decode(por_decoder, target_out.numpy()[0])])

    # print(decode(por_decoder, tokens))
    # print(decode(por_decoder, target_out.numpy()[0]))

final_rouge = rouge.compute()
final_blue = bleu.compute()

print(final_rouge)
print(final_blue)
