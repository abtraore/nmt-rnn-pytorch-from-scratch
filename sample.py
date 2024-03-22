import json
import numpy as np

from utils.datasets import EnToPorDataset
from utils.models import Translator
import torch

from torch.utils.data import DataLoader, random_split

from utils.data_preprocessor import (
    get_encoder_decoder,
    load_data,
    make_maps,
    clean_data,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


SOS_ID = 1
EOS_ID = 2
UNITS = 256
BATCH_SIZE = 64
VOCAB_SIZE = 12000


dataset = EnToPorDataset("data/por-eng/por.txt")
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

model = Translator(UNITS, dataset.en_vocab_size, dataset.por_vocab_size)
model.load_state_dict(torch.load("translator_.pt"))
model = model.to(device)


por_data, en_data = load_data("data/por-eng/por.txt")
por_stoi, por_itos = make_maps(por_data)
en_stoi, en_itos = make_maps(en_data)


por_encoder, por_decoder = get_encoder_decoder(por_stoi, por_itos)
en_encoder, en_decoder = get_encoder_decoder(en_stoi, en_itos)


# Split the dataset
train_dataset, val_dataset = random_split(
    dataset, [train_size, val_size], generator=torch.Generator().manual_seed(1234)
)


text = "I love languages"
text = clean_data(np.array([[text]])).item()
text = torch.tensor(list(map(en_encoder, text.split()))).unsqueeze(0).to(device)
print(text)


def translate(model, text, max_length=50, device="cpu"):

    tokens = []
    context = model.encoder(text)

    h_0 = torch.zeros((1, 1, UNITS)).to(device)
    c_0 = torch.zeros((1, 1, UNITS)).to(device)
    next_token = torch.ones((SOS_ID, SOS_ID), dtype=torch.long).to(device)

    for _ in range(max_length):

        logits, states = model.decoder(
            context,
            next_token,
            h_0=h_0,
            c_0=c_0,
            train=False,
            return_state=True,
        )

        h_0, c_0 = states

        logits = logits[:, -1, :]

        next_token = torch.argmax(logits, dim=-1, keepdim=True)

        tokens.append(next_token[0][0].cpu().numpy().item())

        if tokens[-1] == EOS_ID:
            break

    token = " ".join(
        list(filter(lambda x: x != "" and x != "[EOS]", list(map(por_decoder, tokens))))
    ).strip()

    # true = " ".join(
    #     list(
    #         filter(
    #             lambda x: x != "" and x != "[EOS]",
    #             list(map(por_decoder, true_target.cpu().numpy())),
    #         )
    #     )
    # ).strip()

    print(f"Prediction: {token}")

    return tokens


tokens = translate(model, text=text, device=device)
