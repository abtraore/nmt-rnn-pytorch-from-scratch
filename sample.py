from utils.datasets import EnToPorDataset
from utils.models import Translator
from torch.optim import Adam
import torch.nn.functional as F
import torch
from tqdm import tqdm

from torch.utils.data import DataLoader, random_split

from utils.data_preprocessor import get_encoder_decoder, load_data, make_maps

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

SOS_ID = 1
EOS_ID = 2
UNITS = 256
BATCH_SIZE = 64
VOCAB_SIZE = 12000

model = Translator(UNITS, VOCAB_SIZE)
model.load_state_dict(torch.load("translator.pt"))
model = model.to(device)

dataset = EnToPorDataset("data/por-eng/por.txt")
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size


por_data, en_data = load_data("data/por-eng/por.txt")
por_stoi, por_itos = make_maps(por_data)
encoder, decoder = get_encoder_decoder(por_stoi, por_itos)


# Split the dataset
train_dataset, val_dataset = random_split(
    dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

text = torch.unsqueeze(val_dataset.__getitem__(0)[0], 0)
true_target = val_dataset.__getitem__(0)[2]


def translate(model, text, true_target, max_length=50):

    tokens = []

    context, _ = model.encoder(text)

    h_0 = torch.zeros((1, 1, UNITS))
    c_0 = torch.zeros((1, 1, UNITS))
    next_token = torch.ones((SOS_ID, SOS_ID), dtype=torch.long)

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

        tokens.append(next_token[0][0].numpy().item())

        if tokens[-1] == EOS_ID:
            break

    token = " ".join(
        list(filter(lambda x: x != "" and x != "[EOS]", list(map(decoder, tokens))))
    ).strip()

    true = " ".join(
        list(
            filter(
                lambda x: x != "" and x != "[EOS]",
                list(map(decoder, true_target.numpy())),
            )
        )
    ).strip()

    print(f"Prediction: {token}")
    print(f"True: {true}")

    return tokens


tokens = translate(model, text=text, true_target=true_target)
