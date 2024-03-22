import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from utils.trainer import loops
from utils.datasets import EnToPorDataset
from utils.models import Translator

device = "cuda" if torch.cuda.is_available() else "cpu"

UNITS = 256
EPOCHS = 5
BATCH_SIZE = 64
VOCAB_SIZE = 12000

dataset = EnToPorDataset("data/por-eng/por.txt")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset, [train_size, val_size], generator=torch.Generator().manual_seed(1234)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = Translator(UNITS, dataset.en_vocab_size, dataset.por_vocab_size)

optimizer = Adam(params=model.parameters())

loops(model, EPOCHS, train_loader, val_loader, optimizer, device)

model = model.cpu()
torch.save(model.state_dict(), "translator_.pt")
