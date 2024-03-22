from utils.datasets import EnToPorDataset
from utils.models import Translator
from torch.optim import Adam
import torch.nn.functional as F
import torch
from tqdm import tqdm

from torch.utils.data import DataLoader, random_split

device = "cuda" if torch.cuda.is_available() else "cpu"


UNITS = 256
EPOCHS = 5
BATCH_SIZE = 64
VOCAB_SIZE = 12000

model = Translator(UNITS, VOCAB_SIZE)
model = model.to(device)

dataset = EnToPorDataset("data/por-eng/por.txt")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# Split the dataset
train_dataset, val_dataset = random_split(
    dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


optimizer = Adam(params=model.parameters())


for epoch in range(EPOCHS):

    train_total_loss = 0.0
    train_total_acc = 0.0
    model.train(True)
    for data in tqdm(train_loader):

        # Target need to be shift to the right.
        context, target_in, target_out = data

        context = context.to(device)
        target_in = target_in.to(device)
        target_out = target_out.to(device)

        flat_target = torch.flatten(target_out, 0, 1)

        optimizer.zero_grad()

        out = model(context, target_in)

        flat_out = torch.flatten(out, 0, 1)

        loss = F.cross_entropy(flat_out, flat_target)

        acc = (
            (torch.argmax(F.softmax(flat_out, dim=1), 1) == flat_target).float().mean()
        )

        loss.backward()
        optimizer.step()

        train_total_loss += loss.item()
        train_total_acc += acc.item()

    train_total_loss = train_total_loss / len(train_loader)
    train_total_acc = train_total_acc / len(train_loader)

    with torch.no_grad():
        model.train(False)
        val_total_loss = 0.0
        val_total_acc = 0.0
        for data in tqdm(val_loader):

            # Target need to be shift to the right.
            context, target_in, target_out = data

            context = context.to(device)
            target_in = target_in.to(device)
            target_out = target_out.to(device)

            flat_target = torch.flatten(target_out, 0, 1)

            optimizer.zero_grad()

            out = model(context, target_in)

            flat_out = torch.flatten(out, 0, 1)

            loss = F.cross_entropy(flat_out, flat_target)

            acc = (
                (torch.argmax(F.softmax(flat_out, dim=1), 1) == flat_target)
                .float()
                .mean()
            )

            val_total_loss += loss.item()
            val_total_acc += acc.item()

    val_total_loss = val_total_loss / len(val_loader)
    val_total_acc = val_total_acc / len(val_loader)

    print(
        f"Epoch: {epoch}| Train Loss: {train_total_loss} | Train Acc: {train_total_acc}"
    )
    print(f"Epoch: {epoch}| Val Loss: {val_total_loss} | Val Acc: {val_total_acc}\n")


model = model.cpu()
torch.save(model.state_dict(), "translator.pt")
