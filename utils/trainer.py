import torch

from tqdm import tqdm

from .metrics import mask_loss, mask_acc


def loops(model, epochs, train_loader, val_loader, optimizer, scheduler, device):

    model = model.to(device)

    for epoch in tqdm(range(epochs), desc="Epochs", leave=False):

        train_total_loss = 0.0
        train_total_acc = 0.0
        model.train(True)
        for _, data in enumerate(tqdm(train_loader, desc="Training", leave=False)):

            context, target_in, target_out = data

            context = context.to(device)
            target_in = target_in.to(device)
            target_out = target_out.to(device)

            optimizer.zero_grad()

            out = model(context, target_in)

            loss = mask_loss(out, target_out)
            acc = mask_acc(out, target_out)

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
            for data in tqdm(val_loader, desc="Validating", leave=False):

                context, target_in, target_out = data

                context = context.to(device)
                target_in = target_in.to(device)
                target_out = target_out.to(device)

                out = model(context, target_in)

                loss = mask_loss(out, target_out)
                acc = mask_acc(out, target_out)

                val_total_loss += loss.item()
                val_total_acc += acc.item()

        val_total_loss = val_total_loss / len(val_loader)
        val_total_acc = val_total_acc / len(val_loader)

        scheduler.step(val_total_loss)

        print(
            f" Epoch: {epoch+1} | Train Loss: {train_total_loss:.4} | Train Acc: {train_total_acc:.4}"
        )
        print(
            f" Epoch: {epoch+1} | Val Loss: {val_total_loss:.4} | Val Acc: {val_total_acc:.4}\n"
        )
