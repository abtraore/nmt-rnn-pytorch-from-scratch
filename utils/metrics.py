import torch
import torch.nn.functional as F


def mask_loss(y_pred, y_true):

    y_pred = torch.flatten(y_pred, 0, 1)
    y_true = torch.flatten(y_true, 0, 1)

    mask_true = (y_true != 0).float()
    loss = F.cross_entropy(y_pred, y_true, reduction="none")
    loss *= mask_true
    loss = torch.sum(loss) / torch.sum(mask_true)

    return loss


def mask_acc(y_pred, y_true):

    y_pred = torch.flatten(y_pred, 0, 1)
    y_true = torch.flatten(y_true, 0, 1)

    mask_true = (y_true != 0).float()
    acc = (torch.argmax(F.softmax(y_pred, dim=1), 1) == y_true).float()
    acc *= mask_true
    acc = torch.sum(acc) / torch.sum(mask_true)

    return acc


def jaccard_similarity(candidate, reference):

    candidate = set(candidate)
    reference = set(reference)

    inter = candidate.intersection(reference)
    union = candidate.union(reference)

    return len(inter) / len(union)
