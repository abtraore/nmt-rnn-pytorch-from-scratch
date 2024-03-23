import torch
import numpy as np

from .configs import EnglishToPortugueseConfig

SOS_ID = EnglishToPortugueseConfig.sos_id
EOS_ID = EnglishToPortugueseConfig.eos_id
UNITS = EnglishToPortugueseConfig.units


def translate(model, text, max_length=50, temperature=0.0, device="cpu"):

    tokens = []
    logits_list = []
    context = model.encoder(text)

    h_0 = torch.zeros((1, 1, UNITS)).to(device)
    c_0 = torch.zeros((1, 1, UNITS)).to(device)
    next_token = torch.ones((SOS_ID, SOS_ID), dtype=torch.long).to(device)

    for _ in range(max_length):
        with torch.no_grad():
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

            if temperature == 0.0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                next_token = torch.multinomial(
                    torch.exp(logits), num_samples=1, replacement=True
                )

            logits_list.append(logits[0][next_token[0][0]].cpu().numpy().item())
            tokens.append(next_token[0][0].cpu().numpy().item())

            if tokens[-1] == EOS_ID:
                break

    return tokens, np.mean(logits_list)


def decode(decoder, tokens):

    return " ".join(
        list(filter(lambda x: x != "" and x != "[EOS]", list(map(decoder, tokens))))
    ).strip()


def generate_sample(model, text, temperature=0.4, n_samples=2, device="cpu"):

    samples = []
    logits = []
    for _ in range(n_samples):
        tokens, logit = translate(
            model, text=text, temperature=temperature, device=device
        )

        samples.append(tokens)
        logits.append(logit)

    return samples, logits


def average_overlap(samples, similarity_fn):
    scores = {}

    for index_candidate, candidate in enumerate(samples):
        overlap = 0

        for index_reference, reference in enumerate(samples):
            if index_candidate == index_reference:
                continue

            overlap += similarity_fn(candidate, reference)

        score = overlap / (len(samples) - 1)
        score = round(score, 3)

        scores[index_candidate] = score

    return scores
