import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, units, vocab_size):
        super().__init__()
        self.units = units
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, units, padding_idx=0)
        self.biLSTM = nn.LSTM(units, units, bidirectional=True, batch_first=True)

    def forward(self, context):

        x = self.embedding(context)

        output, (h_n, c_n) = self.biLSTM(x)

        fwd_out = output[:, :, :256]
        bwd_out = output[:, :, 256:]

        output = fwd_out + bwd_out

        h_n = torch.sum(h_n, dim=0)
        c_n = torch.sum(c_n, dim=0)

        return output, h_n, c_n


class CrossAttention(nn.Module):
    def __init__(self, units=256):
        super().__init__()

        self.mha = nn.MultiheadAttention(embed_dim=units, num_heads=1, batch_first=True)
        self.layerNorm = nn.LayerNorm(units)

    def forward(self, context, target):

        att_output, _ = self.mha(query=target, value=context, key=context)
        output = att_output + target
        output = self.layerNorm(output)

        print(att_output.shape)

        return output


class Decoder(nn.Module):
    def __init__(self, units, vocab_size):
        super().__init__()
        self.units = units
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, units, padding_idx=0)
        self.preLSTM = nn.LSTM(units, units, batch_first=True)
        self.cross_attention = CrossAttention()
        self.postLSTM = nn.LSTM(units, units, batch_first=True)
        self.linear = nn.Linear(3840, vocab_size)
        self.logSoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, context, target, state=None, return_state=False):

        x = self.embedding(target)

        x, (h_n, c_n) = self.preLSTM(x)

        attention_output = self.cross_attention(context, x)

        x, _ = self.postLSTM(attention_output)

        x = torch.flatten(x, 1)
        x = self.linear(x)
        logits = self.logSoftmax(x)

        if return_state:
            return logits, [h_n, c_n]

        return logits


fake_context = torch.ones((64, 14)).int()
fake_target = torch.ones((64, 15)).int()

emb = nn.Embedding(64, 256)

enc = Encoder(256, 12000)
dec = Decoder(256, 12000)
ca = CrossAttention()

out, h_n, c_n = enc(fake_context)

dec(out, fake_target)
