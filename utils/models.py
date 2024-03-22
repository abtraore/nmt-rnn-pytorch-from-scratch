import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, units, vocab_size):
        super().__init__()
        self.units = units
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, units, padding_idx=0)
        self.biLSTM = nn.LSTM(units, units, bidirectional=True, batch_first=True)

    def forward(self, context):

        x = self.embedding(context)

        output, _ = self.biLSTM(x)

        # Sum the features (hidden states) of the biLSTM.
        fwd_out = output[:, :, :256]
        bwd_out = output[:, :, 256:]

        output = fwd_out + bwd_out

        return output


class CrossAttention(nn.Module):
    def __init__(self, units=256):
        super().__init__()

        self.mha = nn.MultiheadAttention(embed_dim=units, num_heads=1, batch_first=True)
        self.layerNorm = nn.LayerNorm(units)

    def forward(self, context, target):

        att_output, _ = self.mha(query=target, value=context, key=context)
        output = att_output + target
        output = self.layerNorm(output)

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
        self.linear = nn.Linear(units, vocab_size)
        self.logSoftmax = nn.LogSoftmax(dim=-1)

    def forward(
        self, context, target, h_0=None, c_0=None, train=True, return_state=False
    ):

        x = self.embedding(target)

        # Initialize the hidden state and the cell state when not training (sampling).
        if train == False:
            x, (h_n, c_n) = self.preLSTM(x, (h_0, c_0))
        else:
            x, _ = self.preLSTM(x)

        attention_output = self.cross_attention(context, x)

        x, _ = self.postLSTM(attention_output)

        x = self.linear(x)
        logits = self.logSoftmax(x)

        if return_state:
            return logits, [h_n, c_n]

        return logits


class Translator(nn.Module):
    def __init__(self, units, en_vocab_size, por_vocab):
        super().__init__()
        self.encoder = Encoder(units=units, vocab_size=en_vocab_size)
        self.decoder = Decoder(units=units, vocab_size=por_vocab)

    def forward(self, context, target):
        x = self.encoder(context)
        x = self.decoder(x, target)
        return x
