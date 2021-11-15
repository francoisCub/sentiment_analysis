# Model and system definition
from torch import nn
import torch
from math import sqrt


class RNN(nn.Module):
    def __init__(self, type="LSTM", embedding_size=64, hidden_size=100, num_class=2, num_layers=1, attention_type=None, output_layer_type="linear", vocab=None, vectors=None, vocab_size=None):
        super().__init__()
        if vocab is None:
            self.embedding = torch.nn.Embedding(
                vocab_size, embedding_size, padding_idx=0)
        else:
            self.embedding = torch.nn.Embedding.from_pretrained(
                vectors, freeze=True, padding_idx=vocab["<pad>"])

        self.type = type
        if self.type == "LSTM":
            self.rnn = nn.LSTM(embedding_size, hidden_size,
                               num_layers=num_layers, batch_first=True)
        elif self.type == "GRU":
            self.rnn = nn.GRU(embedding_size, hidden_size,
                              num_layers=num_layers, batch_first=True)
        elif self.type == "RNN":
            self.rnn = nn.RNN(embedding_size, hidden_size,
                              num_layers=num_layers, batch_first=True)
        else:
            raise ValueError('RNN type should be in ["RNN", "LSTM", "GRU"]')

        self.attention_type = attention_type
        if self.attention_type is not None and self.attention_type not in ["last_hidden_layer", "self"]:
            raise ValueError(
                'Attention type should be in ["last_hidden_layer", "self"]')
        self.scaling_factor = 1.0 / sqrt(hidden_size)

        self.output_layer_type = output_layer_type
        if self.output_layer_type == "linear":
            self.output = nn.Linear(hidden_size, num_class)
        else:  # MLP
            self.output = nn.Sequential(nn.Linear(hidden_size, hidden_size*2), nn.LeakyReLU(),
                                        nn.Dropout(0.5), nn.Linear(
                                            hidden_size*2, hidden_size), nn.LeakyReLU(),
                                        nn.Linear(hidden_size, num_class))

    def forward(self, x: torch.Tensor, lengths: torch.LongTensor):
        # Embedding
        x = self.embedding(x)
        x = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths=lengths.to("cpu"), enforce_sorted=False, batch_first=True)

        # RNN
        if self.attention_type is None:
            _, hn = self.rnn(x)  # discard output sequence
        else:
            out_sequence, hn = self.rnn(x)

        if self.type == "LSTM":
            hn = hn[0]  # hidden state, not cell state
        hn = hn[-1, :, :]  # last hidden layer

        # Attention
        if self.attention_type is not None:
            out_sequence, _ = torch.nn.utils.rnn.pad_packed_sequence(
                out_sequence, batch_first=True)
            if self.attention_type == "last_hidden_layer":
                # n: batch size
                # L: sequence length
                # H: embedding size

                # last hidden layer attention
                # [n, 1, L] = [n, 1, H] x [n, H, L]
                att_weights = torch.bmm(hn.unsqueeze(
                    1), out_sequence.transpose(1, 2))
                att_weights = torch.nn.functional.softmax(att_weights, dim=-1)
                # [n, 1, H] = [n, 1, L] x [n, L, H]
                hn = torch.bmm(att_weights, out_sequence)
                hn = hn.squeeze(1) * self.scaling_factor

            else:  # self attention first then last hidden layer attention
                # self attetion
                # [n, L, L] = [n, L, H] x [n, H, L]
                att_weights = torch.bmm(
                    out_sequence, out_sequence.transpose(1, 2))
                att_weights = torch.nn.functional.softmax(att_weights, dim=-1)
                # [n, L, H] = [n, L, L] x [n, L, H]
                att_out_sequence = torch.bmm(att_weights, out_sequence)

                # last hidden layer attention
                # [n, 1, L] = [n, 1, H] x [n, H, L]
                att_weights = torch.bmm(hn.unsqueeze(
                    1), att_out_sequence.transpose(1, 2))
                att_weights = torch.nn.functional.softmax(att_weights, dim=-1)
                # [n, 1, H] = [n, 1, L] x [n, L, H]
                hn = torch.bmm(att_weights, att_out_sequence)
                hn = hn.squeeze(1) * self.scaling_factor

        # Output layer
        x = self.output(hn)
        return x
