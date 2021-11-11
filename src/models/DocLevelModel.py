from torch import nn


class DocLevelModel(nn.Module):
    def __init__(self, vocab=None, vectors=None, dim=300, num_class=2, method="pretrained-average", output_layer_type="linear") -> None:
        super().__init__()
        self.num_class = num_class

        self.method = method
        if self.method not in ["pretrained-average", "from-scratch-average", "wme", "sentence-bert"]:
            raise ValueError(
                'Method should be in ["pretrained-average", "from-scratch-average", "wme", "sentence-bert"]')

        if self.method == "pretrained-average":
            if vocab is None or vectors is None:
                raise ValueError(
                    "pre-trained: vocab and vectors should passed for averaging technique")
            self.embedding = nn.Embedding.from_pretrained(
                vectors, freeze=True, padding_idx=vocab["<pad>"])
        elif self.method == "from-scratch-average":
            if vocab is None:
                raise ValueError(
                    "from-scratch-average: vocab should passed for averaging technique")
            self.embedding = nn.Embedding(
                len(vocab), dim, padding_idx=vocab["<pad>"])
        elif self.method == "sentence-bert":
            if dim != 384:
                raise ValueError("dim of sentence-bert should be 384")
        elif self.method == "wme":
            pass
        else:
            raise NotImplementedError()

        if output_layer_type == "linear":
            self.output_layer = nn.Linear(dim, 2)
        elif output_layer_type == "MLP":
            self.output_layer = nn.Sequential(nn.Linear(dim, 300), nn.ReLU(), nn.Dropout(0.2),
                                              nn.Linear(
                                                  300, 24), nn.LeakyReLU(),
                                              nn.Linear(24, self.num_class))
        else:
            raise NotImplementedError()

    def forward(self, x, length):
        if self.method == "pretrained-average" or self.method == "from-scratch-average":
            x = self.embedding(x)
            x = x.mean(dim=1)

        out = self.output_layer(x)

        return out
