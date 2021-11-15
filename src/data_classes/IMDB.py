import numpy as np
import torch
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import IMDB
from torchtext.vocab import build_vocab_from_iterator


class IMDBClass(Dataset):
    """IMDB dataset."""

    def __init__(self, train=True, root_dir="./data", transform=None):
        self.imdbs_train, self.imdbs_test = IMDB(root_dir)
        self.imdbs = self.imdbs_train if train else self.imdbs_test
        self.root_dir = root_dir
        self.train = train

        self.unknown = "<unk>"
        self.pad = "<pad>"

        def text_pipeline(x): return tokenizer(x)
        def label_pipeline(x): return 1 if x == 'pos' else 0

        # Build vocabulary if needed
        tokenizer = get_tokenizer('basic_english')

        def yield_tokens(data_iter):
            for _, text in data_iter:
                yield tokenizer(text)
        if transform is None and train:
            self.vocab = build_vocab_from_iterator(yield_tokens(
                self.imdbs_train), specials=[self.unknown, self.pad])
            self.vocab.set_default_index(self.vocab[self.unknown])
        else:
            self.vocab = transform

        # Tokenize
        self.data = [[torch.tensor(label_pipeline(label), dtype=torch.long), text_pipeline(
            item)] for label, item in self.imdbs]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        def text_pipeline(x): return torch.tensor(
            self.vocab(x), dtype=torch.long)

        labels = self.data[idx][0]
        if isinstance(idx, int):
            return text_pipeline(self.data[idx][1]), labels
        text_list = [text_pipeline(self.data[i][1]) for i in idx]

        return text_list, torch.tensor(labels, dtype=torch.long)

    def vocab_size(self):
        return len(self.vocab)


class IMDBSentenceClass(Dataset):
    """
        IMDB dataset with pre-encoded sentence.
        see models/encode.ipynb to encode sentence before calling this class.
    """

    def __init__(self, train=True, root_dir="./data", format=None):
        imdbs_train, imdbs_test = IMDB(root_dir)
        imdbs = imdbs_train if train else imdbs_test
        self.root_dir = root_dir
        self.train = train

        self.unknown = "<unk>"
        self.pad = "<pad>"

        def label_pipeline(x): return torch.tensor(
            1, dtype=torch.long) if x == 'pos' else torch.tensor(0, dtype=torch.long)
        self.labels = [label_pipeline(label) for label, _ in imdbs]

        self.format = format
        if self.format == "bert":
            if train:
                self.encoded_text = torch.load(".data/IMDB_bert_train.pt")
            else:
                self.encoded_text = torch.load(".data/IMDB_bert_test.pt")
        elif self.format == "wme":
            if train:
                self.encoded_text = np.load(
                    ".data/IMDB_train_wme_train_300.npy")
                mean = self.encoded_text.mean()
                std = self.encoded_text.std()
                self.encoded_text = (self.encoded_text - mean) / std
            else:
                self.encoded_text = np.load(
                    ".data/IMDB_train_wme_train_300.npy")
                # mean and std computed on train set
                mean = self.encoded_text.mean()
                std = self.encoded_text.std()
                self.encoded_text = np.load(".data/IMDB_test_wme_test_300.npy")
                self.encoded_text = (self.encoded_text - mean) / std

            self.encoded_text = torch.from_numpy(self.encoded_text).float()
        else:
            raise ValueError("Format in 'wme' or 'bert'")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.encoded_text[idx], self.labels[idx]
