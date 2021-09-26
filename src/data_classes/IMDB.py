from torch.utils.data import Dataset
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch
from torch.nn.utils.rnn import pad_sequence


class IMDBClass(Dataset):
    """IMDB dataset."""

    def __init__(self, train=True, root_dir="./data"):
        self.imdbs_train, self.imdbs_test= IMDB()
        self.root_dir = root_dir

        # Build vocabulary
        tokenizer = get_tokenizer('basic_english')
        def yield_tokens(data_iter):
            for _, text in data_iter:
                yield tokenizer(text)
        self.vocab = build_vocab_from_iterator(yield_tokens(self.imdbs_train), specials=["<unk>"])
        self.vocab.set_default_index(self.vocab["<unk>"])

        # Preprocess data
        self.imdbs_train, self.imdbs_test= IMDB() # iterator has been consumed
        text_pipeline = lambda x: self.vocab(tokenizer(x))
        label_pipeline = lambda x: 1 if x == 'pos' else 0

        # Tokenize
        self.data_train = [[torch.tensor(label_pipeline(label), dtype=torch.long), torch.tensor(self.vocab(tokenizer(item)), dtype=torch.long)] for label, item in self.imdbs_train]
        self.data_test = [[torch.tensor(label_pipeline(label), dtype=torch.long), torch.tensor(self.vocab(tokenizer(item)), dtype=torch.long)] for label, item in self.imdbs_test]


    def __len__(self):
        return len(self.data_train)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        labels = self.data_train[idx][0]
        if isinstance(idx, int):
            idx = [idx]
        #     return self.data_train[idx][1], torch.tensor(labels, dtype=torch.long)
        text_list = [self.data_train[i][1] for i in idx]
        padded_text = pad_sequence(text_list, batch_first=True)

        return padded_text, torch.tensor(labels, dtype=torch.long)

    def vocab_size(self):
        return len(self.vocab)