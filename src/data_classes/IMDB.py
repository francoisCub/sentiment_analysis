from torch.utils.data import Dataset
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch
from torch.nn.utils.rnn import pad_sequence


class IMDBClass(Dataset):
    """IMDB dataset."""

    def __init__(self, train=True, root_dir="./data", transform=None):
        self.imdbs_train, self.imdbs_test= IMDB(root_dir)
        self.imdbs = self.imdbs_train if train else self.imdbs_test
        self.root_dir = root_dir
        self.train = train

        self.unknown = "<unk>"
        self.pad = "<pad>"

        # Build vocabulary
        tokenizer = get_tokenizer('basic_english')
        def yield_tokens(data_iter):
            for _, text in data_iter:
                yield tokenizer(text)
        if transform is None and train:
            self.vocab = build_vocab_from_iterator(yield_tokens(self.imdbs_train), specials=[self.unknown, self.pad])
            self.vocab.set_default_index(self.vocab[self.unknown])
        else:
            self.vocab = transform

        # Preprocess data
        # self.imdbs_train, self.imdbs_test = IMDB() # iterator has been consumed
        text_pipeline = lambda x: tokenizer(x) # self.vocab(tokenizer(x))
        label_pipeline = lambda x: 1 if x == 'pos' else 0

        # Tokenize
        self.data = [[torch.tensor(label_pipeline(label), dtype=torch.long),text_pipeline(item)] for label, item in self.imdbs]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        text_pipeline = lambda x: torch.tensor(self.vocab(x), dtype=torch.long)

        labels = self.data[idx][0]
        if isinstance(idx, int):
            return text_pipeline(self.data[idx][1]), torch.tensor(labels, dtype=torch.long)
        text_list = [text_pipeline(self.data[i][1]) for i in idx]

        return text_list, torch.tensor(labels, dtype=torch.long)

    def vocab_size(self):
        return len(self.vocab)