from torch.utils.data import Dataset
from torchtext.datasets import YelpReviewFull
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch


class YelpReviewClass(Dataset):
    """Yelp review full dataset."""

    def __init__(self, train=True, root_dir="./data", transform=None):
        self.yelp_train, self.yelp_test = YelpReviewFull(root_dir)
        self.yelp = self.yelp_train if train else self.yelp_test
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
            self.vocab = build_vocab_from_iterator(yield_tokens(
                self.yelp_train), specials=[self.unknown, self.pad])
            self.vocab.set_default_index(self.vocab[self.unknown])
        else:
            self.vocab = transform

        # Preprocess data
        def text_pipeline(x): return tokenizer(x)  # self.vocab(tokenizer(x))
        def label_pipeline(x): return x-1

        # Tokenize
        self.data = [[torch.tensor(label_pipeline(
            label), dtype=torch.long), text_pipeline(item)] for label, item in self.yelp]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        def text_pipeline(x): return torch.tensor(
            self.vocab(x), dtype=torch.long)

        labels = self.data[idx][0]
        if isinstance(idx, int):
            return text_pipeline(self.data[idx][1]), torch.tensor(labels, dtype=torch.long)
        text_list = [text_pipeline(self.data[i][1]) for i in idx]

        return text_list, torch.tensor(labels, dtype=torch.long)

    def vocab_size(self):
        return len(self.vocab)
