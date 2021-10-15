from pytorch_lightning import LightningDataModule
from torch import LongTensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split
from torchtext.datasets import IMDB

from .IMDB import IMDBClass


def collate_fn(batch):
    x = [item[0] for item in batch]
    lengths = LongTensor(list(map(len, x)))
    x = pad_sequence(x, batch_first=True)
    y = LongTensor([item[1] for item in batch])
    return x, y, lengths

class IMDBLightningDataModule(LightningDataModule):
    def __init__(self, vocab, data_dir=".data", batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.vocab = vocab
        self.batch_size = batch_size
    
    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        IMDB(self.data_dir)

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        if stage == "fit" or stage is None:
            data_full = IMDBClass(root_dir=self.data_dir, train=True, transform=self.vocab)
            self.data_train, self.data_val = random_split(data_full, [22500, 2500])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.data_test = IMDBClass(root_dir=self.data_dir, train=False, transform=self.vocab)
        
    def train_dataloader(self):
        # train_split = Dataset(...)
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)

    def val_dataloader(self):
        # val_split = Dataset(...)
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)

    def test_dataloader(self):
        # test_split = Dataset(...)
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)

