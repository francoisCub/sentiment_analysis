from pytorch_lightning import LightningDataModule
from torch import LongTensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split
from torchtext.datasets import IMDB, YelpReviewFull

from .IMDB import IMDBClass
from .YelpReview import YelpReviewClass

def get_collate_fn(trunc=-1):
    def collate_fn(batch):
        x = [item[0] for item in batch]
        if trunc > 0:
            x = [sentence[0:trunc] for sentence in x]
        lengths = LongTensor(list(map(len, x)))
        x = pad_sequence(x, batch_first=True)
        y = LongTensor([item[1] for item in batch])
        return x, y, lengths
    return collate_fn

class TextLightningDataModule(LightningDataModule):
    def __init__(self, vocab, data_dir=".data", batch_size=32, dataset="IMDB", num_workers=0, trunc=-1):
        super().__init__()
        self.data_dir = data_dir
        self.vocab = vocab
        self.batch_size = batch_size
        self.dataset = dataset
        self.num_workers = num_workers
        self.trunc = trunc
        if self.dataset not in ["IMDB", "Yelp"]:
            raise ValueError('dataset should be in ["IMDB", "Yelp"]')
    
    def prepare_data(self):
        if self.dataset == "IMDB":
            IMDB(self.data_dir)
        else:
            YelpReviewFull(self.data_dir)

    def setup(self, stage):
        if stage == "fit" or stage is None:
            if self.dataset == "IMDB":
                data_full = IMDBClass(root_dir=self.data_dir, train=True, transform=self.vocab) 
                self.data_train, self.data_val = random_split(data_full, [22500, 2500])
            else:
                data_full = YelpReviewClass(root_dir=self.data_dir, train=True, transform=self.vocab)
                self.data_train, self.data_val = random_split(data_full, [585000, 65000])
            

        if stage == "test" or stage is None:
            self.data_test = IMDBClass(root_dir=self.data_dir, train=False, transform=self.vocab) if self.dataset == "IMDB" else YelpReviewClass(root_dir=self.data_dir, train=False, transform=self.vocab)
        
    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, collate_fn=get_collate_fn(self.trunc), num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, collate_fn=get_collate_fn(self.trunc), num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False, collate_fn=get_collate_fn(self.trunc), num_workers=self.num_workers)

