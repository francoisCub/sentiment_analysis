# Model and system definition
import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim import Adam

from .RNN import RNN

# from torch.utils.data import DataLoader



class LightningClassifier(LightningModule):
    def __init__(self, model_type="LSTM", embedding_size=64, hidden_size=100, num_class=2,  num_layers=1, learning_rate=0.001, vocab=None, vectors=None, data_dir=".data"):
        super().__init__()
        self.model_type = model_type
        if model_type == "LSTM":
            self.model = RNN(embedding_size=embedding_size, lstm_hidden_size=hidden_size, num_class=num_class, num_layers=num_layers, vocab=vocab, vectors=vectors)
        else:
            raise ValueError("Not immplemented: choose between: LSTM or GRU")
        self.learning_rate = learning_rate
        self.loss_function = nn.CrossEntropyLoss()
        self.data_dir = data_dir
    
    def forward(self, x: torch.Tensor, lengths: torch.LongTensor):
        return self.model(x, lengths)
    
    def training_step(self, batch, batch_idx):
        x, y, lengths = batch
        y_hat = self.model(x, lengths)
        loss = self.loss_function(y_hat, y)
        self.log("Train Loss", loss.detach())
        return loss
           
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)
        
    def test_step(self, batch, batch_idx):
        x, y, lengths = batch
        y_hat = self.model(x, lengths)
        loss = self.loss_function(y_hat, y)
        labels_hat = torch.argmax(y_hat, dim=1)
        test_acc = torch.sum(labels_hat == y).item() / (len(y) * 1.0)
        return self.log_dict({'Test Loss': loss, 'Test Acc': test_acc})

    def validation_step(self, batch, batch_idx):
        x, y, lengths = batch
        y_hat = self.model(x, lengths)
        loss = self.loss_function(y_hat, y)
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = torch.sum(labels_hat == y).item() / (len(y) * 1.0)
        return self.log_dict({'Val Loss': loss, 'Val Acc': val_acc})
    
    # def train_dataloader(self):
    #     return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn) # collate_fn=collate_fn

    
    # def test_dataloader(self):
    #     return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
