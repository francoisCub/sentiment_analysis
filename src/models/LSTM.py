# Model and system definition
from torch.optim import Adam
from torch import nn
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

class LSTM(nn.Module):
    def __init__(self, vocab_size=None, embedding_size=64, lstm_hidden_size=100, num_class=2, batch_size=32, learning_rate=0.001, vocab=None, vectors=None):
        super().__init__()
        if vocab is None:
            self.embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        else:
            self.embedding = torch.nn.Embedding.from_pretrained(vectors.vectors, freeze=True, padding_idx=vocab["<pad>"])
        self.lstm = nn.LSTM(embedding_size, lstm_hidden_size, batch_first=True)
        self.linear = nn.Linear(lstm_hidden_size, num_class)
        self.loss_function = nn.CrossEntropyLoss()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lstm_hidden_size = lstm_hidden_size
    
    def forward(self, x: torch.Tensor, lengths: torch.LongTensor):
        x = self.embedding(x)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths=lengths.to("cpu"), enforce_sorted=False, batch_first=True)
        _, (hn, _) = self.lstm(x)
        hn = hn[-1,:,:]
        x = self.linear(hn)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y, lengths = batch
        y_hat = self(x, lengths)
        loss = self.loss_function(y_hat, y)
        self.log("Train Loss", loss.detach())
        return loss
           
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-2)
        
    def test_step(self, batch, batch_idx):
        x, y, lengths = batch
        y_hat = self(x, lengths)
        loss = self.loss_function(y_hat, y)
        labels_hat = torch.argmax(y_hat, dim=1)
        test_acc = torch.sum(labels_hat == y).item() / (len(y) * 1.0)
        return self.log_dict({'Test Loss': loss, 'Test Acc': test_acc})
    
    def train_dataloader(self):
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn) # collate_fn=collate_fn

    
    def test_dataloader(self):
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
