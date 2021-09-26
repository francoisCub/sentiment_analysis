from torch.optim import Adam
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import torch

class LSTM(LightningModule):
    def __init__(self, vocab_size, embedding_size=64, lstm_hidden_size=100, num_class=2):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, lstm_hidden_size, batch_first=True)
        self.linear = nn.Linear(lstm_hidden_size, num_class)
        self.loss_function = nn.CrossEntropyLoss()
        self.batch_size = 10
        self.learning_rate = 0.01
    
    def forward(self, X: torch.Tensor):
        x = self.embedding(X)
        _, (hn, cn) = self.lstm(x)
        # hn  = hn.view(hn.size(0), -1)
        x = nn.functional.relu(hn[0])#.hnsqueeze(1)
        x = self.linear(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = pad_sequence(x, batch_first=True)
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        self.log("Train loss", loss.detach())
        return loss
           
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-2)
    
    # def train_dataloader(self):
    #     return train_iter
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        return dict(
            test_loss=loss,
            log=dict(
                test_loss=loss
            )
        )
    
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = dict(
            test_loss=avg_loss
        )
        return dict(
            avg_test_loss=avg_loss, 
            log=tensorboard_logs
        )
    
    # def test_dataloader(self):
    #     return test_iter