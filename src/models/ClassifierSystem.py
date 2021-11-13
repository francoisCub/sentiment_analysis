# Model and system definition
import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim import Adam
# from torchmetrics import Accuracy

from .RNN import RNN
from .DocLevelModel import DocLevelModel


class LightningClassifier(LightningModule):
    def __init__(self, embedding_level="word", model_type="LSTM", embedding_size=64, hidden_size=100, num_class=2,  num_layers=1, learning_rate=0.001, vocab=None, vectors=None, attention_type=None, output_layer_type="linear", advanced_metrics=False):
        super().__init__()
        self.model_type = model_type
        if embedding_level == "word":
            self.model = RNN(embedding_size=embedding_size, hidden_size=hidden_size, num_class=num_class, num_layers=num_layers, vocab=vocab, vectors=vectors, type=model_type, attention_type=attention_type, output_layer_type=output_layer_type)
        elif embedding_level == "sentence":
            self.model = DocLevelModel(vocab=vocab, vectors=vectors, dim=embedding_size, num_class=num_class, method=model_type, output_layer_type=output_layer_type)
        self.learning_rate = learning_rate
        self.loss_function = nn.CrossEntropyLoss()
        self.num_class = num_class
        self.advanced_metrics = advanced_metrics
    
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
        y_hat = torch.argmax(y_hat, dim=1)
        labels_hat = y_hat
        acc = torch.sum(labels_hat == y).item() / (len(y) * 1.0)
        if self.num_class == 2 and self.advanced_metrics:
            # Precision
            true_pos = torch.sum(labels_hat * y).item()
            pred_pos = torch.sum(labels_hat)
            if pred_pos == 0:
                prec = 0.5
            else:
                prec = true_pos / (pred_pos)
            # Recall
            pos = torch.sum(y)
            if pos == 0:
                rec = 0.5
            else:
                rec = true_pos / (pos)
            self.log_dict({'Test Precision': prec, 'Test Recall': rec})

        return self.log_dict({'Test Loss': loss, 'Test Acc': acc})

    def validation_step(self, batch, batch_idx):
        x, y, lengths = batch
        y_hat = self.model(x, lengths)
        loss = self.loss_function(y_hat, y)
        labels_hat = torch.argmax(y_hat, dim=1)
        acc = torch.sum(labels_hat == y).item() / (len(y) * 1.0)
        return self.log_dict({'Val Loss': loss, 'Val Acc': acc})
    