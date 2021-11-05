# Model and system definition
import torch
from pytorch_lightning import LightningModule
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
from torch.optim import Adam
from torchmetrics import Accuracy, MetricCollection, Precision, Recall

from .RNN import RNN


class LightningClassifier(LightningModule):
    def __init__(self, model_type="LSTM", embedding_size=64, hidden_size=100, num_class=2,  num_layers=1, learning_rate=0.001, vocab=None, vectors=None, attention_type=None, output_layer_type="linear", advanced_metrics=False):
        super().__init__()
        self.model_type = model_type
        self.model = RNN(embedding_size=embedding_size, hidden_size=hidden_size, num_class=num_class, num_layers=num_layers, vocab=vocab, vectors=vectors, type=model_type, attention_type=attention_type, output_layer_type=output_layer_type)
        self.learning_rate = learning_rate
        self.loss_function = nn.CrossEntropyLoss()
        self.num_class = num_class
        self.advanced_metrics = advanced_metrics
        # self.accuracy = Accuracy(num_classes=num_class)
        # self.compute_precision = Precision(num_classes=num_class)
        # self.recall = Recall(num_classes=num_class)
        # self.f1 = F1(num_classes=num_class)
        metrics = MetricCollection([Accuracy(), Precision(), Recall()])
        # self.val_metrics = metrics.clone(prefix='Val ')
        self.test_metrics = metrics.clone(prefix='tm Test ')
    
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
            y_labels = y_hat.cpu().numpy()
            y_numpy = y.cpu().numpy()
            labels = list(range(self.num_class))
            prec = precision_score(y_labels, y_numpy, zero_division=0, labels=labels)
            rec = recall_score(y_labels, y_numpy, zero_division=0, labels=labels)
            f1 = f1_score(y_labels, y_numpy, zero_division=0, labels=labels)
            self.log_dict({'Test Precision': prec, 'Test Recall': rec, 'Test F1': f1})

        # m = self.test_metrics(y_hat, y)
        # self.log_dict(m)

        return self.log_dict({'Test Loss': loss, 'Test Acc': acc})

    def validation_step(self, batch, batch_idx):
        x, y, lengths = batch
        y_hat = self.model(x, lengths)
        loss = self.loss_function(y_hat, y)
        labels_hat = torch.argmax(y_hat, dim=1)
        acc = torch.sum(labels_hat == y).item() / (len(y) * 1.0)
        return self.log_dict({'Val Loss': loss, 'Val Acc': acc})
    