import torch
from sklearn.metrics import f1_score


class GraphTrainer:
    def __init__(self, model, data, optimizer, criterion):
        self.model = model
        self.data = data
        self.optimizer = optimizer
        self.criterion = criterion

    def train_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()

        out = self.model(self.data.x, self.data.edge_index)

        # Calculate loss only on training nodes
        loss = self.criterion(out[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        out = self.model(self.data.x, self.data.edge_index)
        pred = out.argmax(dim=1)

        y_true = self.data.y[self.data.test_mask].cpu().numpy()
        y_pred = pred[self.data.test_mask].cpu().numpy()

        # Macro F1 accounts for the severe class imbalance in fraud datasets
        f1 = f1_score(y_true, y_pred, average='macro')
        return f1