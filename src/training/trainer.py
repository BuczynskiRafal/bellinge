import torch
import time
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, checkpoint_dir):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for X_batch, y_batch in self.train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

                outputs_prob = torch.sigmoid(outputs)

                total_loss += loss.item()
                all_preds.extend(outputs_prob.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        return total_loss / len(self.val_loader), all_preds, all_labels

    def train(self, epochs, checkpoint_interval=5):
        training_log = []

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            train_loss = self.train_epoch()
            val_loss, val_preds, val_labels = self.validate()

            val_preds_binary = (np.array(val_preds) >= 0.5).astype(int)
            val_labels = np.array(val_labels)

            val_acc = accuracy_score(val_labels, val_preds_binary)
            val_prec = precision_score(val_labels, val_preds_binary, zero_division=0)
            val_rec = recall_score(val_labels, val_preds_binary, zero_division=0)
            val_f1 = f1_score(val_labels, val_preds_binary, zero_division=0)

            epoch_time = time.time() - start_time

            log_entry = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'val_precision': val_prec,
                'val_recall': val_rec,
                'val_f1': val_f1,
                'epoch_time': epoch_time
            }
            training_log.append(log_entry)

            print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Val Acc: {val_acc:.4f}, Val Rec: {val_rec:.4f}, Val F1: {val_f1:.4f}, Time: {epoch_time:.2f}s")

            if epoch % checkpoint_interval == 0:
                checkpoint_path = self.checkpoint_dir / f"lstm_epoch_{epoch}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")

        return training_log
