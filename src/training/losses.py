import torch
import torch.nn as nn


def get_loss_function(loss_name="BCELoss", pos_weight=None):
    if loss_name == "BCELoss":
        if pos_weight is not None:
            return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        return nn.BCELoss()
    return nn.BCELoss()
