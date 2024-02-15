import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=1.0, epsilon=1e-7, num_classes=2, class_weights=None):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_classes = num_classes
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None

    def forward(self, y_pred, y_true):
        flat_labels = y_true.reshape(-1)
        alpha = 1 + (1 - torch.sum(flat_labels) / flat_labels.shape[0])
        probs = torch.softmax(y_pred, dim=1)  # Apply softmax to get probabilities
        
        tp = torch.sum(probs[:, 1] * y_true)
        fp = torch.sum(probs[:, 1] * (1 - y_true))
        fn = torch.sum((1 - probs[:, 1]) * y_true)

        tversky = (tp + self.epsilon) / (tp + fp +  alpha * fn + self.epsilon)
        # focal_tversky = torch.pow((1 - tversky), self.gamma)

        # if self.class_weights is not None:
        #     class_weights = self.class_weights.to(y_pred.device)
        #     focal_tversky = focal_tversky * class_weights[:, 1]


        return tversky

import torch
import torch.nn.functional as F

class CustomCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(CustomCrossEntropyLoss, self).__init__()

    def forward(self, input, target, weights = None):
        flat_labels = target.reshape(-1)
        class_counts = torch.bincount(flat_labels)
        total_pixels = torch.sum(class_counts)
        class_frequencies = class_counts / total_pixels
        class_weights = 1.0 / class_frequencies
        class_weights /= torch.sum(class_weights)
        log_softmax = F.log_softmax(input, dim=1)
        #weighted_log_softmax = log_softmax * weights.unsqueeze(1)
        loss = F.nll_loss(log_softmax, target, weight=class_weights)
        return loss

# Example usage:
# Assuming you have a DataLoader `train_loader` with `class_weights` calculated
# and your model output is of shape (batch_size, num_classes, height, width)


# # Inside your training loop
# for inputs, targets in train_loader:
#     inputs, targets = inputs.to(device), targets.to(device)
#     outputs = model(inputs)
#     loss = loss_fn(outputs, targets)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
