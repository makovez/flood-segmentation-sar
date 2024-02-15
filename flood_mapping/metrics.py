import torch

def calculate_iou(predictions, targets, class_index=1):
    # Flatten predictions and targets
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    
    # Calculate intersection and union for class 1
    intersection = torch.sum((predictions == class_index) & (targets == class_index)).float()
    union = torch.sum((predictions == class_index) | (targets == class_index)).float()
    
    # Calculate IoU
    iou = intersection / (union + 1e-8)  # Adding epsilon to avoid division by zero
    
    return iou

def calculate_accuracy(predictions, targets, class_index=1):
    # Flatten predictions and targets
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    
    # Calculate accuracy for class 1
    correct_predictions = torch.sum((predictions == class_index) & (targets == class_index)).float()
    total_predictions = torch.sum(predictions == class_index).float()
    
    # Calculate accuracy
    accuracy = correct_predictions / (total_predictions + 1e-8)  # Adding epsilon to avoid division by zero
    
    return accuracy

# # Usage
# iou_class1 = calculate_iou(predictions, targets)
# accuracy_class1 = calculate_accuracy(predictions, targets)
# print("IoU of class 1:", iou_class1.item())
# print("Accuracy of class 1:", accuracy_class1.item())
