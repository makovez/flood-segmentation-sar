import os
import torch
import torch.optim as optim
from pathlib import Path
from torch import nn
from torchvision import transforms
from torchmetrics import JaccardIndex
from torchmetrics.classification import Accuracy
from tqdm import tqdm

from flood_mapping.models import UNet
from flood_mapping.dataset import FloodDatasetLoader, FloodDataset


class Trainer:
    def __init__(self, data_folder="data", model_folder="model", model_name="unet-voc.pt",
                 saving_interval=10, epoch_number=100, batch_size=2, clip_value=1.0,
                 shuffle_data_loader=False, learning_rate=0.0001, weight_decay=1e-8, momentum=0.9):
        self.data_folder = data_folder
        self.model_folder = Path(model_folder)
        self.model_folder.mkdir(exist_ok=True)
        self.model_path = self.model_folder / model_name
        self.saving_interval = saving_interval
        self.epoch_number = epoch_number
        self.batch_size = batch_size
        self.clip_value = clip_value
        self.shuffle_data_loader = shuffle_data_loader
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.iou_metric = JaccardIndex(task="multiclass", num_classes=2).to(self.device)  # Assuming binary classification
        self.accuracy_metric = Accuracy(task="multiclass", num_classes=2).to(self.device)

    def _create_dataset(self):
        flood_dataset = FloodDataset(self.data_folder, (128, 128))
        return FloodDatasetLoader(flood_dataset.X, flood_dataset.y)

    def _create_transforms(self):
        return transforms.Compose([
            transforms.Lambda(lambda x: torch.clamp(x, 0, self.clip_value)),
            transforms.Normalize(0, self.clip_value)
        ])


    def train(self):
        dataset = self._create_dataset()
        cell_dataset = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle_data_loader)
        normalize_transform = self._create_transforms()

        model = UNet(n_channels=2, labels=2)
        model.to(self.device)

        if os.path.isfile(self.model_path):
            model.load_state_dict(torch.load(self.model_path, map_location=torch.device(self.device)))

        optimizer = optim.RMSprop(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,
                                  momentum=self.momentum)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.epoch_number):
            print(f"Epoch {epoch}")
            losses = []
            iou_scores = []
            accuracy_scores = []

            with tqdm(total=len(cell_dataset), desc=f"Epoch {epoch}") as pbar:
                for i, batch in enumerate(cell_dataset):
                    input, target = batch

                    input = normalize_transform(input)
                    input = input.to(self.device)
                    target = target.type(torch.LongTensor).to(self.device)

                    if input.shape[0] < 2:
                        continue

                    optimizer.zero_grad()
                    output = model(input)
                    loss = criterion(output, target.squeeze())
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())

                    # Calculate IOU and accuracy for each batch
                    iou_batch = self.iou_metric(output.argmax(dim=1), target.squeeze())
                    accuracy_batch = self.accuracy_metric(output.argmax(dim=1), target.squeeze())
                    iou_scores.append(iou_batch.item())
                    accuracy_scores.append(accuracy_batch.item())

                    pbar.update(1)
                    pbar.set_postfix({"Loss": loss.item(), "IoU": iou_batch.item(), "Accuracy": accuracy_batch.item()})

                average_loss = sum(losses) / len(losses)
                average_iou = sum(iou_scores) / len(iou_scores)
                average_accuracy = sum(accuracy_scores) / len(accuracy_scores)

                print(f"Average Loss: {average_loss}, Average IoU: {average_iou}, Average Accuracy: {average_accuracy}")

                if (epoch + 1) % self.saving_interval == 0:
                    print("Saving model")
                    torch.save(model.state_dict(), self.model_path)

        print("Training completed.")
        torch.save(model.state_dict(), self.model_path)



if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
