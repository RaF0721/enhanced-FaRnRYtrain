from ultralytics import YOLO
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision import transforms as T
from SupContrast.losses import SupConLoss


class ContrastiveAugmentation:
    def __init__(self):
        self.augment = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            T.GaussianBlur(kernel_size=(3, 7)),
            T.RandomPerspective(distortion_scale=0.3),
            T.RandomRotation(degrees=30),
            T.ToTensor()
        ])

    def __call__(self, img):
        return self.augment(img), self.augment(img)


class ContrastiveYOLOResNet(nn.Module):
    def __init__(self, resnet_type='resnet50'):
        super().__init__()
        if resnet_type == 'resnet18':
            self.resnet = models.resnet18(pretrained=True)
            feature_dim = 512
        elif resnet_type == 'resnet34':
            self.resnet = models.resnet34(pretrained=True)
            feature_dim = 512
        elif resnet_type == 'resnet50':
            self.resnet = models.resnet50(pretrained=True)
            feature_dim = 2048
        else:
            raise ValueError("Unsupported ResNet type")

        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

    def forward(self, x):
        resnet_features = self.resnet(x)
        resnet_features = resnet_features.view(resnet_features.size(0), -1)
        projection = self.projection_head(resnet_features)
        return projection


model = YOLO("yolov12l.pt")
contrastive_model = ContrastiveYOLOResNet(resnet_type='resnet50')

train_config = {
    'data': '../datasets/BrainTumor/BrainTumorYolov11/data.yaml',
    'epochs': 450,
    'batch': 48,
    'imgsz': 640,
    'scale': 0.5,
    'mosaic': 1.0,
    'mixup': 0.0,
    'copy_paste': 0.1,
    'device': 0,
    'workers': 0,
}

contrastive_criterion = SupConLoss(temperature=0.1)


def train_with_supcontrast():
    results = model.train(**train_config)

    contrastive_optim = torch.optim.AdamW(contrastive_model.parameters(), lr=1e-4)

    transform = T.Compose([T.Resize((train_config['imgsz'], train_config['imgsz'])), T.ToTensor()])
    dataset = datasets.ImageFolder(root=train_config['data'].replace('data.yaml', 'train'), transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    device = torch.device(f"cuda:{train_config['device']}" if torch.cuda.is_available() else "cpu")
    contrastive_model.to(device)

    for epoch in range(40):
        contrastive_model.train()
        running_loss = 0.0

        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            aug1, aug2 = ContrastiveAugmentation()(imgs)
            aug1, aug2 = aug1.to(device), aug2.to(device)

            features1 = contrastive_model(aug1)
            features2 = contrastive_model(aug2)

            features = torch.cat([features1.unsqueeze(1), features2.unsqueeze(1)], dim=1)
            labels = labels.repeat(2)

            loss = contrastive_criterion(features, labels)

            contrastive_optim.zero_grad()
            loss.backward()
            contrastive_optim.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(loader)
        print(f"Epoch [{epoch + 1}/20], Contrastive Loss: {epoch_loss:.4f}")


if __name__ == "__main__":
    train_with_supcontrast()

    metrics = model.val()