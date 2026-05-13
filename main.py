import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, RandomCrop, RandomHorizontalFlip, Normalize
import os


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss_value, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()

    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)

            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    print(
        f"Test Error: \n"
        f"Accuracy: {(100*correct):>0.1f}%, "
        f"Avg loss: {test_loss:>8f} \n"
    )


class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),   # BatchNorm
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),   # BatchNorm
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),   # BatchNorm
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),   # BatchNorm
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),  # BatchNorm
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),  # BatchNorm
            nn.ReLU(),

            nn.MaxPool2d(2),

            # AdaptiveAvgPool
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(128, 512),
            nn.ReLU(),

            # Dropout
            nn.Dropout(0.5),

            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def main():

    batch_size = 64
    epochs = 30

    # Data augmentation for training
    train_transform = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )
    ])

    test_transform = Compose([
        ToTensor(),
        Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )
    ])

    training_data100 = datasets.CIFAR100(
        root="data",
        train=True,
        download=True,
        transform=train_transform,
    )

    test_data100 = datasets.CIFAR100(
        root="data",
        train=False,
        download=True,
        transform=test_transform,
    )

    train_dataloader100 = DataLoader(
        training_data100,
        batch_size=batch_size,
        shuffle=True
    )

    test_dataloader100 = DataLoader(
        test_data100,
        batch_size=batch_size
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using {device} device")

    model100 = ConvNet(num_classes=100).to(device)

    # Label smoothing
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer100 = torch.optim.Adam(
        model100.parameters(),
        lr=0.001,
        weight_decay=1e-4
    )

    # CosineAnnealingLR
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer100,
        T_max=epochs
    )

    start_epoch = 0
    checkpoint_path = "checkpoint.pth"

    if os.path.exists(checkpoint_path):

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device
        )

        model100.load_state_dict(
            checkpoint['model_state_dict']
        )

        optimizer100.load_state_dict(
            checkpoint['optimizer_state_dict']
        )

        scheduler.load_state_dict(
            checkpoint['scheduler_state_dict']
        )

        start_epoch = checkpoint['epoch']

        print(f"Resumed from epoch {start_epoch + 1}")

    for t in range(start_epoch, epochs):

        print(f"Epoch {t+1} (CIFAR-100)\n-------------------------------")

        train(
            train_dataloader100,
            model100,
            loss_fn,
            optimizer100,
            device
        )

        test(
            test_dataloader100,
            model100,
            loss_fn,
            device
        )

        scheduler.step()

        # Save checkpoint
        torch.save({
            'epoch': t + 1,
            'model_state_dict': model100.state_dict(),
            'optimizer_state_dict': optimizer100.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, checkpoint_path)

    torch.save(model100.state_dict(), "cifar100_cnn.pth")

    print("Done CIFAR-100")
    print("Saved model state to cifar100_cnn.pth")


if __name__ == "__main__":
    main()