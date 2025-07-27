import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import argparse
import os
import time

# Define command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network to classify images of flowers.')
    parser.add_argument('data_dir', type=str, help='Directory containing training and validation datasets.')
    parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg16', 'vgg13'], help='Pre-trained model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units for the classifier')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train the model')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save the checkpoint')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training (default: CPU)')
    
    return parser.parse_args()

# Define image transformations
def image_transforms():
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transforms, valid_transforms

# Load data
def load_data(data_dir, train_transforms, valid_transforms):
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = DataLoader(valid_data, batch_size=64)

    return trainloader, validloader, train_data.class_to_idx

# Define the model
def setup_model(arch='vgg16', hidden_units=512):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        model = models.vgg13(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(
        nn.Linear(25088, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )
    model.classifier = classifier

    return model

# Training the model
def train_model(model, trainloader, validloader, criterion, optimizer, scheduler, epochs, device):
    model.to(device)
    steps = 0
    running_loss = 0
    for epoch in range(epochs):
        model.train()
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % 40 == 0:
                model.eval()
                accuracy = 0
                valid_loss = 0
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        valid_loss += loss.item()

                        _, preds = torch.max(outputs, 1)
                        accuracy += torch.sum(preds == labels.data)

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/40:.3f}.. "
                      f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                      f"Valid accuracy: {accuracy.double()/len(validloader.dataset):.3f}")

                running_loss = 0
                model.train()

        scheduler.step()

    return model

# Save checkpoint
def save_checkpoint(model, optimizer, epochs, save_dir, class_to_idx):
    checkpoint = {
        'arch': 'vgg16',
        'epochs': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': class_to_idx
    }
    checkpoint_path = os.path.join(save_dir, 'checkpoint.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

# Main function
def main():
    args = parse_args()
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'

    train_transforms, valid_transforms = image_transforms()
    trainloader, validloader, class_to_idx = load_data(args.data_dir, train_transforms, valid_transforms)

    model = setup_model(arch=args.arch, hidden_units=args.hidden_units)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    model = train_model(model, trainloader, validloader, criterion, optimizer, scheduler, args.epochs, device)

    save_checkpoint(model, optimizer, args.epochs, args.save_dir, class_to_idx)

if __name__ == "__main__":
    main()
