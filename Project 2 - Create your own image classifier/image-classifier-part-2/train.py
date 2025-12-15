import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import argparse
import os

def get_input_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a neural network on a dataset')
    
    parser.add_argument('data_dir', type=str, help='Path to the data directory')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg16', 'vgg13', 'densenet121', 'alexnet'],
                        help='Architecture (vgg16, vgg13, densenet121, alexnet)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, nargs='+', default=[4096, 1024],
                        help='Hidden units in classifier (can specify multiple layers)')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    
    return parser.parse_args()

def get_model(arch, hidden_units, num_classes=102):
    """Build and return a model with the specified architecture."""
    # Load pre-trained model
    try:
        if arch == 'vgg16':
            model = models.vgg16(weights='DEFAULT')
            input_size = 25088
        elif arch == 'vgg13':
            model = models.vgg13(weights='DEFAULT')
            input_size = 25088
        elif arch == 'densenet121':
            model = models.densenet121(weights='DEFAULT')
            input_size = 1024
        elif arch == 'alexnet':
            model = models.alexnet(weights='DEFAULT')
            input_size = 9216
    except:
        # Fallback for older PyTorch versions
        if arch == 'vgg16':
            model = models.vgg16(pretrained=True)
            input_size = 25088
        elif arch == 'vgg13':
            model = models.vgg13(pretrained=True)
            input_size = 25088
        elif arch == 'densenet121':
            model = models.densenet121(pretrained=True)
            input_size = 1024
        elif arch == 'alexnet':
            model = models.alexnet(pretrained=True)
            input_size = 9216
    
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Build classifier
    layers = []
    prev_size = input_size
    
    for hidden_size in hidden_units:
        layers.append(nn.Linear(prev_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.5))
        prev_size = hidden_size
    
    # Output layer
    layers.append(nn.Linear(prev_size, num_classes))
    layers.append(nn.LogSoftmax(dim=1))
    
    classifier = nn.Sequential(*layers)
    
    # Replace classifier
    if arch.startswith('vgg') or arch == 'alexnet':
        model.classifier = classifier
    elif arch == 'densenet121':
        model.classifier = classifier
    
    return model

def get_data_loaders(data_dir):
    """Create and return data loaders for train, validation, and test sets."""
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')
    
    # Define transforms
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    valid_test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_test_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=valid_test_transforms)
    
    # Create data loaders
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
    
    return trainloader, validloader, testloader, train_dataset

def train_model(model, trainloader, validloader, criterion, optimizer, device, epochs):
    """Train the model."""
    model.train()
    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Validation
        model.eval()
        valid_loss = 0
        accuracy = 0
        with torch.no_grad():
            for inputs, labels in validloader:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)
                
                valid_loss += batch_loss.item()
                
                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        model.train()
        
        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {running_loss/len(trainloader):.3f}.. "
              f"Validation loss: {valid_loss/len(validloader):.3f}.. "
              f"Validation accuracy: {accuracy/len(validloader):.3f}")
    
    return model

def main():
    args = get_input_args()
    
    # Set device
    device = torch.device("cuda" if (args.gpu and torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")
    
    # Get data loaders
    print("Loading data...")
    trainloader, validloader, testloader, train_dataset = get_data_loaders(args.data_dir)
    
    # Get number of classes
    num_classes = len(train_dataset.classes)
    print(f"Number of classes: {num_classes}")
    
    # Build model
    print(f"Building model with architecture: {args.arch}")
    model = get_model(args.arch, args.hidden_units, num_classes)
    model.to(device)
    
    # Define loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    # Train model
    print("Training model...")
    model = train_model(model, trainloader, validloader, criterion, optimizer, device, args.epochs)
    
    # Test model
    print("Testing model...")
    model.eval()
    test_loss = 0
    accuracy = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
            
            test_loss += batch_loss.item()
            
            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
    print(f"Test loss: {test_loss/len(testloader):.3f}.. "
          f"Test accuracy: {accuracy/len(testloader):.3f}")
    
    # Save checkpoint
    checkpoint_path = os.path.join(args.save_dir, 'checkpoint.pth')
    checkpoint = {
        'arch': args.arch,
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'class_to_idx': train_dataset.class_to_idx,
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': args.epochs,
        'hidden_units': args.hidden_units,
        'learning_rate': args.learning_rate,
        'num_classes': num_classes
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == '__main__':
    main()

