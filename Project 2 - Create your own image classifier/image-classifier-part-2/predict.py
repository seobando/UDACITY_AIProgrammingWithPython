import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import argparse
import json
import os

def get_input_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Predict flower name from an image')
    
    parser.add_argument('input', type=str, help='Path to image')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K predictions')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='Path to category names JSON file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    
    return parser.parse_args()

def load_checkpoint(filepath):
    """Load a checkpoint and rebuild the model."""
    checkpoint = torch.load(filepath, map_location='cpu')
    
    # Get architecture
    arch = checkpoint.get('arch', 'vgg16')
    
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
        else:
            model = models.vgg16(weights='DEFAULT')
            input_size = 25088
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
        else:
            model = models.vgg16(pretrained=True)
            input_size = 25088
    
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Load classifier
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image_path):
    """Process a PIL image for use in a PyTorch model."""
    # Open the image
    img = Image.open(image_path)
    
    # Resize where shortest side is 256 pixels, keeping aspect ratio
    width, height = img.size
    if width < height:
        img = img.resize((256, int(256 * height / width)))
    else:
        img = img.resize((int(256 * width / height), 256))
    
    # Crop out the center 224x224 portion
    width, height = img.size
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2
    img = img.crop((left, top, right, bottom))
    
    # Convert to numpy array
    np_image = np.array(img) / 255.0
    
    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # Reorder dimensions: color channel first
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

def predict(image_path, model, device, topk=5):
    """Predict the class (or classes) of an image using a trained deep learning model."""
    # Process the image
    img = process_image(image_path)
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img).float()
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        output = model.forward(img_tensor)
        ps = torch.exp(output)
        top_p, top_idx = ps.topk(topk, dim=1)
    
    # Convert to lists
    top_p = top_p.cpu().numpy()[0]
    top_idx = top_idx.cpu().numpy()[0]
    
    # Invert class_to_idx to get idx_to_class
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    
    # Convert indices to classes
    top_classes = [idx_to_class[idx] for idx in top_idx]
    
    return top_p, top_classes

def main():
    args = get_input_args()
    
    # Set device
    device = torch.device("cuda" if (args.gpu and torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    model = load_checkpoint(args.checkpoint)
    model.to(device)
    model.eval()
    
    # Load category names
    if os.path.exists(args.category_names):
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
    else:
        print(f"Warning: Category names file {args.category_names} not found. Using class indices.")
        cat_to_name = None
    
    # Make prediction
    print(f"Predicting image: {args.input}")
    probs, classes = predict(args.input, model, device, args.top_k)
    
    # Display results
    print("\nTop {} predictions:".format(args.top_k))
    print("-" * 50)
    for i in range(len(classes)):
        class_name = cat_to_name[classes[i]] if cat_to_name else classes[i]
        print(f"{i+1}. {class_name} (class: {classes[i]}) - Probability: {probs[i]:.4f}")

if __name__ == '__main__':
    main()

