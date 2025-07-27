import torch
from torch import nn
from torchvision import datasets, transforms, models
import argparse
from PIL import Image
import json
import numpy as np

# Command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Predict the class of an image using a trained model.')
    parser.add_argument('image_path', type=str, help='Path to the image file.')
    parser.add_argument('checkpoint', type=str, help='Path to the trained model checkpoint file.')
    parser.add_argument('--top_k', type=int, default=1, help='Return top K most likely classes.')
    parser.add_argument('--category_names', type=str, default=None, help='Mapping of categories to real names.')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference (default: CPU)')
    
    return parser.parse_args()

# Image preprocessing
def process_image(image_path):
    img = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    return img_tensor

# Load checkpoint
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained=False)
    model.classifier = checkpoint['model_state_dict']['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.eval()
    return model

# Inference function
def predict(image_path, model, top_k, device):
    model.to(device)
    img_tensor = process_image(image_path)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.exp(outputs)
        top_p, top_class = probabilities.topk(top_k, dim=1)

    return top_p.cpu().numpy()[0], top_class.cpu().numpy()[0]

# Main function
def main():
    args = parse_args()
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    
    model = load_checkpoint(args.checkpoint)
    
    top_p, top_class = predict(args.image_path, model, args.top_k, device)

    # Convert class indices to class labels
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    class_names = [idx_to_class[i] for i in top_class]

    # Load category names if provided
    if args.category_names:
        with open(args.category_names, 'r') as f:
            category_names = json.load(f)
        class_names = [category_names.get(c, c) for c in class_names]

    # Display results
    print(f"Top {args.top_k} predictions:")
    for i in range(args.top_k):
        print(f"Class: {class_names[i]}, Probability: {top_p[i]:.4f}")

if __name__ == "__main__":
    main()
