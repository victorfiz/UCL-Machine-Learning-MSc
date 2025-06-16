import torchvision
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import os
import torch.nn as nn
import torch.optim as optim
from datasets import *


# CAM Model using ResNet backbone
class CAMModel(nn.Module):
    def __init__(self, num_classes=37):  # Oxford-IIIT Pet has 37 categories
        super(CAMModel, self).__init__()
        # Load a pre-trained ResNet
        self.resnet = torchvision.models.resnet50(weights='IMAGENET1K_V2')
        # self.resnet = torchvision.models.wide_resnet50_2(weights='IMAGENET1K_V2')  # for a wider ResNet
        
        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(self.resnet.children())[:-2]) # for last Conv layer
        # self.features = nn.Sequential(*list(self.resnet.children())[:-3]) # for 2nd to last Conv layer
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classification layer
        # self.fc = nn.Linear(1024, num_classes)  # 1024 is output channels for ResNet-50 in the 2nd to last Conv layer, which is a 1024x14x14 image
        self.fc = nn.Linear(2048, num_classes)  # 2048 is output channels for ResNet-50 in the last Conv layer, which is a 2048x7x7 image
    
    def forward(self, x):
        x = x.to(torch.float)
        
        # Feature extraction
        features = self.features(x)
        
        # Save features for CAM generation
        self.last_features = features
        
        # Global Average Pooling
        x = self.gap(features)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.fc(x)
        return x

def train_model(model, train_loader, num_epochs):
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Training on {device}...")

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, image_data in train_loader:
            images = images.to(device)
            labels = image_data[DatasetSelection.Class].to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')

    # Save the model checkpoint
    torch.save(model.state_dict(), './models/cam_model.pth')
    return model

print("Creating model...")
model = CAMModel()
if not os.path.exists('./models/'): os.mkdir('./models')
# Set to True if you want to train, False to load pre-trained
TRAIN_MODEL = True
dataset = InMemoryPetSegmentationDataset(
        DATA_DIR, ANNOTATION_DIR, targets_list=[DatasetSelection.Class])

if TRAIN_MODEL:
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = train_model(model, train_loader, num_epochs=8)
else:
    # Load pretrained model if it exists
    if os.path.exists('./models/cam_model.pth'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load('./models/cam_model.pth', map_location=device, weights_only=False))
        model = model.to(device)
        print("Loaded pre-trained model")
    else:
        print("No pre-trained model found")

def generate_cam(model, img, label):
    """
    Generate Class Activation Map for an image.
    
    Args:
        model: The trained model with CAM capability
        img_tensor: Preprocessed image tensor
        label: Class label for which to generate CAM
    
    Returns:
        CAM numpy array
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        model(img.unsqueeze(0).to(device))  # run a forward pass
        # Get weights from the final FC layer for the specified class
        fc_weights = model.fc.weight[label].cpu()
        
        # Get feature maps from the last convolutional layer
        feature_maps = model.last_features.squeeze(0).cpu()  # will be non-negative since last op is RELU
        # Calculate weighted sum of feature maps
        cam = (fc_weights[:, None, None] * feature_maps).sum(0)
        cam = torch.sigmoid(cam)
    
    # Convert to numpy and resize
    cam = cam.detach()
    return cam

cams = []
for fname, d in zip(dataset.available_images, dataset):
    image, image_data = d
    label = image_data[DatasetSelection.Class]
    if label >= 0:
        res = generate_cam(model, image, label)
        cams.append((fname, res))

save_cam_dataset([c[0] for c in cams], [c[1] for c in cams])


print("DONE!")