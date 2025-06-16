import os
import numpy as np
# if matplotlib is installed
try:
    import matplotlib.pyplot as plt
    from tqdm.notebook import tqdm
    import cv2
except:
    print("Matplotlib or tqdm not installed. Skipping visualization.")
    plt = None
    tqdm = lambda x: x
    cv2 = None
from PIL import Image
from datetime import datetime
import logging
import sys
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader

def setup_folder_and_logger(name):
    if not os.path.exists('logs'):
        os.makedirs('logs')
    experiment_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(f'logs/{experiment_name}'):
        os.makedirs(f'logs/{experiment_name}')
    
    experiment_folder = f'logs/{experiment_name}'   
        
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Log to stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Log to file
    file_handler = logging.FileHandler(f'{experiment_folder}/run.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return experiment_folder, logger

def evaluate_and_visualize_trimaps(results, experiment_folder, iou_threshold=0.5):
    """
    Evaluate segmentation and classification results with trimap ground truth.
    Uses only NumPy and Matplotlib.
    
    Args:
        results: List of result dictionaries from SAM predictions
        experiment_folder: Folder to save visualizations and metrics
        iou_threshold: Threshold to consider a mask as correct
    """
    # Create metrics subfolder
    metrics_folder = os.path.join(experiment_folder, 'metrics')
    os.makedirs(metrics_folder, exist_ok=True)
    
    # Initialize metrics
    classification_metrics = {
        'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0,
        'class_correct': 0
    }
    
    segmentation_metrics = {
        'iou': [],
        'dice': [],
        'pixel_precision': [],
        'pixel_recall': [],
        'pixel_accuracy': [],
        'boundary_f1': [],
        'has_gt_mask': 0,
        'has_pred_mask': 0,
        'correct_masks': 0
    }
    
    # Process each result
    for result in results:
        # Classification evaluation
        pred_label = result['pred_labels']
        true_label = result['true_labels']
        
        if pred_label == true_label:
            classification_metrics['class_correct'] += 1
        
        # Segmentation evaluation
        masks = result['masks']
        trimap = np.array(result['mask_labels'])
        
        # Convert trimap to binary mask:
        # In trimap: 0=background, 1=foreground, 2=unknown
        # For evaluation: Consider only definite foreground (1) as foreground
        gt_mask = (trimap == 1).astype(np.uint8)
        
        # Create evaluation mask that ignores unknown regions
        # This mask indicates pixels we should consider in the evaluation
        evaluation_mask = (trimap != 2).astype(np.uint8)
        
        # Skip if ground truth mask is empty or all zeros/unknown
        if gt_mask.sum() == 0:
            continue
            
        segmentation_metrics['has_gt_mask'] += 1
        
        # Skip if no predicted masks
        if len(masks) == 0:
            continue
            
        segmentation_metrics['has_pred_mask'] += 1
        
        # Get the best mask (highest IoU with ground truth)
        best_iou = 0
        best_mask = None
        
        for mask in masks:
            # Only consider pixels not in the "unknown" region of the trimap
            valid_pixels = evaluation_mask == 1
            valid_mask = mask * valid_pixels
            valid_gt = gt_mask * valid_pixels
            
            # Calculate IoU on valid regions only
            intersection = np.logical_and(valid_mask, valid_gt).sum()
            union = np.logical_or(valid_mask, valid_gt).sum()
            iou = intersection / union if union > 0 else 0
            
            if iou > best_iou:
                best_iou = iou
                best_mask = mask
        
        if best_mask is not None:
            # Calculate metrics for the best mask (only on valid regions)
            mask_pred = best_mask * evaluation_mask
            mask_true = gt_mask * evaluation_mask
            
            # IoU
            intersection = np.logical_and(mask_pred, mask_true).sum()
            union = np.logical_or(mask_pred, mask_true).sum()
            iou = intersection / union if union > 0 else 0
            segmentation_metrics['iou'].append(iou)
            
            # Dice coefficient
            dice = (2 * intersection) / (mask_pred.sum() + mask_true.sum()) if (mask_pred.sum() + mask_true.sum()) > 0 else 0
            segmentation_metrics['dice'].append(dice)
            
            # Pixel-wise precision and recall
            true_pos = intersection
            false_pos = mask_pred.sum() - true_pos
            false_neg = mask_true.sum() - true_pos
            
            pixel_precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            pixel_recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            
            segmentation_metrics['pixel_precision'].append(pixel_precision)
            segmentation_metrics['pixel_recall'].append(pixel_recall)
            
            # Pixel-wise accuracy (on valid regions only)
            total_valid_pixels = evaluation_mask.sum()
            correct_pixels = ((mask_pred == mask_true) & (evaluation_mask == 1)).sum()
            pixel_accuracy = correct_pixels / total_valid_pixels if total_valid_pixels > 0 else 0
            segmentation_metrics['pixel_accuracy'].append(pixel_accuracy)
            
            # Simplified boundary F1
            # For trimaps, focus on boundaries of the definite foreground region
            def compute_boundary(mask, kernel_size=3):
                h, w = mask.shape
                boundary = np.zeros_like(mask)
                
                for i in range(h):
                    for j in range(w):
                        if mask[i, j]:
                            # Check if this pixel is at a boundary
                            i_min = max(0, i - 1)
                            i_max = min(h, i + 2)
                            j_min = max(0, j - 1)
                            j_max = min(w, j + 2)
                            
                            if not np.all(mask[i_min:i_max, j_min:j_max]):
                                boundary[i, j] = 1
                                
                return boundary
            
            # Extract boundaries, considering only valid regions
            mask_pred_boundary = compute_boundary(mask_pred) & evaluation_mask
            mask_true_boundary = compute_boundary(mask_true) & evaluation_mask
            
            boundary_intersection = np.logical_and(mask_pred_boundary, mask_true_boundary).sum()
            boundary_precision = boundary_intersection / mask_pred_boundary.sum() if mask_pred_boundary.sum() > 0 else 0
            boundary_recall = boundary_intersection / mask_true_boundary.sum() if mask_true_boundary.sum() > 0 else 0
            boundary_f1 = 2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall) if (boundary_precision + boundary_recall) > 0 else 0
            
            segmentation_metrics['boundary_f1'].append(boundary_f1)
            
            # Check if this mask exceeds the IoU threshold
            if iou >= iou_threshold:
                segmentation_metrics['correct_masks'] += 1
                # Update confusion matrix for the class
                classification_metrics['tp'] += 1
            else:
                classification_metrics['fp'] += 1
                
            # Save visualization of the trimap vs. predicted mask
            plt.figure(figsize=(15, 5))
            
            # Plot original trimap
            plt.subplot(1, 3, 1)
            trimap_vis = np.zeros((*trimap.shape, 3), dtype=np.uint8)
            trimap_vis[trimap == 0] = [0, 0, 255]    # Background in blue
            trimap_vis[trimap == 1] = [0, 255, 0]    # Foreground in green
            trimap_vis[trimap == 2] = [255, 0, 0]    # Unknown in red
            plt.imshow(trimap_vis)
            plt.title('Trimap Ground Truth')
            plt.axis('off')
            
            # Plot binary ground truth
            plt.subplot(1, 3, 2)
            plt.imshow(gt_mask, cmap='gray')
            plt.title('Binary GT (Foreground only)')
            plt.axis('off')
            
            # Plot predicted mask
            plt.subplot(1, 3, 3)
            plt.imshow(best_mask, cmap='gray')
            plt.title(f'Predicted Mask (IoU: {iou:.3f})')
            plt.axis('off')
            
            # Save individual comparison
            os.makedirs(f"{metrics_folder}/mask_comparisons", exist_ok=True)
            plt.savefig(f"{metrics_folder}/mask_comparisons/sample_{len(segmentation_metrics['iou'])}.png")
            plt.close()
    
    # Calculate classification metrics
    total_samples = len(results)
    classification_accuracy = classification_metrics['class_correct'] / total_samples if total_samples > 0 else 0
    
    # Convert segmentation metrics to numpy arrays
    for key in ['iou', 'dice', 'pixel_precision', 'pixel_recall', 'pixel_accuracy', 'boundary_f1']:
        segmentation_metrics[key] = np.array(segmentation_metrics[key])
    
    # Calculate averages and standard deviations
    avg_metrics = {}
    std_metrics = {}
    
    for key in ['iou', 'dice', 'pixel_precision', 'pixel_recall', 'pixel_accuracy', 'boundary_f1']:
        if len(segmentation_metrics[key]) > 0:
            avg_metrics[key] = np.mean(segmentation_metrics[key])
            std_metrics[key] = np.std(segmentation_metrics[key])
        else:
            avg_metrics[key] = 0
            std_metrics[key] = 0
    
    # Calculate correct mask ratio
    total_with_gt = segmentation_metrics['has_gt_mask']
    segmentation_metrics['correct_mask_ratio'] = segmentation_metrics['correct_masks'] / total_with_gt if total_with_gt > 0 else 0
    
    # Print metrics summary
    print("\nCLASSIFICATION METRICS:")
    print(f"Accuracy: {classification_accuracy:.4f}")
    
    print("\nSEGMENTATION METRICS (evaluated on non-ambiguous regions only):")
    print(f"Mean IoU: {avg_metrics['iou']:.4f} ± {std_metrics['iou']:.4f}")
    print(f"Mean Dice: {avg_metrics['dice']:.4f} ± {std_metrics['dice']:.4f}")
    print(f"Mean Pixel Precision: {avg_metrics['pixel_precision']:.4f} ± {std_metrics['pixel_precision']:.4f}")
    print(f"Mean Pixel Recall: {avg_metrics['pixel_recall']:.4f} ± {std_metrics['pixel_recall']:.4f}")
    print(f"Mean Pixel Accuracy: {avg_metrics['pixel_accuracy']:.4f} ± {std_metrics['pixel_accuracy']:.4f}")
    print(f"Mean Boundary F1: {avg_metrics['boundary_f1']:.4f} ± {std_metrics['boundary_f1']:.4f}")
    print(f"Correct Mask Ratio (IoU > {iou_threshold}): {segmentation_metrics['correct_mask_ratio']:.4f}")
    print(f"Samples with GT masks: {segmentation_metrics['has_gt_mask']}/{total_samples}")
    print(f"Samples with predicted masks: {segmentation_metrics['has_pred_mask']}/{segmentation_metrics['has_gt_mask']}")
    
    # Save metrics to text file
    with open(f"{metrics_folder}/metrics_summary.txt", "w") as f:
        f.write("CLASSIFICATION METRICS:\n")
        f.write(f"Accuracy: {classification_accuracy:.4f}\n\n")
        
        f.write("SEGMENTATION METRICS (evaluated on non-ambiguous regions only):\n")
        f.write(f"Mean IoU: {avg_metrics['iou']:.4f} ± {std_metrics['iou']:.4f}\n")
        f.write(f"Mean Dice: {avg_metrics['dice']:.4f} ± {std_metrics['dice']:.4f}\n")
        f.write(f"Mean Pixel Precision: {avg_metrics['pixel_precision']:.4f} ± {std_metrics['pixel_precision']:.4f}\n")
        f.write(f"Mean Pixel Recall: {avg_metrics['pixel_recall']:.4f} ± {std_metrics['pixel_recall']:.4f}\n")
        f.write(f"Mean Pixel Accuracy: {avg_metrics['pixel_accuracy']:.4f} ± {std_metrics['pixel_accuracy']:.4f}\n")
        f.write(f"Mean Boundary F1: {avg_metrics['boundary_f1']:.4f} ± {std_metrics['boundary_f1']:.4f}\n")
        f.write(f"Correct Mask Ratio (IoU > {iou_threshold}): {segmentation_metrics['correct_mask_ratio']:.4f}\n")
        f.write(f"Samples with GT masks: {segmentation_metrics['has_gt_mask']}/{total_samples}\n")
        f.write(f"Samples with predicted masks: {segmentation_metrics['has_pred_mask']}/{segmentation_metrics['has_gt_mask']}\n")
    
    # VISUALIZATIONS (same as before but with updated metrics)
    # [Include the same visualization code as in the previous function]
    
    # 1. IoU Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(segmentation_metrics['iou'], bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(x=iou_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({iou_threshold})')
    plt.title('Distribution of IoU Scores', fontsize=14)
    plt.xlabel('IoU Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{metrics_folder}/iou_distribution.png")
    plt.close()
    
    # 2. IoU vs Dice Scatter Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(segmentation_metrics['iou'], segmentation_metrics['dice'], alpha=0.7, s=50)
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2)  # Line showing y=x
    plt.title('IoU vs Dice Coefficient', fontsize=14)
    plt.xlabel('IoU Score', fontsize=12)
    plt.ylabel('Dice Coefficient', fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{metrics_folder}/iou_vs_dice.png")
    plt.close()
    
    # 3. Precision-Recall Scatter Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(segmentation_metrics['pixel_precision'], segmentation_metrics['pixel_recall'], alpha=0.7, s=50)
    plt.title('Pixel-wise Precision vs Recall', fontsize=14)
    plt.xlabel('Precision', fontsize=12)
    plt.ylabel('Recall', fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{metrics_folder}/precision_recall.png")
    plt.close()
    
    # 4. Bar chart of average metrics
    plt.figure(figsize=(12, 6))
    metric_names = ['IoU', 'Dice', 'Precision', 'Recall', 'Accuracy', 'Boundary F1']
    metric_values = [avg_metrics['iou'], avg_metrics['dice'], 
                     avg_metrics['pixel_precision'], avg_metrics['pixel_recall'], 
                     avg_metrics['pixel_accuracy'], avg_metrics['boundary_f1']]
    
    bars = plt.bar(metric_names, metric_values, color='skyblue', edgecolor='black')
    
    # Add error bars
    error_values = [std_metrics['iou'], std_metrics['dice'], 
                   std_metrics['pixel_precision'], std_metrics['pixel_recall'], 
                   std_metrics['pixel_accuracy'], std_metrics['boundary_f1']]
    
    plt.errorbar(metric_names, metric_values, yerr=error_values, fmt='none', ecolor='black', capsize=5)
    
    # Add value labels on the bars
    for bar, value in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width()/2, value + 0.01, f'{value:.3f}', 
                ha='center', va='bottom', fontsize=10)
    
    plt.title('Average Segmentation Metrics', fontsize=14)
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0, 1.1)  # Make sure there's space for the error bars and text
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{metrics_folder}/average_metrics.png")
    plt.close()
    
    # Return the metrics
    return {
        'classification_accuracy': classification_accuracy,
        'avg_metrics': avg_metrics,
        'std_metrics': std_metrics,
        'correct_mask_ratio': segmentation_metrics['correct_mask_ratio']
    }

# CAM Model using ResNet backbone
class CAMModel(nn.Module):
    def __init__(self, num_classes=37):  # Oxford-IIIT Pet has 37 categories
        super(CAMModel, self).__init__()
        # Load a pre-trained ResNet
        self.resnet = torchvision.models.resnet50(weights='IMAGENET1K_V1')
        
        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(self.resnet.children())[:-2])
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classification layer
        self.fc = nn.Linear(2048, num_classes)  # 2048 is output channels for ResNet-50
        
        # Define transforms
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
    
    def forward(self, x):
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
    
    def predict(self, image):
        """
        Takes a PIL image, applies transformations and returns prediction
        
        Args:
            image: PIL Image
            
        Returns:
            model output
        """
        device = next(self.parameters()).device
        self.eval()
        
        # Apply transforms
        img_tensor = self.transforms(image)
        
        # Forward pass
        with torch.no_grad():
            output = self(img_tensor.unsqueeze(0).to(device))
            
        return output, img_tensor

    def generate_cam(self, img, label=None):
        """
        Generate Class Activation Map for an image.
        
        Args:
            model: The trained model with CAM capability
            img_tensor: Preprocessed image tensor
            label: Class label for which to generate CAM. If None, the predicted class is used.
        
        Returns:
            CAM numpy array
        """
        model = self
        device = next(model.parameters()).device
        model.eval()
        
        # Forward pass
        with torch.no_grad():
            output = model(img.unsqueeze(0).to(device))
        
        # get class prediction as well
        if label is None:
            _, pred_label = torch.max(output, 1)
            pred_label = pred_label.item()
        else:
            pred_label = label
                    
        # Get weights from the final FC layer for the specified class
        fc_weights = model.fc.weight[pred_label].cpu()
        
        # Get feature maps from the last convolutional layer
        feature_maps = model.last_features.squeeze(0).cpu()
        
        # Calculate weighted sum of feature maps
        cam = torch.zeros(feature_maps.shape[1:])
        for i, weight in enumerate(fc_weights):
            cam += weight * feature_maps[i]
        
        # Apply ReLU and normalize
        cam = torch.maximum(cam, torch.tensor(0.0))
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Convert to numpy and resize
        cam = cam.detach().numpy()
        
        return pred_label, cam
    

    def plot_cam_overlay(self, img, cam, label=None, sampled_points=None, save_path=None):
        """
        Plot image with CAM overlay.
        
        Args:
            img: Original image as numpy array (H,W,C) with RGB channels
            cam: Resized CAM numpy array (same H,W as img)
            label: Optional label to display in title
        """
        # Convert CAM to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Create overlay (60% original image, 40% heatmap)
        overlay = np.uint8(0.6 * img + 0.4 * heatmap)
        
        # Plot
        plt.figure(figsize=(8, 8))
        plt.imshow(overlay)
        if label is not None:
            plt.title(f'Class: {label}')
            
        # plot points
        if sampled_points is not None:
            for idx in sampled_points:
                plt.scatter(idx[0], idx[1], c='red', s=10)
                
        plt.axis('off')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()
        
    def sample_cam_points(self, cam_resized, strategy='local_peak'):
        assert strategy in ['peak', 'local_peak', 'random'], "Invalid strategy"
        if strategy == 'peak':
            peak = np.unravel_index(np.argmax(cam_resized), cam_resized.shape)
            return np.array([[peak[1], peak[0]]])
        
        elif strategy == 'local_peak':
            # find local peaks
            kernel = np.ones((3,3))
            cam_dilated = cv2.dilate(cam_resized, kernel)
            local_peaks = np.where(cam_resized == cam_dilated)
            local_peaks = np.stack(local_peaks, axis=1)
            return local_peaks
            
        else:
            cam_resized = np.where(cam_resized < 0.7, 0, cam_resized)    
            heatmap_distribution = (cam_resized.flatten()/cam_resized.sum())
            heatmap_distribution = heatmap_distribution**2
            sampled_points = np.random.choice(np.arange(cam_resized.size), size=50, p=cam_resized.flatten()/cam_resized.sum())
            sampled_points = np.unravel_index(sampled_points, cam_resized.shape)
            sampled_points = np.stack(sampled_points, axis=1)
            # reverse x, y
            sampled_points = sampled_points[:, ::-1]
            return sampled_points
        # Add this method to the CAMModel class

    def batch_generate_cam(self, batch_tensors, batch_labels):
        """Generate CAMs for a batch of images"""
        self.eval()
        device = next(self.parameters()).device
        batch_size = batch_tensors.size(0)
        batch_cams = []
        batch_labels_processed = []
        
        with torch.no_grad():
            # Forward pass for the whole batch
            outputs = self(batch_tensors.to(device))
            
            # Get feature maps for the whole batch
            batch_features = self.last_features.cpu()
            
            # Process each image in the batch
            for i in range(batch_size):
                label = batch_labels[i]
                # Get weights from the final FC layer for the specified class
                fc_weights = self.fc.weight[label].cpu()
                
                # Get feature maps for this image
                feature_maps = batch_features[i]
                
                # Calculate weighted sum of feature maps
                cam = torch.zeros(feature_maps.shape[1:])
                for j, weight in enumerate(fc_weights):
                    cam += weight * feature_maps[j]
                
                # Apply ReLU and normalize
                cam = torch.maximum(cam, torch.tensor(0.0))
                if cam.max() > 0:
                    cam = cam / cam.max()
                
                # Convert to numpy
                cam = cam.detach().numpy()
                batch_cams.append(cam)
                batch_labels_processed.append(label)
        
        return batch_labels_processed, batch_cams

if __name__ == "__main__":
    print(os.getcwd())
    experiment_folder, logger = setup_folder_and_logger(__name__)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the dataset without transforms
    test_dataset = OxfordIIITPet(root='./data', 
                            split='test',
                            target_types="category",
                            transform=None,  # No transforms here
                            download=True)
    
    # make new CAMModel and load the pre-trained weights
    model = CAMModel()
    model.load_state_dict(torch.load('./weakly_supervised/models/cam_model.pth', map_location=device, weights_only=False))
    model = model.to(device)
    logger.info("Loading model:\n", model) 
    
    # idx = 0 # Change this index to visualize different images
    
    # # Get image and label from dataset - now the image is a PIL image
    # original_img, label = test_dataset[idx]
    
    # # Apply transforms inside the predict method
    # output, img_tensor = model.predict(original_img)
    
    # logger.info(f'Class: {label}')
    
    # # Generate CAM
    # pred_label, pred_cam = model.generate_cam(img_tensor, label)
    
    # # Convert original image to numpy array for plotting
    # original_np = np.array(original_img)

    # # Resize CAM to match original image size
    # cam_resized = cv2.resize(pred_cam, (original_img.width, original_img.height))
    
    # points = model.sample_cam_points(cam_resized)

    # # Plot overlay
    # model.plot_cam_overlay(original_np, cam_resized, pred_label, sampled_points=points, save_path=f'{experiment_folder}/cam_overlay.png')


    # Define batch size
    BATCH_SIZE = 4

    # Randomly sample 10 samples
    num_samples = 10
    
    num_batches = num_samples // BATCH_SIZE
    
    class_indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    class_indices = np.sort(class_indices)

    def pil_to_tensor(pil_image):
        transform = transforms.Compose([
            transforms.ToTensor(),        
            transforms.Resize((224, 224)),
        ])
        return transform(pil_image)

    # Process in batches
    for batch_start in tqdm(range(0, num_batches, BATCH_SIZE)):
        batch_indices = class_indices[batch_start:batch_start+BATCH_SIZE]
        logger.info(f"Processing batch: {batch_indices}")
        # Prepare batch data
        batch_images = []
        batch_labels = []
        batch_original_images = []
        batch_classes = []
        
        for idx in batch_indices:
            image, label = test_dataset[idx]
            batch_original_images.append(image)
            batch_images.append(pil_to_tensor(image))
            batch_labels.append(label)
            batch_classes.append(test_dataset.classes[label].replace(" ", "_").lower())
        
        logger.info(f"Batch classes: {batch_classes}")
        logger.info(f"Batch labels: {batch_labels}")
        logger.info(f"Batch images: {len(batch_images)}")
        
        # Stack preprocessed images for batch processing
        batch_tensors = torch.stack(batch_images)
        logger.info(f"Stacked batch tensor shape: {batch_tensors.shape}")
        
        # Generate CAMs in batch
        batch_pred_labels, batch_cams = model.batch_generate_cam(batch_tensors, batch_labels)
        
        logger.info(f"Batch predicted labels: {batch_pred_labels}")
        
        # Process each image with its CAM
        for i in range(len(batch_indices)):
            logger.info(f"Plotting each CAM {i}")
            image = batch_original_images[i]
            label = batch_labels[i]
            class_name = batch_classes[i]
            cam = batch_cams[i]
            
            idx_in_dataset = batch_indices[i]
            output_idx = batch_start + i
            
            # Resize CAM to match image dimensions
            cam_resized = cv2.resize(cam, (image.width, image.height))
            
            # Sample points from CAM
            points = model.sample_cam_points(cam_resized, strategy='peak')
            
            # Plot CAM overlay
            model.plot_cam_overlay(
                np.array(image), 
                cam_resized, 
                label, 
                sampled_points=points, 
                save_path=f'{experiment_folder}/cam_overlay_{output_idx}.png'
            )
            