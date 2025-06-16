import numpy as np
import os

# import matplotlib.pyplot as plt
# def evaluate_and_visualize_trimaps(results, experiment_folder, iou_threshold=0.5):
#     """
#     Evaluate segmentation and classification results with trimap ground truth.
#     Uses only NumPy and Matplotlib.
    
#     Args:
#         results: List of result dictionaries from SAM predictions
#         experiment_folder: Folder to save visualizations and metrics
#         iou_threshold: Threshold to consider a mask as correct
#     """
#     # Create metrics subfolder
#     metrics_folder = os.path.join(experiment_folder, 'metrics')
#     os.makedirs(metrics_folder, exist_ok=True)
    
#     # Initialize metrics
#     classification_metrics = {
#         'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0,
#         'class_correct': 0
#     }
    
#     segmentation_metrics = {
#         'iou': [],
#         'dice': [],
#         'pixel_precision': [],
#         'pixel_recall': [],
#         'pixel_accuracy': [],
#         'boundary_f1': [],
#         'has_gt_mask': 0,
#         'has_pred_mask': 0,
#         'correct_masks': 0
#     }
    
#     # Process each result
#     for result in results:
#         # Classification evaluation
#         pred_label = result['pred_labels']
#         true_label = result['true_labels']
        
#         if pred_label == true_label:
#             classification_metrics['class_correct'] += 1
        
#         # Segmentation evaluation
#         masks = result['masks']
#         trimap = np.array(result['mask_labels'])
        
#         # Convert trimap to binary mask:
#         # In trimap: 0=background, 1=foreground, 2=unknown
#         # For evaluation: Consider only definite foreground (1) as foreground
#         gt_mask = (trimap == 1).astype(np.uint8)
        
#         # Create evaluation mask that ignores unknown regions
#         # This mask indicates pixels we should consider in the evaluation
#         evaluation_mask = (trimap != 2).astype(np.uint8)
        
#         # Skip if ground truth mask is empty or all zeros/unknown
#         if gt_mask.sum() == 0:
#             continue
            
#         segmentation_metrics['has_gt_mask'] += 1
        
#         # Skip if no predicted masks
#         if len(masks) == 0:
#             continue
            
#         segmentation_metrics['has_pred_mask'] += 1
        
#         # Get the best mask (highest IoU with ground truth)
#         best_iou = 0
#         best_mask = None
        
#         for mask in masks:
#             # Only consider pixels not in the "unknown" region of the trimap
#             valid_pixels = evaluation_mask == 1
#             valid_mask = mask * valid_pixels
#             valid_gt = gt_mask * valid_pixels
            
#             # Calculate IoU on valid regions only
#             intersection = np.logical_and(valid_mask, valid_gt).sum()
#             union = np.logical_or(valid_mask, valid_gt).sum()
#             iou = intersection / union if union > 0 else 0
            
#             if iou > best_iou:
#                 best_iou = iou
#                 best_mask = mask
        
#         if best_mask is not None:
#             # Calculate metrics for the best mask (only on valid regions)
#             mask_pred = best_mask * evaluation_mask
#             mask_true = gt_mask * evaluation_mask
            
#             # IoU
#             intersection = np.logical_and(mask_pred, mask_true).sum()
#             union = np.logical_or(mask_pred, mask_true).sum()
#             iou = intersection / union if union > 0 else 0
#             segmentation_metrics['iou'].append(iou)
            
#             # Dice coefficient
#             dice = (2 * intersection) / (mask_pred.sum() + mask_true.sum()) if (mask_pred.sum() + mask_true.sum()) > 0 else 0
#             segmentation_metrics['dice'].append(dice)
            
#             # Pixel-wise precision and recall
#             true_pos = intersection
#             false_pos = mask_pred.sum() - true_pos
#             false_neg = mask_true.sum() - true_pos
            
#             pixel_precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
#             pixel_recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            
#             segmentation_metrics['pixel_precision'].append(pixel_precision)
#             segmentation_metrics['pixel_recall'].append(pixel_recall)
            
#             # Pixel-wise accuracy (on valid regions only)
#             total_valid_pixels = evaluation_mask.sum()
#             correct_pixels = ((mask_pred == mask_true) & (evaluation_mask == 1)).sum()
#             pixel_accuracy = correct_pixels / total_valid_pixels if total_valid_pixels > 0 else 0
#             segmentation_metrics['pixel_accuracy'].append(pixel_accuracy)
            
#             # Simplified boundary F1
#             # For trimaps, focus on boundaries of the definite foreground region
#             def compute_boundary(mask, kernel_size=3):
#                 h, w = mask.shape
#                 boundary = np.zeros_like(mask)
                
#                 for i in range(h):
#                     for j in range(w):
#                         if mask[i, j]:
#                             # Check if this pixel is at a boundary
#                             i_min = max(0, i - 1)
#                             i_max = min(h, i + 2)
#                             j_min = max(0, j - 1)
#                             j_max = min(w, j + 2)
                            
#                             if not np.all(mask[i_min:i_max, j_min:j_max]):
#                                 boundary[i, j] = 1
                                
#                 return boundary
            
#             # Extract boundaries, considering only valid regions
#             mask_pred_boundary = compute_boundary(mask_pred) & evaluation_mask
#             mask_true_boundary = compute_boundary(mask_true) & evaluation_mask
            
#             boundary_intersection = np.logical_and(mask_pred_boundary, mask_true_boundary).sum()
#             boundary_precision = boundary_intersection / mask_pred_boundary.sum() if mask_pred_boundary.sum() > 0 else 0
#             boundary_recall = boundary_intersection / mask_true_boundary.sum() if mask_true_boundary.sum() > 0 else 0
#             boundary_f1 = 2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall) if (boundary_precision + boundary_recall) > 0 else 0
            
#             segmentation_metrics['boundary_f1'].append(boundary_f1)
            
#             # Check if this mask exceeds the IoU threshold
#             if iou >= iou_threshold:
#                 segmentation_metrics['correct_masks'] += 1
#                 # Update confusion matrix for the class
#                 classification_metrics['tp'] += 1
#             else:
#                 classification_metrics['fp'] += 1
                
#             # Save visualization of the trimap vs. predicted mask
#             plt.figure(figsize=(15, 5))
            
#             # Plot original trimap
#             plt.subplot(1, 3, 1)
#             trimap_vis = np.zeros((*trimap.shape, 3), dtype=np.uint8)
#             trimap_vis[trimap == 0] = [0, 0, 255]    # Background in blue
#             trimap_vis[trimap == 1] = [0, 255, 0]    # Foreground in green
#             trimap_vis[trimap == 2] = [255, 0, 0]    # Unknown in red
#             plt.imshow(trimap_vis)
#             plt.title('Trimap Ground Truth')
#             plt.axis('off')
            
#             # Plot binary ground truth
#             plt.subplot(1, 3, 2)
#             plt.imshow(gt_mask, cmap='gray')
#             plt.title('Binary GT (Foreground only)')
#             plt.axis('off')
            
#             # Plot predicted mask
#             plt.subplot(1, 3, 3)
#             plt.imshow(best_mask, cmap='gray')
#             plt.title(f'Predicted Mask (IoU: {iou:.3f})')
#             plt.axis('off')
            
#             # Save individual comparison
#             os.makedirs(f"{metrics_folder}/mask_comparisons", exist_ok=True)
#             plt.savefig(f"{metrics_folder}/mask_comparisons/sample_{len(segmentation_metrics['iou'])}.png")
#             plt.close()
    
#     # Calculate classification metrics
#     total_samples = len(results)
#     classification_accuracy = classification_metrics['class_correct'] / total_samples if total_samples > 0 else 0
    
#     # Convert segmentation metrics to numpy arrays
#     for key in ['iou', 'dice', 'pixel_precision', 'pixel_recall', 'pixel_accuracy', 'boundary_f1']:
#         segmentation_metrics[key] = np.array(segmentation_metrics[key])
    
#     # Calculate averages and standard deviations
#     avg_metrics = {}
#     std_metrics = {}
    
#     for key in ['iou', 'dice', 'pixel_precision', 'pixel_recall', 'pixel_accuracy', 'boundary_f1']:
#         if len(segmentation_metrics[key]) > 0:
#             avg_metrics[key] = np.mean(segmentation_metrics[key])
#             std_metrics[key] = np.std(segmentation_metrics[key])
#         else:
#             avg_metrics[key] = 0
#             std_metrics[key] = 0
    
#     # Calculate correct mask ratio
#     total_with_gt = segmentation_metrics['has_gt_mask']
#     segmentation_metrics['correct_mask_ratio'] = segmentation_metrics['correct_masks'] / total_with_gt if total_with_gt > 0 else 0
    
#     # Print metrics summary
#     print("\nCLASSIFICATION METRICS:")
#     print(f"Accuracy: {classification_accuracy:.4f}")
    
#     print("\nSEGMENTATION METRICS (evaluated on non-ambiguous regions only):")
#     print(f"Mean IoU: {avg_metrics['iou']:.4f} ± {std_metrics['iou']:.4f}")
#     print(f"Mean Dice: {avg_metrics['dice']:.4f} ± {std_metrics['dice']:.4f}")
#     print(f"Mean Pixel Precision: {avg_metrics['pixel_precision']:.4f} ± {std_metrics['pixel_precision']:.4f}")
#     print(f"Mean Pixel Recall: {avg_metrics['pixel_recall']:.4f} ± {std_metrics['pixel_recall']:.4f}")
#     print(f"Mean Pixel Accuracy: {avg_metrics['pixel_accuracy']:.4f} ± {std_metrics['pixel_accuracy']:.4f}")
#     print(f"Mean Boundary F1: {avg_metrics['boundary_f1']:.4f} ± {std_metrics['boundary_f1']:.4f}")
#     print(f"Correct Mask Ratio (IoU > {iou_threshold}): {segmentation_metrics['correct_mask_ratio']:.4f}")
#     print(f"Samples with GT masks: {segmentation_metrics['has_gt_mask']}/{total_samples}")
#     print(f"Samples with predicted masks: {segmentation_metrics['has_pred_mask']}/{segmentation_metrics['has_gt_mask']}")
    
#     # Save metrics to text file
#     with open(f"{metrics_folder}/metrics_summary.txt", "w") as f:
#         f.write("CLASSIFICATION METRICS:\n")
#         f.write(f"Accuracy: {classification_accuracy:.4f}\n\n")
        
#         f.write("SEGMENTATION METRICS (evaluated on non-ambiguous regions only):\n")
#         f.write(f"Mean IoU: {avg_metrics['iou']:.4f} ± {std_metrics['iou']:.4f}\n")
#         f.write(f"Mean Dice: {avg_metrics['dice']:.4f} ± {std_metrics['dice']:.4f}\n")
#         f.write(f"Mean Pixel Precision: {avg_metrics['pixel_precision']:.4f} ± {std_metrics['pixel_precision']:.4f}\n")
#         f.write(f"Mean Pixel Recall: {avg_metrics['pixel_recall']:.4f} ± {std_metrics['pixel_recall']:.4f}\n")
#         f.write(f"Mean Pixel Accuracy: {avg_metrics['pixel_accuracy']:.4f} ± {std_metrics['pixel_accuracy']:.4f}\n")
#         f.write(f"Mean Boundary F1: {avg_metrics['boundary_f1']:.4f} ± {std_metrics['boundary_f1']:.4f}\n")
#         f.write(f"Correct Mask Ratio (IoU > {iou_threshold}): {segmentation_metrics['correct_mask_ratio']:.4f}\n")
#         f.write(f"Samples with GT masks: {segmentation_metrics['has_gt_mask']}/{total_samples}\n")
#         f.write(f"Samples with predicted masks: {segmentation_metrics['has_pred_mask']}/{segmentation_metrics['has_gt_mask']}\n")
    
#     # VISUALIZATIONS (same as before but with updated metrics)
#     # [Include the same visualization code as in the previous function]
    
#     # 1. IoU Distribution
#     plt.figure(figsize=(10, 6))
#     plt.hist(segmentation_metrics['iou'], bins=20, alpha=0.7, color='blue', edgecolor='black')
#     plt.axvline(x=iou_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({iou_threshold})')
#     plt.title('Distribution of IoU Scores', fontsize=14)
#     plt.xlabel('IoU Score', fontsize=12)
#     plt.ylabel('Frequency', fontsize=12)
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f"{metrics_folder}/iou_distribution.png")
#     plt.close()
    
#     # 2. IoU vs Dice Scatter Plot
#     plt.figure(figsize=(8, 8))
#     plt.scatter(segmentation_metrics['iou'], segmentation_metrics['dice'], alpha=0.7, s=50)
#     plt.plot([0, 1], [0, 1], 'r--', linewidth=2)  # Line showing y=x
#     plt.title('IoU vs Dice Coefficient', fontsize=14)
#     plt.xlabel('IoU Score', fontsize=12)
#     plt.ylabel('Dice Coefficient', fontsize=12)
#     plt.xlim(0, 1)
#     plt.ylim(0, 1)
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(f"{metrics_folder}/iou_vs_dice.png")
#     plt.close()
    
#     # 3. Precision-Recall Scatter Plot
#     plt.figure(figsize=(8, 8))
#     plt.scatter(segmentation_metrics['pixel_precision'], segmentation_metrics['pixel_recall'], alpha=0.7, s=50)
#     plt.title('Pixel-wise Precision vs Recall', fontsize=14)
#     plt.xlabel('Precision', fontsize=12)
#     plt.ylabel('Recall', fontsize=12)
#     plt.xlim(0, 1)
#     plt.ylim(0, 1)
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(f"{metrics_folder}/precision_recall.png")
#     plt.close()
    
#     # 4. Bar chart of average metrics
#     plt.figure(figsize=(12, 6))
#     metric_names = ['IoU', 'Dice', 'Precision', 'Recall', 'Accuracy', 'Boundary F1']
#     metric_values = [avg_metrics['iou'], avg_metrics['dice'], 
#                     avg_metrics['pixel_precision'], avg_metrics['pixel_recall'], 
#                     avg_metrics['pixel_accuracy'], avg_metrics['boundary_f1']]
    
#     bars = plt.bar(metric_names, metric_values, color='skyblue', edgecolor='black')
    
#     # Add error bars
#     error_values = [std_metrics['iou'], std_metrics['dice'], 
#                 std_metrics['pixel_precision'], std_metrics['pixel_recall'], 
#                 std_metrics['pixel_accuracy'], std_metrics['boundary_f1']]
    
#     plt.errorbar(metric_names, metric_values, yerr=error_values, fmt='none', ecolor='black', capsize=5)
    
#     # Add value labels on the bars
#     for bar, value in zip(bars, metric_values):
#         plt.text(bar.get_x() + bar.get_width()/2, value + 0.01, f'{value:.3f}', 
#                 ha='center', va='bottom', fontsize=10)
    
#     plt.title('Average Segmentation Metrics', fontsize=14)
#     plt.ylabel('Score', fontsize=12)
#     plt.ylim(0, 1.1)  # Make sure there's space for the error bars and text
#     plt.grid(axis='y', alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(f"{metrics_folder}/average_metrics.png")
#     plt.close()
    
#     # Return the metrics
#     return {
#         'classification_accuracy': classification_accuracy,
#         'avg_metrics': avg_metrics,
#         'std_metrics': std_metrics,
#         'correct_mask_ratio': segmentation_metrics['correct_mask_ratio']
#     }
    
print("Matplotlib is not installed. Skipping visualization.")
import numpy as np

def evaluate_trimaps(results, experiment_folder, iou_threshold=0.5):
    """
    Evaluate segmentation and classification results with trimap ground truth.
    Uses only NumPy for calculations.
    
    Args:
        results: List of result dictionaries from SAM predictions
        experiment_folder: Folder to save metrics files
        iou_threshold: Threshold to consider a mask as correct
        
    Returns:
        Dictionary of computed metrics
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
    
    # Return the metrics
    return {
        'classification_accuracy': classification_accuracy,
        'avg_metrics': avg_metrics,
        'std_metrics': std_metrics,
        'correct_mask_ratio': segmentation_metrics['correct_mask_ratio']
    }