import gc
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.models as models
import torch
import torch.nn as nn
import os
from torchvision import transforms
import torch.nn as nn
from evaluation_metrics import accuracy_score, recall_score, jaccard_score, f1_score
import numpy as np
from typing import Dict, List, Tuple, Set
from itertools import product
from datasets import *

from IPython.display import display
from PIL import Image, ImageDraw, ImageFont
from plot_utils import plot_image, combine_images_side_by_side, tensor_to_pil_image, apply_heatmap

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

def free_memory():
    gc.collect()
    torch.cuda.empty_cache()

data_dir = 'data/images'
annotation_dir = 'data/annotations'  # Add annotations directory
image_files = sorted([f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.png'))])[:4]

for idx, img_file in enumerate(image_files):
    img_path = os.path.join(data_dir, img_file)
    img = plot_image(img_path, 'Original')

    trimap_file = img_file.replace('.jpg', '.png')
    trimap_path = os.path.join(annotation_dir, 'trimaps', trimap_file)
    tri_img = plot_image(trimap_path, 'Segmentation Mask')

    combined_img = combine_images_side_by_side(img, tri_img)
    display(combined_img)

# Optional: Display annotation statistics


def print_annotation_info():
    print("\nDataset Annotations Include:")
    print("1. Species/Breed Names (37 categories)")
    print("2. Head Bounding Box (ROI)")
    print("3. Trimap Segmentation:")
    print("   - 1: Pet")
    print("   - 2: Background")
    print("   - 3: Border/Undefined")

print_annotation_info()

def visualize_predictions(model, dataset, num_samples=8, visual_fname='foo.png'):
    model.eval()
    rows = []

    for idx in range(num_samples):
        img, sample_data = dataset[idx]
        with torch.no_grad():
            pred = model(img.unsqueeze(0))
            pred_trimap = (pred[DatasetSelection.Trimap] > 0.0).squeeze().cpu()
            pred_cam = torch.sigmoid(pred[DatasetSelection.CAM]).float().squeeze().cpu()
            pred_bbox = (pred[DatasetSelection.BBox] > 0.0).squeeze().cpu()
            pred_sam = (pred[DatasetSelection.SAM] > 0.0).squeeze().cpu()

        true_mask = sample_data[DatasetSelection.Trimap]
        true_mask[true_mask == -100] = 2
        true_mask = true_mask.cpu()

        img = img.cpu()
        img_pil = tensor_to_pil_image(img, normalize=True)
        true_mask_pil = tensor_to_pil_image(true_mask, colormap=True, trimap=True)
        pred_trimap_pil = tensor_to_pil_image(pred_trimap, colormap=True)
        pred_cam_pil = apply_heatmap(pred_cam)
        pred_bbox_pil = tensor_to_pil_image(pred_bbox, colormap=True)
        pred_sam_pil = tensor_to_pil_image(pred_sam, colormap=True)

        font = ImageFont.load_default()
        imgs = [img_pil, true_mask_pil, pred_trimap_pil, pred_cam_pil, pred_bbox_pil, pred_sam_pil]
        labeled_imgs = []

        _, heights = zip(*(i.size for i in imgs))
        max_height = max(heights)
        label_height = 15

        label_map = {
            0: "Original Image",
            1: "True Mask",
            2: "Trimap Mask",
            3: "Cam Heatmap",
            4: "Bounding box",
            5: "SAM Mask"
        }

        for i, im in enumerate(imgs):
            im = im.resize((im.width, max_height))
            labeled = Image.new("RGB", (im.width, max_height + label_height), color=(255, 255, 255))
            draw = ImageDraw.Draw(labeled)
            text_width = draw.textlength(label_map[i], font=font)
            text_x = (im.width - text_width) // 2
            draw.text((text_x, 0), label_map[i], fill=(0, 0, 0), font=font)
            labeled.paste(im, (0, label_height))
            labeled_imgs.append(labeled)

        total_width = sum(i.width for i in labeled_imgs)
        combined_row = Image.new('RGB', (total_width, max_height + label_height))
        x_offset = 0
        for im in labeled_imgs:
            combined_row.paste(im, (x_offset, 0))
            x_offset += im.width

        rows.append(combined_row)

    total_height = sum(row.height for row in rows)
    full_width = rows[0].width
    final_img = Image.new('RGB', (full_width, total_height), color=(255, 255, 255))
    y_offset = 0
    for row in rows:
        final_img.paste(row, (0, y_offset))
        y_offset += row.height

    os.makedirs('visuals', exist_ok=True)
    final_img.save(os.path.join('visuals', visual_fname))
    display(final_img)


def evaluate_model_metrics(model, dataloader, device):
    """
    Evaluate model performance using various classification metrics.

    Args:
        model: PyTorch model to evaluate
        dataloader: PyTorch DataLoader containing validation/test data
        device: Device to run model on ('cuda' or 'cpu')

    Returns:
        dict: Dictionary containing computed metrics
    """
    model.eval()

    # Initialize metric accumulators
    total_accuracy = 0
    total_recall = 0
    total_jaccard = 0
    total_f1 = 0
    total_samples = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = (outputs[DatasetSelection.Trimap]
                     > 0.0).cpu().numpy().flatten()
            # Move targets to CPU
            targets = targets[DatasetSelection.Trimap].cpu().numpy().flatten()
            # mask to filter out the 'unknown' class
            mask = (targets == 0) | (targets == 1)
            preds = preds[mask]
            targets = targets[mask]

            # Calculate metrics for the batch and accumulate
            total_accuracy += accuracy_score(targets, preds) * len(targets)
            total_recall += recall_score(targets, preds, average='macro') * len(targets)
            total_jaccard += jaccard_score(targets, preds, average='macro') * len(targets)
            total_f1 += f1_score(targets, preds, average='macro') * len(targets)
            total_samples += len(targets)

    # Calculate average metrics
    metrics = {
        'accuracy': float(total_accuracy / total_samples),
        'recall': float(total_recall / total_samples),
        'jaccard': float(total_jaccard / total_samples),
        'f1': float(total_f1 / total_samples),
    }

    print(f"\nModel Performance Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.6f}")
    print(f"Recall: {metrics['recall']:.6f}")
    print(f"Jaccard Index: {metrics['jaccard']:.6f}")
    print(f"F1 Score: {metrics['f1']:.6f}")

    return metrics


def load_checkpoint(model, checkpoint_path):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['epoch'], checkpoint['best_loss']


class CustomLoss(nn.Module):
    def __init__(self, targets_weights: Dict[DatasetSelection, float]):
        super().__init__()
        self.targets_weights = targets_weights
        self.loss_funcs = {
            DatasetSelection.Trimap: TrimapLoss(),
            DatasetSelection.BBox: BBoxLoss(),
            DatasetSelection.CAM: CamLoss(),
            DatasetSelection.SAM: SAMLoss(),
        }
        assert set(self.targets_weights.keys()).issubset(
            self.loss_funcs.keys())

    def forward(self, logits, targets):
        total_loss = None
        for d in set(logits.keys()).intersection(self.targets_weights).intersection(targets):
            l = self.loss_funcs[d](logits[d], targets[d]) * \
                self.targets_weights[d]
            if not total_loss:
                total_loss = l
            else:
                total_loss += l
        return total_loss


class TrimapLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=-100)  # uses mean by default

    def forward(self, logits, targets):
        background_logits = torch.zeros_like(logits)  # fake bg logits
        logits_2ch = torch.cat([background_logits, logits], dim=1)
        res = self.loss_fn(logits_2ch, targets.long())
        return torch.nan_to_num(res)


class CamLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(
            reduction='none')  # we'll reduce manually
        self.gap = nn.AvgPool2d(32, 32)

    def forward(self, logits, targets):
        # Mask everything that is not marked with -1 (ignore signal)
        mask = (targets != -1).float()

        resized_logits = self.gap(logits).squeeze()

        loss = self.loss_fn(resized_logits, targets)
        loss = loss * mask  # zero out ignored areas

        # Avoid division by zero
        denom = mask.sum()
        if denom == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        return loss.sum() / denom


class BBoxLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, logits, bboxes):
        """
        Args:
            logits: Tensor of shape (B, C=2, H, W) — logits for 2 classes (background/foreground)
            bboxes: Tensor of shape (B, 4) — (xmin, ymin, xmax, ymax) per image; use -1 for missing
        Returns:
            Scalar loss
        """
        B, _, H, W = logits.shape
        background_logits = torch.zeros_like(logits)  # fake bg logits
        logits_2ch = torch.cat([background_logits, logits], dim=1)
        targets = torch.full((B, H, W), -100, dtype=torch.long,
                             device=logits.device)  # Default to ignore

        for i in range(B):
            # print(bboxes[i])
            xmin, ymin, xmax, ymax = bboxes[i]
            if all(bboxes[i] >= 0):  # Valid bbox
                targets[i, :, :] = 0  # Background everywhere
                targets[i, ymin:ymax, xmin:xmax] = 1  # Foreground inside bbox

        loss = self.loss_fn(logits_2ch, targets)
        return torch.nan_to_num(loss)


class SAMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=-100)  # uses mean by default

    def forward(self, logits, targets):
        background_logits = torch.zeros_like(logits)  # fake bg logits
        logits_2ch = torch.cat([background_logits, logits], dim=1)
        res = self.loss_fn(logits_2ch, targets.long())
        if torch.isnan(res).item():
            print('SAMLoss is NaN')
        return torch.nan_to_num(res)


def train_model(
    model,
    targets_weights,
    train_dataloader,
    epochs,
    learning_rate,
    optimizer_name='adam',
    scheduler_name='cosine',
    checkpoint_dir='checkpoints',
    resume_from=None
):
    """Train the segmentation model with checkpointing and scheduling"""

    # Set up optimizer
    if optimizer_name.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Set up scheduler
    if scheduler_name.lower() == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs)
    elif scheduler_name.lower() == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, factor=0.5)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    criterion = CustomLoss(targets_weights)
    end_criterion = CustomLoss(
        {t: (1.0 if t == DatasetSelection.Trimap else 0.0) for t in targets_weights})
    best_loss = float('inf')
    start_epoch = 0

    # Resume from checkpoint if specified
    if resume_from:
        model, start_epoch, best_loss = load_checkpoint(model, resume_from)
        print(
            f"Resuming from epoch {start_epoch} with best loss {best_loss:.6f}")

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("\nStarting training...")
    accumulation_steps = 2  # number of batches to accumulate gradients over

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0

        optimizer.zero_grad()  # initialize gradients at start of epoch

        for i, (images, image_data) in enumerate(train_dataloader, disable=True):
            # images = images.to(device)
            # image_data = image_data.to(device)

            outputs = model(images)

            if (epoch + 1) / epochs <= 0.8:
                loss = criterion(outputs, image_data)
            else:
                loss = end_criterion(outputs, image_data)

            loss = loss / accumulation_steps  # normalize loss for accumulation
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * accumulation_steps  # un-normalize to keep consistent logging

        # Optionally: handle remainder batches
        if (i + 1) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()


            del image_data[DatasetSelection.BBox]

            free_memory()

        epoch_loss = running_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")

        # Update scheduler
        if scheduler_name.lower() == 'cosine':
            scheduler.step()
        else:
            scheduler.step(epoch_loss)

        # Save checkpoint if best loss
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'best_loss': best_loss
            }
            torch.save(checkpoint, f"{checkpoint_dir}/best_model.pth")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'best_loss': best_loss
            }
            torch.save(
                checkpoint, f"{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth")

    print("\nTraining complete!")
    return model


print("\n=== Using Segmentation Models PyTorch (SMP) for improved performance ===\n")
TARGETS_LIST = [DatasetSelection.CAM,
                DatasetSelection.Trimap, DatasetSelection.BBox, DatasetSelection.SAM]
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-3
OPTIMIZER_NAME = 'adam'
SCHEDULER_NAME = 'reduce_on_plateau'
CHECKPOINT_DIR = 'checkpoints/'
RESUME_FROM = None  # os.path.join(CHECKPOINT_DIR, 'checkpoint_epoch_30.pth')
seed = 42

torch.manual_seed(seed)


class ConvSegHead(nn.Module):
    def __init__(self, out_channels=1):
        super(ConvSegHead, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.block(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class ResNetUNet(nn.Module):
    def __init__(self, out_channels, pretrained=True):
        super(ResNetUNet, self).__init__()
        
        # Load pretrained ResNet34
        resnet = models.resnet34(pretrained=pretrained)
        
        # Encoder (ResNet layers)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1  # 64 channels
        self.encoder2 = resnet.layer2  # 128 channels
        self.encoder3 = resnet.layer3  # 256 channels
        self.encoder4 = resnet.layer4  # 512 channels
        
        # Decoder - Fix channel dimensions to match ResNet34 output sizes
        self.decoder4 = DoubleConv(512, 256)
        self.decoder3 = DoubleConv(256 + 256, 128)
        self.decoder2 = DoubleConv(128 + 128, 64)
        self.decoder1 = DoubleConv(64 + 64, 64)
        self.decoder0 = DoubleConv(64 + 64, 64)
        self.last_conv = DoubleConv(64 + 64, 64)
        
        # Upsampling
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Additional conv to handle skip from first maxpool
        self.conv_original_size = DoubleConv(3, 64)
        
        # # Final layer
        # self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Save input dimensions for later use
        h, w = x.size()[2:]
        
        # Original size
        x_original = self.conv_original_size(x)
        
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x_conv = x
        x = self.firstmaxpool(x)
        
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        
        # Decoder with skip connections
        # Use size matching for each level
        x = self.up(x4)
        x = self.decoder4(x)
        
        # Ensure x3 and x have same spatial dimensions before concatenating
        x = self.up(x)
        if x.size()[2:] != x3.size()[2:]:
            x = F.interpolate(x, size=x3.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, x3], dim=1)
        x = self.decoder3(x)
        
        x = self.up(x)
        if x.size()[2:] != x2.size()[2:]:
            x = F.interpolate(x, size=x2.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, x2], dim=1)
        x = self.decoder2(x)
        
        x = self.up(x)
        if x.size()[2:] != x1.size()[2:]:
            x = F.interpolate(x, size=x1.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, x1], dim=1)
        x = self.decoder1(x)
        
        x = self.up(x)
        if x.size()[2:] != x_conv.size()[2:]:
            x = F.interpolate(x, size=x_conv.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, x_conv], dim=1)
        x = self.decoder0(x)
        
        x = self.up(x)
        if x.size()[2:] != x_original.size()[2:]:
            x = F.interpolate(x, size=x_original.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, x_original], dim=1)
        x = self.last_conv(x)
        
        # # Final layer
        # return self.final(x)

        # skip segmentation head
        return x

class CustomUNet(nn.Module):
    def __init__(self):
        super(CustomUNet, self).__init__()
        # self.feature_extractor = smp.Unet(
        #     encoder_name="resnet34",
        #     encoder_weights="imagenet",
        #     in_channels=3,
        # )
        self.feature_extractor = ResNetUNet(out_channels=1)
        # self.feature_extractor.segmentation_head = nn.Identity()

        self.seg_heads = nn.ModuleDict()
        self.seg_heads[DatasetSelection.Trimap.name] = ConvSegHead()
        self.seg_heads[DatasetSelection.CAM.name] = ConvSegHead()
        self.seg_heads[DatasetSelection.BBox.name] = ConvSegHead()
        self.seg_heads[DatasetSelection.SAM.name] = ConvSegHead()

    def forward(self, img):
        features = self.feature_extractor(img)
        res = {}
        for dataset_name, head in self.seg_heads.items():
            res[DatasetSelection[dataset_name]] = head(features)
        return res


# Define transformations for training
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.to(device).squeeze()),
])

data_transform = transforms.Compose([
    base_transform,
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.Lambda(lambda x: x.to(torch.float32)),
])

target_transform = transforms.Compose([
    base_transform,
    transforms.Lambda(lambda x: x.to(torch.int8)),
])

print('Loading Dataset')
dataset = InMemoryPetSegmentationDataset(
    DATA_DIR, ANNOTATION_DIR, targets_list=TARGETS_LIST)
# dataset_perm = torch.randperm(len(dataset))

GT_PROPORTIONS = [0.1]
LOSS_WEIGHTS = [0.0, 1.0]

for idx, experiment_weights in enumerate(product(GT_PROPORTIONS, LOSS_WEIGHTS, LOSS_WEIGHTS, LOSS_WEIGHTS)):
    free_memory()

    # print('MEM at start: ', torch.cuda.memory_summary(
    #     device=device, abbreviated=True))
    gt_prop, cam_loss_weight, bbox_loss_weight, sam_loss_weight = experiment_weights
    print('gt_prop, cam_loss_weight, bbox_loss_weight, sam_loss_weight')
    print(gt_prop, cam_loss_weight, bbox_loss_weight, sam_loss_weight)
    # set based on how much of the dataset we can use
    dataset.change_gt_proportion(gt_prop)
    dataset.use_augmentation = True

    # Create train/val split
    train_size = int(0.8*len(dataset))
    val_size = len(dataset) - train_size

    train_dataset = torch.utils.data.Subset(
        dataset, range(train_size))

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Create SMP model - Using UNet with ResNet34 encoder pre-trained on ImageNet
    smp_model = CustomUNet().to(device)

    model = train_model(
        smp_model,
        {DatasetSelection.Trimap: 1.0, DatasetSelection.CAM: cam_loss_weight,
            DatasetSelection.BBox: bbox_loss_weight, DatasetSelection.SAM: sam_loss_weight},
        train_dataloader,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        optimizer_name=OPTIMIZER_NAME,
        scheduler_name=SCHEDULER_NAME,
        checkpoint_dir=CHECKPOINT_DIR,
        resume_from=RESUME_FROM
    )
    smp_model, smp_epoch, smp_best_loss = load_checkpoint(
        smp_model, 'checkpoints/best_model.pth')
    print("Epoch with lowest loss: ", smp_epoch)

    # reset GT proportion to perform evaluation on trimaps correctly
    dataset.change_gt_proportion(1.0)
    dataset.use_augmentation = False
    val_dataset = torch.utils.data.Subset(
        dataset, range(train_size, train_size + val_size))
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    metrics = visualize_predictions(
        smp_model, val_dataset, visual_fname=f"{idx}.jpg")
    metrics = evaluate_model_metrics(smp_model, val_dataloader, device)
    os.makedirs('run_results', exist_ok=True)
    with open(f'run_results/{idx}.txt', 'w') as file:
        res_str = str(experiment_weights)+'\n'+str(metrics)
        file.write(res_str)

    del train_dataset, val_dataset, train_dataloader, val_dataloader, model, smp_model