
import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
# import matplotlib.pyplot as plt
from PIL import Image
import sys
import torchvision.transforms as transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader
import torch.utils.data
import functools
from torchvision import transforms
import torch.nn.functional as F


sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'sam2')))
from utils.tqdm import tqdm # homemade tqdm


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
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )
    
np.random.seed(3)
    
sys.path.insert(0, f"{os.path.abspath('')}/..")
from utils.logger import setup_folder_and_logger
experiment_folder, logger = setup_folder_and_logger(__name__)

def resize_tensor(tensor, size):
    """Resize a numpy array using PyTorch's interpolate function"""
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor).float()
    
    # Add batch and channel dimensions if needed
    if len(tensor.shape) == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    
    # Resize tensor
    resized = F.interpolate(tensor, size=(size[1], size[0]), mode='bilinear', align_corners=False)
    
    # Convert back to numpy and original format
    return resized.squeeze(0).squeeze(0).numpy()

def threshold_binary(arr, thresh, maxval):
    """Apply binary threshold to array"""
    binary = np.zeros_like(arr)
    binary[arr > thresh] = maxval
    return binary
def apply_jet_colormap(gray_img):
    """Apply a jet colormap to a grayscale image without matplotlib"""
    # Ensure values are in [0, 1] range
    gray_normalized = np.clip(gray_img, 0, 1)
    
    # Create RGB channels for jet colormap
    r = np.zeros_like(gray_normalized)
    g = np.zeros_like(gray_normalized)
    b = np.zeros_like(gray_normalized)
    
    # Red channel
    r = np.where(gray_normalized < 0.5, 0, np.where(gray_normalized < 0.8, (gray_normalized - 0.5) / 0.3, 1.0))
    
    # Green channel
    g = np.where(gray_normalized < 0.2, 0, np.where(gray_normalized < 0.5, (gray_normalized - 0.2) / 0.3, 
                np.where(gray_normalized < 0.8, 1.0, (1.0 - gray_normalized) / 0.2)))
    
    # Blue channel
    b = np.where(gray_normalized < 0.2, (0.5 + gray_normalized) / 0.2, np.where(gray_normalized < 0.5, 1.0, 
                np.where(gray_normalized < 0.8, (0.8 - gray_normalized) / 0.3, 0)))
    
    # Stack RGB channels and convert to uint8
    rgb = np.stack([r, g, b], axis=-1) * 255
    return rgb.astype(np.uint8)

def add_weighted(img1, alpha, img2, beta, gamma=0):
    """Blend two images with weights"""
    return np.clip(img1 * alpha + img2 * beta + gamma, 0, 255).astype(np.uint8)

# === ARGUMENT PARSER ===
import argparse
parser = argparse.ArgumentParser(description="Process images with SAM2.")
parser.add_argument(
    "--dataset_size",
    type=int,
    default=8,
    help="Number of images to process. Default is take all.",
)
args = parser.parse_args()


if __name__ == "__main__":
    # === PATH SETUP ===
    from torchvision.datasets import OxfordIIITPet    
    # if base_dir not exist, download data
    # if not os.path.exists(base_dir):
    # Load the dataset
    print("Loading dataset...")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', "oxford-iiit-pet"))
    annotations_dir = os.path.join(base_dir, 'annotations')
    images_dir = os.path.join(base_dir, 'images')
    xml_dir = os.path.join(annotations_dir, 'xmls')
    list_file = os.path.join(annotations_dir, 'list.txt')
    output_dir = os.path.join(os.path.dirname(__file__), 'sam_masks')
    os.makedirs(output_dir, exist_ok=True)
    

    from torchvision.datasets import OxfordIIITPet    
    train_dataset = OxfordIIITPet(root=base_dir.rsplit('/', 1)[0],
                                split='trainval',
                                target_types=["category", "segmentation"],
                                download=True)

    test_dataset = OxfordIIITPet(root=base_dir.rsplit('/', 1)[0],
                            split='test',
                            target_types=["category", "segmentation"],
                            download=True)

    print("base_dir", os.listdir(base_dir))
    print("annotations_dir:", os.listdir(annotations_dir))
    
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if not os.path.exists("./models"): os.makedirs("./models")
    # curl from <url> to ./models/sam2.pt
    if not os.path.exists("./models/sam2.pt"):
        print("Downloading SAM2 model...")
        import subprocess
        try:
            result = subprocess.run([
                "curl", "-L", "-o", "./models/sam2.pt", 
                "https://www.dropbox.com/scl/fi/zttjbnyc9d85g191m3i2m/sam2.pt?rlkey=f19ttsgqpqdfhfpq1ktq3fkmt&st=wglxuapy&dl=1"
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("Download completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error downloading model: {e}")
            print(f"stderr: {e.stderr.decode()}")
            sys.exit(1)
            
    if not os.path.exists("./models/cam_model.pt"):
        print("Downloading SAM2 model...")
        import subprocess
        try:
            result = subprocess.run([
                "curl", "-L", "-o", "./models/cam_model.pt", 
                "https://www.dropbox.com/scl/fi/3lglht1bb9xbqlsied8xm/cam_model.pth?rlkey=3ku9bxmnovmhizzpu1sw6vfkx&st=xj14bcpo&dl=1"
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("Download completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error downloading model: {e}")
            print(f"stderr: {e.stderr.decode()}")
            sys.exit(1)
            
    sam2_model = torch.load("./models/sam2.pt")

    predictor = SAM2ImagePredictor(sam2_model)
    
    from cam import CAMModel

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # make new CAMModel and load the pre-trained weights
    model = CAMModel()
    model.load_state_dict(torch.load('./models/cam_model.pt', map_location=device, weights_only=False))
    model = model.to(device)
    
    # Define a custom collate function that doesn't try to stack the PIL images
    def custom_collate(batch):
        images = [item[0] for item in batch]  # Keep as list of PIL images
        # labels = [torch.tensor(item[1]) for item in batch]  # Convert labels to tensors
        # labels are now a integer (class) and a segmentation mask (PIL)
        labels = [item[1][0] for item in batch]  # Keep as integer class
        masks = [item[1][1] for item in batch]  # Keep as segmentation masks
        
        return images, labels, masks

    # Define batch size
    # Randomly sample 8 indices
    num_samples = args.dataset_size
    BATCH_SIZE = 4
    PLOT=False

    np.random.seed(42)  # for reproducibility
    selected_indices = np.random.choice(len(test_dataset), num_samples, replace=False)

    # Create a Subset of the dataset with just our selected indices
    from torch.utils.data import Subset
    selected_dataset = Subset(test_dataset, selected_indices)

    # Create a DataLoader with custom collate function
    dataloader = torch.utils.data.DataLoader(
        selected_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate  # Use our custom collate function
    )

    # Process all batches
    results = []
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        # Unpack batch
        images, labels, mask_labels = batch
        # segmentation_labels = [label[1] for label in labels]
        # labels = [label[0] for label in labels]
        
        print("Size of images: ", images[0].width, images[0].height)
        
        # Apply preprocessing in batch using map
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Convert images to tensors
        tensor_images = list(map(preprocess, images))
        tensor_batch = torch.stack(tensor_images)
        
        batch_pred_labels, batch_cams = model.batch_generate_cam(tensor_batch, labels)
        
        logger.info(f"True labels: {labels} Predicted labels: {batch_pred_labels}")

        # Resize CAM back to the individual image's dimensions
        for i, cam in enumerate(batch_cams):
            image = images[i]
            cam_resized = resize_tensor(cam, (image.width, image.height))
            batch_cams[i] = cam_resized
            
        batch_points = list(map(functools.partial(model.sample_cam_points, strategy='peak'), batch_cams))
        
        # Process each image with its CAM
        for i, (image, label, cam_resized, points) in enumerate(zip(images, labels, batch_cams, batch_points)):
            
            # if PLOT:
            #     model.plot_cam_overlay(
            #         np.array(image),
            #         cam_resized, 
            #         label, 
            #         sampled_points=points, 
            #         save_path=f'{experiment_folder}/sam_{batch_idx}_{class_}.png'
            #     )
            
            
            # SAM the CAM
            predictor.set_image(image)
            masks, scores, logits = predictor.predict(
                point_coords=points,
                point_labels=np.repeat(1, len(points)),
                multimask_output=True,
            )
            
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            scores = scores[sorted_ind]
            logits = logits[sorted_ind]
            mask_input = logits[np.argmax(scores), :, :]
            masks, scores, _ = predictor.predict(
                point_coords=points,
                point_labels=np.repeat(1, len(points)),
                mask_input=mask_input[None, :, :],
                multimask_output=True,
            )
            

            logit_mask = np.sum(logits, axis=0)
            logit_mask = resize_tensor(logit_mask, (image.width, image.height))
            
            binary_cam_mask = threshold_binary(cam_resized, 0.7, 1)
            
            heatmap = apply_jet_colormap(binary_cam_mask)
            
            # overlay cam_resized onto image
            img_array = np.array(image)
            overlay = add_weighted(img_array, 0.8, heatmap, 0.2)
                
                
            # remove masks that have area smaller than binary_mask
            idx_to_keep = np.sum(masks, axis=(1, 2)) > np.sum(binary_cam_mask)
            if np.sum(idx_to_keep) > 0:
                masks = masks[idx_to_keep]
                scores = scores[idx_to_keep]

            # sort masks by score
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            scores = scores[sorted_ind]

            # if PLOT:
            #     show_masks(overlay, masks, scores, point_coords=points, input_labels=np.repeat(1, len(points)), savepath=f'{experiment_folder}/sam_{batch_idx}_{class_}.png')

            # Save results in compressed format to calculate metrics later
            results.append({
                "pred_labels": batch_pred_labels[i],
                "true_labels": label,
                "masks": masks.astype(np.uint8),
                "mask_labels": np.array(mask_labels[i]),
            })

    # save results
    import pickle
    with open(f"{experiment_folder}/results.pkl", "wb") as f:
        pickle.dump(results, f)
        

    # load results
    with open(f"{experiment_folder}/results.pkl", "rb") as f:
        results = pickle.load(f)

    from utils.metrics import evaluate_trimaps
    # Run evaluation and visualization
    metrics = evaluate_trimaps(results, experiment_folder, iou_threshold=0.5)

    # Access metrics if needed
    print(f"Classification accuracy: {metrics['classification_accuracy']}")
    print(f"Average IoU: {metrics['avg_metrics']['iou']}")