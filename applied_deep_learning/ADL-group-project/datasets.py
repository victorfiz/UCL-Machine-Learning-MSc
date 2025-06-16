import xml.etree.ElementTree as ET
import os
import torch
import numpy as np
from PIL import Image
from enum import Enum
from typing import List, Set
from torch.utils.data import Dataset
from torchvision import transforms

TF = transforms.functional

print("NEED TO HAVE FIRST RUN:\n./load_oxfordpets.sh")

# Set up the data directories
DATA_DIR = 'data/images'
ANNOTATION_DIR = 'data/annotations'

# Enum for dataset targets
DatasetSelection = Enum('DatasetSelection', [
    ('Trimap', 1),
    ('Class', 2),
    ('BBox', 3),
    ('CAM', 4),
    ('SAM', 5)
])

# Device selection
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

# Transformations
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.Lambda(lambda x: x.to(torch.half)),
    transforms.Lambda(lambda x: x.to(device).squeeze()),
])

TRIMAP_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda x: torch.tensor(np.array(x), dtype=torch.int8)),
    transforms.Lambda(lambda x: x.to(device).squeeze()),
])


class InMemoryPetSegmentationDataset(Dataset):
    def __init__(self, data_dir, annotation_dir, targets_list: List[DatasetSelection], image_transform=IMAGE_TRANSFORM, trimap_transform=TRIMAP_TRANSFORM, image_shape=(224, 224), use_augmentation=False):
        available_targets = set(DatasetSelection)
        assert set(targets_list).issubset(available_targets)
        self.targets_list = targets_list

        self.data_dir = data_dir
        self.annotation_dir = annotation_dir
        self.image_transform = image_transform
        self.trimap_transform = trimap_transform
        self.image_shape = image_shape
        self.use_augmentation = use_augmentation
        self.samples = []

        image_files = [f for f in os.listdir(
            data_dir) if f.endswith(('.jpg', '.png'))]
        self.image_ind_dict = {
            f.split('.')[0]: i for i, f in enumerate(image_files)}
        self.available_images: List[str] = sorted(
            self.image_ind_dict.keys())  #[:200]
        dataset_permutation = torch.randperm(len(self.available_images))
        self.available_images = [self.available_images[i]
                                 for i in dataset_permutation]

        self.masking_permutation = torch.randperm(len(self.available_images))
        self.selected_trimap_inds: Set[int] = set(
            range(len(self.available_images)))

        if DatasetSelection.Trimap in self.targets_list:
            self.trimap_tensor = torch.empty(
                (len(self.available_images), *image_shape), dtype=torch.int8, device=device)
            self.trimap_name_to_index = {}
            self.dummy_trimap = -100 * \
                torch.ones(image_shape, dtype=torch.int8, device=device)
        else:
            self.trimap_tensor = None
            self.trimap_name_to_index = {}

        contents = np.genfromtxt(os.path.join(annotation_dir, 'list.txt'), skip_header=6, usecols=(0, 1),
                                 dtype=[('name', np.str_, 32), ('grades', np.uint8)])
        self.labels_dict = {str(x[0]): int(x[1] - 1) for x in contents}

        xml_dir = os.path.join(annotation_dir, 'xmls')
        self.bbbox_dict = {}
        for filename in os.listdir(xml_dir):
            tree = ET.parse(os.path.join(xml_dir, filename))
            root = tree.getroot()
            xmin, ymin, xmax, ymax = (
                int(root[5][4][i].text) for i in range(4))
            width, height = int(root[3][0].text), int(root[3][1].text)
            self.bbbox_dict[root[1].text.split('.')[0]] = torch.tensor([
                xmin * image_shape[0] / width,
                ymin * image_shape[1] / height,
                xmax * image_shape[0] / width,
                ymax * image_shape[1] / height,
            ]).to(torch.uint8)

        sam_dir = os.path.join("weakly_supervised/sam_masks")
        sam_file = os.path.join(sam_dir, "triangle3-point-sam.pt")
        if os.path.exists(sam_file):
            sam_data = torch.load(sam_file)
            self.sam_data_dict = {name: {"mask": mask, "score": score} for name, mask, score in zip(
                sam_data["image_names"], sam_data["masks"], sam_data["scores"])}
            print(
                f"Loaded SAM predictions for {len(self.sam_data_dict)} images.")
        else:
            self.sam_data_dict = {}

        for idx, fname in enumerate(self.available_images, desc="Loading dataset", disable=False):
            img_path = os.path.join(data_dir, fname + '.jpg')
            img = Image.open(img_path).convert('RGB')
            img = self.image_transform(img)
            sample_data = {}

            if DatasetSelection.Trimap in self.targets_list:
                trimap = self.load_trimap(fname)
                self.trimap_tensor[idx].copy_(trimap)
                self.trimap_name_to_index[fname] = idx
                placeholder = torch.empty_like(self.dummy_trimap)
                placeholder.set_(self.dummy_trimap)
                sample_data[DatasetSelection.Trimap] = placeholder

            if DatasetSelection.Class in self.targets_list:
                sample_data[DatasetSelection.Class] = self.labels_dict.get(
                    fname, -100)

            if DatasetSelection.BBox in self.targets_list:
                sample_data[DatasetSelection.BBox] = self.bbbox_dict.get(
                    fname, -1 * torch.ones(4, dtype=torch.uint8))

            if DatasetSelection.CAM in self.targets_list:
                cam_path = os.path.join(annotation_dir, 'heatmaps', fname)
                sample_data[DatasetSelection.CAM] = (
                    torch.load(cam_path, weights_only=False)
                    if os.path.exists(cam_path)
                    else -torch.ones((7, 7))
                ).to(device)

            if DatasetSelection.SAM in self.targets_list:
                sam_pred = self.sam_data_dict.get(fname, None)
                if sam_pred is not None:
                    sample_data[DatasetSelection.SAM] = trimap_transform(
                        sam_pred["mask"][None, ...])
                else:
                    sample_data[DatasetSelection.SAM] = -100 * \
                        torch.ones(image_shape, dtype=torch.int8,
                                   device=img.device)

            self.samples.append((img, sample_data))

    def load_trimap(self, fname):
        trimap_path = os.path.join(
            self.annotation_dir, 'trimaps', fname + '.png')
        trimap = Image.open(trimap_path)
        trimap = self.trimap_transform(trimap)
        trimap[trimap == 1] = -100
        trimap[trimap == 2] = 0
        trimap[trimap == 3] = 1
        return trimap

    def change_gt_proportion(self, gt_proportion):
        assert 0.0 <= gt_proportion <= 1.0
        N = len(self.available_images)
        active_set = set(range(int(gt_proportion * N)))

        for idx in range(N):
            _, sample_data = self.samples[idx]
            if DatasetSelection.Trimap not in sample_data:
                continue
            placeholder = sample_data[DatasetSelection.Trimap]
            if idx in active_set:
                placeholder.set_(self.trimap_tensor[idx])
            else:
                placeholder.set_(self.dummy_trimap)

        self.selected_trimap_inds = active_set

    def __len__(self):
        return len(self.available_images)

    def __getitem__(self, idx):
        if not self.use_augmentation:
            return self.samples[self.masking_permutation[idx]]

        img, sample_data = self.samples[self.masking_permutation[idx]]

        # Generate random booleans for horizontal and vertical flip.
        hflip, vflip = (torch.rand((2,)) > 0.5).tolist()

        # --- Flip the image in-place ---
        # Assuming img shape is (C, H, W):
        if hflip:
            # Flip horizontally (along the width dimension, i.e. last dimension)
            flipped = torch.flip(img, dims=(-1,))
            img.copy_(flipped)

        if vflip:
            # Flip vertically (along the height dimension, i.e. second dimension)
            flipped = torch.flip(img, dims=(-2,))
            img.copy_(flipped)

        # --- Flip each target as appropriate ---
        # 1. Trimap (if available)
        if DatasetSelection.Trimap in sample_data:
            trimap = sample_data[DatasetSelection.Trimap]
            if hflip:
                trimap.copy_(torch.flip(trimap, dims=(-1,)))
            if vflip:
                trimap.copy_(torch.flip(trimap, dims=(-2,)))

        # 2. CAM target (if available)
        if DatasetSelection.CAM in sample_data:
            cam = sample_data[DatasetSelection.CAM]
            if hflip:
                cam.copy_(torch.flip(cam, dims=(-1,)))
            if vflip:
                cam.copy_(torch.flip(cam, dims=(-2,)))

        # 3. SAM target (if available)
        if DatasetSelection.SAM in sample_data:
            sam = sample_data[DatasetSelection.SAM]
            if hflip:
                sam.copy_(torch.flip(sam, dims=(-1,)))
            if vflip:
                sam.copy_(torch.flip(sam, dims=(-2,)))

        # 4. Bounding Box (if available)
        # Bounding boxes are stored as [xmin, ymin, xmax, ymax]
        if DatasetSelection.BBox in sample_data:
            bbox = sample_data[DatasetSelection.BBox]
            # For the horizontal flip, the x-coordinate must be adjusted.
            if hflip:
                img_width = self.image_shape[0]
                old_xmin = bbox[0].clone()
                old_xmax = bbox[2].clone()
                bbox[0].copy_(img_width - old_xmax)
                bbox[2].copy_(img_width - old_xmin)
            # For the vertical flip, adjust the y-coordinates.
            if vflip:
                img_height = self.image_shape[1]
                old_ymin = bbox[1].clone()
                old_ymax = bbox[3].clone()
                bbox[1].copy_(img_height - old_ymax)
                bbox[3].copy_(img_height - old_ymin)

        return img, sample_data


def save_cam_dataset(image_names, cams):
    assert len(image_names) == len(cams)
    os.makedirs(os.path.join(ANNOTATION_DIR, 'heatmaps'), exist_ok=True)
    for fname, cam in zip(image_names, cams):
        torch.save(cam, os.path.join(ANNOTATION_DIR, 'heatmaps', fname))