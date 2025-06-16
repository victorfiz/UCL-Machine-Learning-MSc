import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision.transforms.functional import to_pil_image
import numpy as np

def plot_image(img_path, postfix='Original'):
    if postfix == 'Original':
        img = Image.open(img_path).convert("RGB")
    else:
        # Load grayscale trimap
        img = Image.open(img_path).convert("L")

        # Remap values: 0 → 0, 1 → 255, 2 → 127
        remapped_img = Image.new("L", img.size)
        src = img.load()
        dst = remapped_img.load()

        for y in range(img.size[1]):
            for x in range(img.size[0]):
                val = src[x, y]
                if val == 1:
                    dst[x, y] = 0
                elif val == 2:
                    dst[x, y] = 255
                elif val == 3:
                    dst[x, y] = 127

        img = remapped_img.convert("RGB")  # Convert to RGB for display

    # Add header above image
    width, height = img.size
    font = ImageFont.load_default()
    text = f'{img_path.split("/")[-1].split("_")[0]} - {postfix}'

    dummy_img = Image.new("RGB", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    text_bbox = dummy_draw.textbbox((0, 0), text, font=font)
    text_height = text_bbox[3] - text_bbox[1]

    new_height = height + text_height + 10
    new_img = Image.new("RGB", (width, new_height), color=(255, 255, 255))

    draw = ImageDraw.Draw(new_img)
    text_position = ((width - (text_bbox[2] - text_bbox[0])) // 2, 5)
    draw.text(text_position, text, fill=(0, 0, 0), font=font)

    # Paste image below the label
    new_img.paste(img, (0, text_height + 10))

    return new_img

def combine_images_side_by_side(img, trimap_img, trimap_img2=None, cam_heatmap=None):
    img = img.convert("RGB")
    trimap_img = trimap_img.convert("RGB")

    width, height = img.size
    combined_width = width * 2
    combined_img = Image.new("RGB", (combined_width, height))

    combined_img.paste(img, (0, 0))
    combined_img.paste(trimap_img, (width, 0))

    return combined_img

def tensor_to_pil_image(tensor, normalize=False, colormap=False, trimap=False):
    if normalize:
        tensor = tensor * torch.tensor([0.229, 0.224, 0.225])[:, None, None] + \
                 torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        tensor = tensor.clip(0, 1)
    if tensor.ndim == 2:
        return to_pil_image(tensor.unsqueeze(0)) if not colormap else apply_colormap(tensor, trimap)
    elif tensor.ndim == 3:
        return to_pil_image(tensor)

def apply_heatmap(tensor):
    arr = tensor.numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    arr = (arr * 255).astype(np.uint8)

    gray_img = Image.fromarray(arr, mode='L')

    color_img = ImageOps.colorize(gray_img, black="blue", white="yellow", mid="mediumseagreen")
    return color_img

def apply_colormap(tensor, trimap=False):
    tensor = tensor.byte()
    color_img = Image.new("RGB", tensor.size())
    src = tensor.numpy()
    dst = color_img.load()
    if trimap:
        for y in range(tensor.size(0)):
            for x in range(tensor.size(1)):
                val = src[y][x]
                if val == 0:
                    dst[x, y] = (255, 255, 255)
                elif val == 1:
                    dst[x, y] = (127, 127, 127)
                elif val == 2:
                    dst[x, y] = (0, 0, 0)
                else:
                    dst[x, y] = (255, 0, 0)
    else:
        for y in range(tensor.size(0)):
            for x in range(tensor.size(1)):
                val = src[y][x]
                if val == 0:
                    dst[x, y] = (255, 255, 255)
                elif val == 1:
                    dst[x, y] = (0, 0, 0)
                elif val == 2:
                    dst[x, y] = (0, 0, 0)
                else:
                    dst[x, y] = (255, 0, 0)
    return color_img