import os
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

from .config import SOURCE_IMAGES_DIR, OUTPUT_SEGMENTS_DIR, CHECKPOINT_DIR, IMG_SIZE, DEVICE

os.makedirs(OUTPUT_SEGMENTS_DIR, exist_ok=True)
processor = SegformerImageProcessor.from_pretrained(str(CHECKPOINT_DIR))
model = SegformerForSemanticSegmentation.from_pretrained(str(CHECKPOINT_DIR))
model.to(DEVICE)
model.eval()
image_paths = sorted(glob(os.path.join(str(SOURCE_IMAGES_DIR), "*.jpg")))

for img_path in tqdm(image_paths, desc="Segmenting images"):
    image = Image.open(img_path).convert("RGB")
    width, height = image.size
    inputs = processor(image, size=IMG_SIZE, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(DEVICE)
    with torch.no_grad():
        raw_logits = model(pixel_values=pixel_values).logits
        logits = F.interpolate(raw_logits, size=(height, width), mode="bilinear", align_corners=False)
    pred_mask = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    rgb = np.array(image)
    mask_3c = np.stack([pred_mask]*3, axis=-1)
    road_only = rgb * mask_3c
    seg_img = Image.fromarray(road_only)
    fname = os.path.splitext(os.path.basename(img_path))[0] + "_segment.png"
    seg_img.save(os.path.join(str(OUTPUT_SEGMENTS_DIR), fname))

print("Done! Segmented road images written to:", OUTPUT_SEGMENTS_DIR)
