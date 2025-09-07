import argparse
import os
import urllib

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision.transforms.functional as TF
from sklearn.decomposition import PCA

PATCH_SIZE = 16
IMAGE_SIZE = 1080

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DINOV3_GITHUB_LOCATION = "facebookresearch/dinov3"
MODEL_DINOV3_VITS = "dinov3_vits16"
MODEL_DINOV3_VITSP = "dinov3_vits16plus"
MODEL_DINOV3_VITB = "dinov3_vitb16"
MODEL_DINOV3_VITL = "dinov3_vitl16"
MODEL_DINOV3_VITHP = "dinov3_vith16plus"
MODEL_DINOV3_VIT7B = "dinov3_vit7b16"

MODEL_TO_NUM_LAYERS = {
    MODEL_DINOV3_VITS: 12,
    MODEL_DINOV3_VITSP: 12,
    MODEL_DINOV3_VITB: 12,
    MODEL_DINOV3_VITL: 24,
    MODEL_DINOV3_VITHP: 32,
    MODEL_DINOV3_VIT7B: 40,
}

def load_image(path: str) -> Image:
    if path.startswith("http"):
        with urllib.request.urlopen(path) as f:
            return Image.open(f).convert("RGB")
    else:
        return Image.open(path).convert("RGB")
    
def resize_transform(
        mask_image: Image,
        image_size: int,
        patch_size: int,
    ) -> torch.Tensor:
        w, h = mask_image.size
        h_patches = int(image_size / patch_size)
        w_patches = int((w * image_size) / (h * patch_size))
        return TF.to_tensor(TF.resize(mask_image, (h_patches * patch_size, w_patches * patch_size)))        
def visualise_pca(image_loc, only_fg, MODEL_NAME=MODEL_DINOV3_VITHP):

    if os.getenv("DINOV3_LOCATION") is not None:
        DINOV3_LOCATION = os.getenv("DINOV3_LOCATION")
    else:
        DINOV3_LOCATION = DINOV3_GITHUB_LOCATION

    print(f"DINOv3 location set to {DINOV3_LOCATION}")

    model = torch.hub.load(
        repo_or_dir=DINOV3_LOCATION,
        model=MODEL_NAME,
        source="local" if DINOV3_LOCATION != DINOV3_GITHUB_LOCATION else "github",
    )
    model.cuda()
            
    # image resize transform to dimensions divisible by patch size
    image = load_image(image_loc)
    image_resized = resize_transform(image, IMAGE_SIZE, PATCH_SIZE)
    image_resized_norm = TF.normalize(image_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    h_patches, w_patches = [int(d / PATCH_SIZE) for d in image_resized.shape[1:]]

    n_layers = MODEL_TO_NUM_LAYERS[MODEL_NAME]

    with torch.inference_mode():
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            feats = model.get_intermediate_layers(image_resized_norm.unsqueeze(0).cuda(), n=range(n_layers), reshape=True, norm=True)
            x = feats[-1].squeeze().detach().cpu()
            dim = x.shape[0]
            x = x.view(dim, -1).permute(1, 0)

    pca = PCA(n_components=3, whiten=True)
    segment_loc = image_loc.replace("/frames", "/segmentation")

    if only_fg and os.path.exists(segment_loc):
        segmentation = load_image(segment_loc)
        segmentation, _, _ = segmentation.split()  
        segmentation = segmentation.resize((w_patches, h_patches), Image.NEAREST)
        segmentation_mask = np.array(segmentation) > 0
        segmentation_mask = segmentation_mask.flatten()
        pca.fit(x[segmentation_mask])
    else:
        pca.fit(x)    
    projected_image = torch.from_numpy(pca.transform(x.numpy())).view(h_patches, w_patches, 3)

    # multiply by 2.0 and pass through a sigmoid to get vibrant colors 
    projected_image = torch.nn.functional.sigmoid(projected_image.mul(2.0)).permute(2, 0, 1)

    # mask the background using the fg_score_mf
    if only_fg and os.path.exists(segment_loc):
        projected_image *= torch.from_numpy(segmentation_mask).view(1, h_patches, w_patches)
        plt.figure(dpi=300) 
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Input Image")
        plt.axis('off')
        segmentation = load_image(segment_loc)
        segmentation, _, _ = segmentation.split()  
        plt.subplot(1, 3, 2)
        plt.imshow(segmentation)
        plt.title("Segmentation Mask")
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(projected_image.permute(1, 2, 0))
        plt.title("Top 3 PCA on fg")
        plt.axis('off')
        plt.show()
    else:
        plt.figure(dpi=300) 
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Input Image")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(projected_image.permute(1, 2, 0))
        plt.title("Top 3 PCA on full image")
        plt.axis('off')
        plt.show()    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, default="./mmr/train/frames/video_01_000004740.png")
    parser.add_argument("--only_fg", type=int, choices=[0,1], default=0, help="Only use foreground pixels (0/1)")
    args = parser.parse_args()
    assert os.path.exists(args.image_path)  
    visualise_pca(args.image_path, args.only_fg)

if __name__ == "__main__":
    main()