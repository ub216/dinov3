import os
from typing import List, Tuple, Optional
import cv2
import torch
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

class ResizeAndNormalize:
    def __init__(self, size=(256, 256), mean=IMAGENET_MEAN, std=IMAGENET_STD, thr=0.5, num_classes=1):
        self.size = size  # (H, W)
        self.mean = mean
        self.std = std
        self.thr = thr
        self.num_classes = num_classes

    def __call__(self, img_bgr: np.ndarray, mask_hwc: np.ndarray):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_resized = TF.resize(
            img_pil, self.size,
            interpolation=InterpolationMode.BICUBIC, antialias=True
        )
        img_t = TF.to_tensor(img_resized)                   # Convert image to tensor
        img_t = TF.normalize(img_t, self.mean, self.std)    # Normalize with ImageNet mean and std
        H, W = self.size
        mask_ts = []
        if 0:#self.num_classes == mask_hwc.shape[2]:
            for c in range(mask_hwc.shape[2]):
                m = Image.fromarray(mask_hwc[..., c])
                m = TF.resize(m, (H, W), interpolation=InterpolationMode.NEAREST)
                mt = TF.to_tensor(m)[0]        # Take the single channel
                mask_ts.append(mt)
            mask_t = torch.stack(mask_ts, dim=0).float()   # Stack into (C,H,W)
            mask_t = (mask_t > self.thr).float()           # Binarize mask by threshold
        else:
            m = torch.from_numpy(mask_hwc).permute(2, 0, 1)
            mask_t = TF.resize(m, (H, W), interpolation=InterpolationMode.NEAREST).float()
        return img_t, mask_t


SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
def _list_files(dir_path: str, exts: Tuple[str, ...]) -> List[str]:
    out = []
    for root, _, files in os.walk(dir_path):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                out.append(os.path.join(root, f))
    out.sort()
    return out

def _swap_dir_keep_name(path: str, src_dir_name: str, dst_dir_name: str, dst_ext: Optional[str]) -> str:
    parts = path.replace("\\", "/").split("/")
    try:
        i = parts.index(src_dir_name)
    except ValueError:
        raise RuntimeError(f"Directory name '{src_dir_name}' not found in path: {path}")
    parts[i] = dst_dir_name
    dst_path = "/".join(parts)
    if dst_ext is not None:
        base, _ = os.path.splitext(dst_path)
        dst_path = base + dst_ext
    return dst_path

class FolderDataset(data.Dataset):
    """
    Directory structure:
      root/
        train/
          img/
          label/
        test/
          img/
          label/
    """
    def __init__(
        self,
        root: str,
        split: str = "train",
        img_dir_name: str = "img",
        label_dir_name: str = "label",
        img_exts: Tuple[str, ...] = SUPPORTED_EXTS,
        mask_ext: Optional[str] = None,  
        transform: Optional[ResizeAndNormalize] = None,
        strict_pair: bool = True,       
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.img_dir_name = img_dir_name
        self.label_dir_name = label_dir_name
        self.img_dir = os.path.join(root, split, img_dir_name)
        self.label_dir = os.path.join(root, split, label_dir_name)
        self.mask_ext = mask_ext
        self.transform = transform
        self.strict_pair = strict_pair

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        if not os.path.isdir(self.label_dir):
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")

        self.img_paths = _list_files(self.img_dir, img_exts)
        if len(self.img_paths) == 0:
            raise RuntimeError(f"No images found in {self.img_dir} (supported extensions: {img_exts})")

        self.pairs: List[Tuple[str, str]] = []
        for ip in self.img_paths:
            mp = _swap_dir_keep_name(ip, self.img_dir_name, self.label_dir_name, self.mask_ext)
            if not os.path.isfile(mp) and self.mask_ext is None:
                base, _ = os.path.splitext(mp)
                found = False
                for ext in SUPPORTED_EXTS:
                    cand = base + ext
                    if os.path.isfile(cand):
                        mp = cand
                        found = True
                        break
                if not found and self.strict_pair:
                    raise FileNotFoundError(f"Label file not found: {base}.(any extension with same name)")
                elif not found:
                    continue
            elif not os.path.isfile(mp) and self.strict_pair:
                raise FileNotFoundError(f"Label file not found: {mp}")
            elif not os.path.isfile(mp):
                continue

            self.pairs.append((ip, mp))

        if len(self.pairs) == 0:
            raise RuntimeError(f"No valid (img, label) pairs in {self.split} set!")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # (H,W,3) BGR uint8
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # (H,W) uint8
        if mask is None:
            raise RuntimeError(f"Failed to read label: {mask_path}")
        # (H,W) -> (H,W,1)
        mask_hwc = mask[..., None]

        if self.transform is not None:
            img_t, mask_t = self.transform(img, mask_hwc)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            m = torch.from_numpy(mask_hwc).permute(2, 0, 1).float() / 255.0
            mask_t = (m > 0.5).float()

        if mask_t.max().item() < 1.0:
            mask_t = (mask_t > 0).float()

        meta = {
            "img_path": img_path,
            "mask_path": mask_path,
            "id": os.path.splitext(os.path.basename(img_path))[0],
        }
        return img_t, mask_t, meta
