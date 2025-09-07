import os
import csv
import cv2
import math
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from metrics import dice_multiclass_torch, iou_multiclass_torch
from helpers import save_eval_visuals_multiclass, plot_per_class_metrics
from dataset import FolderDataset, ResizeAndNormalize
CLS_NAME_MMR = ["BG", "T.clasper", "T.wrist", "T.shaft", "S.needle", "Thred", "S.tool", "N.holder", "Clamp", "Catheter"]
# -------------------- Main Test --------------------
@torch.no_grad()
def run_test(model, loader, device, vis_dir=None, csv_path=None, num_classes=None):
    model.eval()
    os.makedirs(vis_dir, exist_ok=True) if vis_dir else None

    rows = []
    idx_global = 0

    # Running accumulators
    total_dice = np.zeros(num_classes, dtype=np.float64)
    total_iou  = np.zeros(num_classes, dtype=np.float64)
    n_samples  = 0

    pbar = tqdm(loader, desc="[Test]")
    for batch in pbar:
        assert len(batch) == 3
        inputs, targets, meta = batch
        case_ids = list(meta["id"])
        input_path = list(meta["img_path"])

        inputs  = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.no_grad():
            logits = model(inputs)  

            dice_all = dice_multiclass_torch(logits, targets)  # (B, C)
            iou_all  = iou_multiclass_torch(logits, targets)   # (B, C)

        dsc = dice_all.cpu().numpy()
        iou = iou_all.cpu().numpy()

        # Update accumulators
        total_dice += dsc.sum(axis=0)
        total_iou  += iou.sum(axis=0)
        n_samples  += dsc.shape[0]

        B = inputs.size(0)
        for b in range(B):
            rows.append({
                "id": case_ids[b],
                "dice": dsc[b].mean(),
                "iou": iou[b].mean()
            })
            if vis_dir is not None:
                save_eval_visuals_multiclass(
                    idx_global, inputs[b], logits[b], targets[b], vis_dir, input_path=input_path[b], fname_prefix="test"
                )
            idx_global += 1

        pbar.set_postfix(
            dice=(total_dice / n_samples).mean(),
            iou=(total_iou / n_samples).mean()
        )

    # Compute final mean per-class and overall
    mean_dice_per_class = total_dice / n_samples
    mean_iou_per_class  = total_iou / n_samples
    mean_dice = mean_dice_per_class.mean()
    mean_iou  = mean_iou_per_class.mean()

    if csv_path is not None:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            fieldnames = ["id", "dice", "iou"] + [f"dice_class{c}" for c in range(num_classes)]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
            mean_row = {
                "id": "MEAN", "dice": mean_dice, "iou": mean_iou,
                **{f"dice_class{c}": mean_dice_per_class[c] for c in range(num_classes)}
            }
            writer.writerow(mean_row)

    plot_per_class_metrics(mean_dice_per_class, metric_name="Dice", class_names=CLS_NAME_MMR,
                           save_path=csv_path.replace(".csv", "_dice_per_class.png") if csv_path else None)
    plot_per_class_metrics(mean_iou_per_class, metric_name="IoU", class_names=CLS_NAME_MMR,
                           save_path=csv_path.replace(".csv", "_iou_per_class.png") if csv_path else None)


    print("=" * 60)
    print(f"[Test Summary] Dice={mean_dice:.4f}  IoU={mean_iou:.4f}")
    print("=" * 60)
    
    return mean_dice, mean_iou

def load_ckpt_flex(model, ckpt_path, map_location="cpu"):
    obj = torch.load(ckpt_path, map_location=map_location)
    if isinstance(obj, dict) and "state_dict" in obj:
        state = obj["state_dict"]
    else:
        state = obj
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[Warn] Missing keys:", missing)
    if unexpected:
        print("[Warn] Unexpected keys:", unexpected)

def main():
    import argparse
    import os
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./mmr")
    parser.add_argument("--mask_ext", type=str, default=".png")
    parser.add_argument("--input_h", type=int, default=1088)
    parser.add_argument("--input_w", type=int, default=1920)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_classes", type=int, default=10)

    # Segmentation model checkpoint (DPT + decoder)
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to the trained segmentation model checkpoint (.pth).")

    parser.add_argument("--save_root", type=str, default="./runs")
    parser.add_argument("--img_dir_name", type=str, default="frames")
    parser.add_argument("--label_dir_name", type=str, default="segmentation")

    # DINO backbone configuration
    parser.add_argument("--dino_size", type=str, default="s", choices=["b", "s"],
                        help="DINO backbone size: b=ViT-B/16, s=ViT-S/16")
    parser.add_argument("--dino_ckpt", type=str, required=True,
                        help="Path to the pretrained DINO checkpoint (.pth). "
                             "Use ViT-B/16 checkpoint for --dino_size b, or ViT-S/16 for --dino_size s.")
    parser.add_argument("--repo_dir", type=str, default="./dinov3",
                        help="Local path to the DINOv3 torch.hub repo (contains hubconf.py).")

    args = parser.parse_args()

    # Output directories
    save_root = os.path.join(args.save_root, f"dinov3_{args.ckpt.split('/')[-1].replace('.pth','')}_{args.input_h}x{args.input_w}")
    vis_dir   = os.path.join(save_root, "test_vis")
    csv_path  = os.path.join(save_root, "test_metrics.csv")
    os.makedirs(save_root, exist_ok=True)

    # Load DINO backbone depending on size
    if args.dino_size == "b":
        backbone = torch.hub.load(args.repo_dir, 'dinov3_vitb16', source='local', weights=args.dino_ckpt)
    else:
        backbone = torch.hub.load(args.repo_dir, 'dinov3_vits16', source='local', weights=args.dino_ckpt)

    from dpt import DPT
    model = DPT(nclass=args.num_classes, backbone=backbone)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Load segmentation checkpoint
    print(f"[Load segmentation ckpt] {args.ckpt}")
    load_ckpt_flex(model, args.ckpt, map_location=device)


    test_transform = ResizeAndNormalize(size=(args.input_h, args.input_w), num_classes=args.num_classes)
    test_dataset = FolderDataset(
        root=args.data_dir,
        split="test",
        img_dir_name=args.img_dir_name,
        label_dir_name=args.label_dir_name,
        mask_ext=args.mask_ext if hasattr(args, "mask_ext") else None,
        transform=test_transform,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    # Run evaluation
    run_test(model, test_loader, device, vis_dir=vis_dir, csv_path=csv_path, num_classes=args.num_classes)


if __name__ == "__main__":
    main()
