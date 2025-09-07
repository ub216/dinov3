# train_segdinov3.py
import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
import time
import argparse
import random

from metrics import dice_multiclass_torch, iou_multiclass_torch
from dpt import DPT
from dataset import FolderDataset, ResizeAndNormalize
from helpers import save_train_visuals_multiclass, save_eval_visuals_multiclass
from losses import FocalLoss, DicePlusFocalLoss

def train_one_epoch(model, train_loader, optimizer, device, num_classes=1, dice_thr=0.5, vis_dir=None, epoch=0):
    model.train()
    total_loss = 0.0
    dice_all_scores = []
    iou_all_scores = []
    dice_fg_scores = []
    iou_fg_scores = []
    criterion = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()
    first_batch_logged = False
    pbar = tqdm(train_loader, desc=f"[Train e{epoch}]")
    for step, (inputs, targets, _) in enumerate(pbar):
        inputs  = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        if num_classes == 1:
            loss_cls = criterion(logits, targets)
        else:
            loss_cls = criterion(logits, targets.squeeze(1).long())
        loss = loss_cls
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        with torch.no_grad():
            dice_all = dice_multiclass_torch(logits, targets)
            iou_all  = iou_multiclass_torch(logits, targets)
            dice_fg = dice_all[:, 1:].mean().item() # ignore background
            iou_fg = iou_all[:, 1:].mean().item() # ignore background
            dice_all = dice_all.mean().item()
            iou_all  = iou_all.mean().item()
            dice_all_scores.append(dice_all)
            iou_all_scores.append(iou_all)
            dice_fg_scores.append(dice_fg)
            iou_fg_scores.append(iou_fg)
        wandb.log({
            "train/total_loss": loss.item(),
            "train/dice_loss_all": dice_all,
            "train/iou_all": iou_all,
            "train/dice_loss_fg": dice_fg,
            "train/iou_fg": iou_fg,
            "train/lr": optimizer.param_groups[0]["lr"]
        }, step=epoch * len(train_loader) + step)

        pbar.set_postfix(loss=f"{loss.item():.4f}", dice_all=f"{dice_all:.4f}", iou_all=f"{iou_all:.4f}", dice_fg=f"{dice_fg:.4f}", iou_fg=f"{iou_fg:.4f}")
        if (not first_batch_logged) and vis_dir is not None:
            save_train_visuals_multiclass(epoch, inputs, logits, targets, out_dir=vis_dir, max_save=8)
            first_batch_logged = True

    avg_loss = total_loss / max(1, len(train_loader))
    avg_dice_all = float(np.mean(dice_all_scores)) if len(dice_all_scores) > 0 else 0.0
    avg_iou_all  = float(np.mean(iou_all_scores))  if len(iou_all_scores)  > 0 else 0.0
    avg_dice_fg = float(np.mean(dice_fg_scores)) if len(dice_fg_scores) > 0 else 0.0
    avg_iou_fg  = float(np.mean(iou_fg_scores))  if len(iou_fg_scores)  > 0 else 0.0
    print(f"[Train] loss={avg_loss:.4f}  dice={avg_dice_all:.4f}  iou={avg_iou_all:.4f} dice_fg={avg_dice_fg:.4f}  iou_fg={avg_iou_fg:.4f}")
    return avg_loss, avg_dice_all, avg_iou_all, avg_dice_fg, avg_iou_fg

@torch.no_grad()
def evaluate(model, val_loader, device, num_classes=1, dice_thr=0.5, vis_dir=None):
    model.eval()
    total_loss = 0.0
    dice_all_scores = []
    iou_all_scores  = []
    dice_fg_scores = []
    iou_fg_scores  = []
    criterion = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()
    idx_global = 0
    pbar = tqdm(val_loader, desc="[Eval]")
    for (inputs, targets, _) in pbar:
        inputs  = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        if num_classes == 1:
            loss = criterion(logits, targets)
        else:
            loss = criterion(logits, targets.squeeze(1).long())
        total_loss += loss.item()
        dice_all = dice_multiclass_torch(logits, targets)
        iou_all  = iou_multiclass_torch(logits, targets)
        dice_fg = dice_all[:, 1:].mean().item() # ignore background
        iou_fg = iou_all[:, 1:].mean().item() # ignore background
        dice_all = dice_all.mean().item()
        iou_all  = iou_all.mean().item()
        dice_all_scores.append(dice_all)
        iou_all_scores.append(iou_all)
        dice_fg_scores.append(dice_fg)
        iou_fg_scores.append(iou_fg)
        pbar.set_postfix(loss=f"{loss.item():.4f}", dice_all=f"{dice_all:.4f}", iou_all=f"{iou_all:.4f}", dice_fg=f"{dice_fg:.4f}", iou_fg=f"{iou_fg:.4f}")
        if vis_dir is not None:
            os.makedirs(vis_dir, exist_ok=True)
            B = inputs.size(0)
            for b in range(B):
                save_eval_visuals_multiclass(
                    idx_global,
                    inputs[b],
                    logits[b],
                    targets[b],
                    out_dir=vis_dir,
                    fname_prefix="val"
                )
                idx_global += 1
    avg_loss = total_loss / max(1, len(val_loader))
    avg_dice_all = float(np.mean(dice_all_scores)) if len(dice_all_scores) > 0 else 0.0
    avg_iou_all  = float(np.mean(iou_all_scores))  if len(iou_all_scores)  > 0 else 0.0
    avg_dice_fg = float(np.mean(dice_fg_scores)) if len(dice_fg_scores) > 0 else 0.0
    avg_iou_fg  = float(np.mean(iou_fg_scores))  if len(iou_fg_scores)  > 0 else 0.0
    print(f"[Eval] loss={avg_loss:.4f}  dice={avg_dice_all:.4f}  iou={avg_iou_all:.4f} dice_fg={avg_dice_fg:.4f}  iou_fg={avg_iou_fg:.4f}")
    return avg_loss, avg_dice_all, avg_iou_all, avg_dice_fg, avg_iou_fg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./mmr")
    parser.add_argument("--img_ext", type=str, default=".png")
    parser.add_argument("--mask_ext", type=str, default=".png")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input_h", type=int, default=256)
    parser.add_argument("--input_w", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--repo_dir", type=str, default="./dinov3")
    parser.add_argument("--dino_ckpt", type=str, required=True,help="Path to the pretrained DINO checkpoint (.pth). "
                         "Use ViT-B/16 checkpoint for --dino_size b, "
                         "or ViT-S/16 checkpoint for --dino_size s.")
    parser.add_argument("--dino_size", type=str, default="b", choices=["b", "s"],
                        help="DINO backbone size: b=ViT-B/16, s=ViT-S/16")
    parser.add_argument("--last_layer_idx", type=int, default=-1)
    parser.add_argument("--vis_max_save", type=int, default=8)
    parser.add_argument("--img_dir_name", type=str, default="frames")
    parser.add_argument("--label_dir_name", type=str, default="segmentation")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ctime = time.strftime("%Y%m%d-%H%M%S")
    wandb.init(project="mmr", name=f"segdino_{args.dino_size}_{args.input_h}_{ctime}")

    save_root = f"./runs/segdino_{args.dino_size}_{args.input_h}_{ctime}"
    os.makedirs(save_root, exist_ok=True)
    train_vis_dir = os.path.join(save_root, "train_vis")
    val_vis_dir   = os.path.join(save_root, "val_vis")
    ckpt_dir      = os.path.join(save_root, "ckpts")
    os.makedirs(train_vis_dir, exist_ok=True)
    os.makedirs(val_vis_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    if args.dino_size == "b":
        backbone = torch.hub.load(args.repo_dir, 'dinov3_vitb16', source='local', weights=args.dino_ckpt)
    else:
        backbone = torch.hub.load(args.repo_dir, 'dinov3_vits16', source='local', weights=args.dino_ckpt)


    model = DPT(nclass=args.num_classes, backbone=backbone)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )

    root = args.data_dir

    train_transform = ResizeAndNormalize(size=(args.input_h, args.input_w), num_classes=args.num_classes)
    val_transform   = ResizeAndNormalize(size=(args.input_h, args.input_w), num_classes=args.num_classes)

    train_dataset = FolderDataset(
        root=root,
        split="train",
        img_dir_name=args.img_dir_name,
        label_dir_name=args.label_dir_name,
        mask_ext=args.mask_ext if hasattr(args, "mask_ext") else None,
        transform=train_transform,
    )
    val_dataset = FolderDataset(
        root=root,
        split="test",
        img_dir_name=args.img_dir_name,
        label_dir_name=args.label_dir_name,
        mask_ext=args.mask_ext if hasattr(args, "mask_ext") else None,
        transform=val_transform,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )

    best_val_dice = -1.0
    best_val_dice_epoch = -1
    best_val_iou  = -1.0
    best_val_iou_epoch  = -1

    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice_all, train_iou_all, train_dice_fg, train_iou_fg = train_one_epoch(
            model, train_loader, optimizer, device,
            num_classes=args.num_classes, dice_thr=0.5,
            vis_dir=train_vis_dir, epoch=epoch
        )
        wandb.log({
            "train/total_loss_epoch": train_loss,
            "train/dice_loss_all_epoch": train_dice_all,
            "train/iou_all_epoch": train_iou_all,
            "train/dice_loss_fg_epoch": train_dice_fg,
            "train/iou_fg_epoch": train_iou_fg,
            "train/lr_epoch": optimizer.param_groups[0]["lr"],
            }, step=(epoch+1) * len(train_loader))

        val_loss, val_dice_all, val_iou_all, val_dice_fg, val_iou_fg = evaluate(
            model, val_loader, device,
            num_classes=args.num_classes, dice_thr=0.5,
            vis_dir=val_vis_dir)
        wandb.log({
            "eval/total_loss_epoch": val_loss,
            "eval/dice_loss_all_epoch": val_dice_all,
            "eval/iou_all_epoch": val_iou_all,
            "eval/dice_loss_fg_epoch": val_dice_fg,
            "eval/iou_fg_epoch": val_iou_fg,
            }, step=(epoch+1) * len(train_loader))

        latest_path = os.path.join(ckpt_dir, "latest.pth")
        torch.save(
            {"epoch": epoch, "state_dict": model.state_dict(),
             "optimizer": optimizer.state_dict()},
            latest_path
        )

        if val_dice_fg > best_val_dice:
            best_val_dice = val_dice_fg
            best_val_dice_epoch = epoch
            best_path = os.path.join(ckpt_dir, f"best_ep{epoch:03d}_dice{val_dice_fg:.4f}_{val_iou_fg:.4f}.pth")
            torch.save(model.state_dict(), best_path)
            print(f"[Save] New best ckpt: {best_path}")

        if val_iou_fg > best_val_iou:
            best_val_iou = val_iou_fg
            best_val_iou_epoch = epoch

    print("=" * 60)
    print(f"[Summary] Best Val Dice = {best_val_dice:.4f} @ epoch {best_val_dice_epoch}")
    print(f"[Summary] Best Val IoU  = {best_val_iou:.4f}  @ epoch {best_val_iou_epoch}")
    print("=" * 60)

if __name__ == "__main__":
    main()
