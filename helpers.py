import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os

def get_color_palette(num_classes):
    """
    Generate a distinct color palette for each class using HSV -> BGR conversion.
    Returns colors as (num_classes, 3) uint8 array in BGR order.
    """
    hsv_colors = np.zeros((num_classes, 1, 3), dtype=np.uint8)
    for i in range(num_classes):
        h = int((i * 180 / num_classes))  # OpenCV Hue [0,179]
        s = 255
        v = 255
        hsv_colors[i, 0] = [h, s, v]
    
    # Convert HSV to BGR
    bgr_colors = cv2.cvtColor(hsv_colors, cv2.COLOR_HSV2BGR)  # shape: (num_classes,1,3)
    bgr_colors = bgr_colors[:,0,:]
    bgr_colors[0, :] = [0, 0, 0]  # Ensure class 0 is black
    return bgr_colors  # shape: (num_classes, 3)

def tensor_to_rgb(img_t: torch.Tensor) -> np.ndarray:
    img = img_t.detach().cpu().float().clamp(0, 1).numpy()
    img = (img * 255.0).round().astype(np.uint8)
    if img.ndim == 3:  # C,H,W
        img = np.transpose(img, (1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def logits_to_color_mask(logits: torch.Tensor, colors) -> np.ndarray:
    """
    Convert multi-class logits to a color mask
    logits: (C,H,W) or (B,C,H,W)
    returns: (H,W,3) or (B,H,W,3)
    """
    if logits.ndim == 4:  # batch
        masks = [logits_to_color_mask(l, colors) for l in logits]
        return np.stack(masks, axis=0)
    
    # single sample
    if logits.ndim == 3:
        pred_class = torch.argmax(logits, dim=0).cpu().numpy()  # (H,W)
        mask_color = colors[pred_class]
        return mask_color
    else:
        raise ValueError(f"Unexpected logits shape: {logits.shape}")

def targets_to_color_mask(target: torch.Tensor, colors) -> np.ndarray:
    """
    Convert integer mask (C,H,W or H,W) to color mask
    """
    assert target.ndim == 2
    target = target.detach().cpu()
    target_np = target.numpy().astype(np.int8)
    return colors[target_np]

# Example: use in training visualization
def save_train_visuals_multiclass(epoch, inputs, logits, targets, out_dir, max_save=8):
    os.makedirs(out_dir, exist_ok=True)
    b = min(inputs.size(0), max_save)
    colors = get_color_palette(logits.shape[1])
    for i in range(b):
        img_bgr = tensor_to_rgb(inputs[i])
        pred_color = logits_to_color_mask(logits[i], colors)
        gt_color   = targets_to_color_mask(targets[i][0], colors)
        base = os.path.join(out_dir, f"train_ep{epoch:03d}_idx{i:02d}")
        cv2.imwrite(base + "_img.png",  img_bgr)
        cv2.imwrite(base + "_pred.png", pred_color)
        cv2.imwrite(base + "_gt.png",   gt_color)

@torch.no_grad()
def save_eval_visuals_multiclass(idx, inputs, logits, targets, out_dir, input_path=None, fname_prefix="val"):
    os.makedirs(out_dir, exist_ok=True)
    img_bgr = cv2.imread(input_path) if input_path is not None else tensor_to_rgb(inputs)
    colors = get_color_palette(logits.shape[0])
    pred_color = logits_to_color_mask(logits, colors)
    gt_color   = targets_to_color_mask(targets[0], colors)
    base = os.path.join(out_dir, f"{fname_prefix}_{idx:05d}")
    cv2.imwrite(base + "_img.png",  img_bgr)
    cv2.imwrite(base + "_pred.png", pred_color)
    cv2.imwrite(base + "_gt.png",   gt_color)


def plot_per_class_metrics(metrics, class_names=None, metric_name="IoU", save_path=None):
    """
    metrics: list or np.array of per-class values (len = num_classes)
    class_names: list of class labels (len = num_classes) or None
    metric_name: "IoU" or "Dice"
    """
    num_classes = len(metrics)
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(class_names, metrics, color=plt.cm.viridis(np.linspace(0, 1, num_classes)))
    plt.axhline(np.mean(metrics), color="red", linestyle="--", label=f"Mean {metric_name}: {np.mean(metrics):.3f}")
    
    # Annotate values on top of bars
    for bar, val in zip(bars, metrics):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.2f}", 
                 ha="center", va="bottom", fontsize=9)

    plt.ylabel(metric_name)
    plt.title(f"Per-class {metric_name}")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved per-class {metric_name} plot to: {save_path}")