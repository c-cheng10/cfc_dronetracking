"""Train an offline full-frame FRED detector.

Run from the repository root: python -m src.train.event_detector --out runs/fred-detector
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src.data.cnn_windows import load_splits
from src.data.fred_detection import load_detection_split
from src.models.event_detector import STRIDE, EventDetector
from src.train.cnn import SEEDS


BATCH_SIZE = 16
LEARNING_RATE = 0.001
MAX_BOXES = 30


def encode_boxes(boxes_by_frame, height, width, device):
    """Assign each labelled box to the grid cell containing its centre."""
    boxes = np.concatenate(boxes_by_frame)
    frame = np.repeat(np.arange(len(boxes_by_frame)), [len(boxes) for boxes in boxes_by_frame])
    centre = (boxes[:, :2] + boxes[:, 2:]) / (2 * STRIDE)
    cells = centre.astype(np.int64)
    sizes = (boxes[:, 2:] - boxes[:, :2]) / STRIDE
    values = np.concatenate((centre - cells, np.log(sizes)), axis=1).astype(np.float32)
    # The head predicts one box per cell. Use the larger box for a collision,
    # and keep every annotated box in evaluation.
    order = np.argsort(-sizes.prod(axis=1), kind="stable")
    index = frame * height * width + cells[:, 1] * width + cells[:, 0]
    _, first = np.unique(index[order], return_index=True)
    keep = order[first]
    heatmap = np.zeros((len(boxes_by_frame), height, width), dtype=bool)
    geometry = np.zeros((len(boxes_by_frame), 4, height, width), dtype=np.float32)
    y, x = cells[keep, 1], cells[keep, 0]
    heatmap[frame[keep], y, x] = True
    geometry.transpose(0, 2, 3, 1)[frame[keep], y, x] = values[keep]
    return torch.from_numpy(heatmap).to(device), torch.from_numpy(geometry).to(device), len(boxes) - len(keep)


def training_loss(scores, geometry, centres, boxes):
    normalizer = centres.sum().clamp_min(1)
    # Focal loss downweights grid cells the classifier already scores correctly.
    signed = torch.where(centres, scores, -scores)
    heatmap_loss = (F.softplus(-signed) * torch.sigmoid(-signed).square()).sum()
    box_loss = (F.smooth_l1_loss(geometry, boxes, reduction="none") * centres[:, None]).sum()
    return (heatmap_loss + box_loss) / normalizer


def train(frames, centres, boxes, model, epochs, seed):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(frames))
        total_loss = torch.zeros((), device=centres.device)
        for indices in order.split(BATCH_SIZE):
            batch = frames[indices].to(device=centres.device, dtype=torch.float32).log1p_()
            device_indices = indices.to(centres.device)
            optimizer.zero_grad(set_to_none=True)
            scores, geometry = model(batch)
            loss = training_loss(scores, geometry, centres[device_indices], boxes[device_indices])
            loss.backward()
            optimizer.step()
            total_loss += loss.detach() * len(indices)
        mean_loss = total_loss.item() / len(frames)
        if not np.isfinite(mean_loss):
            raise FloatingPointError(f"Non-finite loss for seed {seed}, epoch {epoch}")
        print(f"seed {seed} epoch {epoch:02}/{epochs}: loss {mean_loss:.4f}", flush=True)


@torch.inference_mode()
def predict(model, frames, device):
    """Return the highest scoring local maxima and their boxes for each frame."""
    model.eval()
    predictions = []
    for batch in frames.split(BATCH_SIZE):
        height, width = batch.shape[-2:]
        scores, geometry = model(batch.to(device=device, dtype=torch.float32).log1p_())
        peaks = scores == F.max_pool2d(scores[:, None], kernel_size=3, stride=1, padding=1).squeeze(1)
        confidence, indices = scores.masked_fill(~peaks, -torch.inf).flatten(1).topk(MAX_BOXES)
        values = geometry.flatten(2).gather(2, indices[:, None].expand(-1, 4, -1)).transpose(1, 2)
        cells = torch.stack((indices % scores.shape[-1], indices // scores.shape[-1]), dim=-1)
        centres = (cells + values[..., :2].clamp(0, 1)) * STRIDE
        half_sizes = values[..., 2:].clamp(-4, 6).exp() * (STRIDE / 2)
        scale = centres.new_tensor((width, height))
        corners = torch.cat(((centres - half_sizes).clamp_min(0).minimum(scale),
                             (centres + half_sizes).clamp_min(0).minimum(scale)), dim=-1)
        predictions.extend(zip(confidence.cpu().numpy(), corners.cpu().numpy(), strict=True))
    return predictions


def box_iou(predicted, labelled):
    upper_left = np.maximum(predicted[:, None, :2], labelled[None, :, :2])
    lower_right = np.minimum(predicted[:, None, 2:], labelled[None, :, 2:])
    intersection = np.maximum(lower_right - upper_left, 0).prod(axis=2)
    prediction_area = (predicted[:, 2:] - predicted[:, :2]).prod(axis=1)
    label_area = (labelled[:, 2:] - labelled[:, :2]).prod(axis=1)
    return intersection / (prediction_area[:, None] + label_area[None, :] - intersection)


def average_precision(predictions, labels, overlap):
    """Match boxes one to one, then integrate the precision-recall curve."""
    scores, matches = [], []
    for (confidence, boxes), targets in zip(predictions, labels, strict=True):
        iou = box_iou(boxes, targets)
        matched = np.zeros(len(boxes), dtype=bool)
        if len(targets):
            for i in np.flatnonzero(np.isfinite(confidence)):
                best = np.argmax(iou[i])
                if iou[i, best] >= overlap:
                    matched[i] = True
                    iou[:, best] = -1
        scores.extend(confidence[np.isfinite(confidence)])
        matches.extend(matched[np.isfinite(confidence)])
    order = np.argsort(-np.asarray(scores), kind="stable")
    true_positives = np.cumsum(np.asarray(matches, dtype=bool)[order])
    precision = true_positives / np.arange(1, len(order) + 1)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    recall_step = np.diff(true_positives, prepend=0) / sum(map(len, labels))
    return float(np.sum(precision * recall_step))


def describe_split(sequences, boxes):
    return {"sequences": sequences, "frames": len(boxes), "boxes": sum(map(len, boxes)),
            "empty_frames": sum(len(frame_boxes) == 0 for frame_boxes in boxes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=False)

    train_sequences, val_sequences = load_splits()
    cache = args.out.parent / "fred-detection-cache"
    train_images, train_labels = load_detection_split(train_sequences, cache)
    val_images, val_labels = load_detection_split(val_sequences, cache)
    train_info = describe_split(train_sequences, train_labels)
    val_info = describe_split(val_sequences, val_labels)
    if not train_info["boxes"] or not val_info["boxes"]:
        raise ValueError("Both splits need labelled drone boxes")
    height, width = train_images.shape[-2:]
    if val_images.shape[-2:] != (height, width) or height % STRIDE or width % STRIDE:
        raise ValueError("Frame sizes must match and be divisible by the model stride")
    print(f"{device}: {train_info['frames']:,} train frames, {val_info['frames']:,} validation frames", flush=True)
    centres, boxes, collisions = encode_boxes(train_labels, height // STRIDE, width // STRIDE, device)
    train_frames, val_frames = torch.from_numpy(train_images), torch.from_numpy(val_images)

    scores = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = EventDetector().to(device)
        train(train_frames, centres, boxes, model, args.epochs, seed)
        predictions = predict(model, val_frames, device)
        ap50 = average_precision(predictions, val_labels, 0.50)
        ap75 = average_precision(predictions, val_labels, 0.75)
        scores.append({"seed": seed, "ap50": ap50, "ap75": ap75})
        torch.save({name: value.cpu() for name, value in model.state_dict().items()}, args.out / f"seed_{seed}.pt")
        print(f"seed {seed}: validation AP50 {ap50:.4f}, AP75 {ap75:.4f}", flush=True)

    summary = {}
    for metric in ("ap50", "ap75"):
        values = [run[metric] for run in scores]
        summary[metric] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))}
    report = {
        "device": device,
        "torch_version": torch.__version__,
        "epochs": args.epochs,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "frame_height": height,
        "frame_width": width,
        "training_grid_collisions": collisions,
        "training": train_info,
        "validation": val_info,
        "runs": scores,
        "validation_summary": summary,
    }
    (args.out / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
