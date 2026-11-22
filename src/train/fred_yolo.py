"""Compare YOLO11n and the custom CNN on the same offline FRED frames."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data.fred_detection import load_detection_split
from src.train.event_detector import MAX_BOXES, average_precision


def export_split(root, split, expected, cache):
    image_dir = root / 'images' / split
    label_dir = root / 'labels' / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    frame_index = 0
    for sequence in expected['sequences']:
        images, labels = load_detection_split([sequence], cache)
        for image, boxes in zip(images, labels, strict=True):
            stem = f'{frame_index:05d}'
            height, width = image.shape[1:]
            # Store both polarity counts losslessly in a conventional RGB PNG.
            rgb = np.zeros((height, width, 3), dtype=np.uint8)
            rgb[:, :, :2] = image.transpose(1, 2, 0)
            Image.fromarray(rgb).save(image_dir / f'{stem}.png', compress_level=1)
            centres = (boxes[:, :2] + boxes[:, 2:]) / 2
            sizes = boxes[:, 2:] - boxes[:, :2]
            normalized = np.concatenate((centres, sizes), axis=1) / (width, height, width, height)
            text = ''.join('0 ' + ' '.join(f'{coordinate:.9f}' for coordinate in row) + '\n' for row in normalized)
            (label_dir / f'{stem}.txt').write_text(text)
            frame_index += 1
        print(f'{split} {sequence}: exported {len(labels):,} frames', flush=True)
    if frame_index != expected['frames']:
        raise ValueError(f'{split}: {frame_index} exported frames, expected {expected["frames"]}')


def prepare_dataset(root, baseline, cache):
    manifest = root / 'data.yaml'
    config = {'path': str(root.resolve()), 'train': 'images/train', 'val': 'images/val', 'names': ['drone']}
    source = {split: baseline[split] for split in ('training', 'validation')}
    source_file = root / 'source.json'
    if manifest.exists() and json.loads(source_file.read_text()) != source:
        raise ValueError('Existing export comes from a different CNN benchmark')
    if not manifest.exists():
        export_split(root, 'train', source['training'], cache)
        export_split(root, 'val', source['validation'], cache)
        source_file.write_text(json.dumps(source, indent=2) + '\n')
    for split, key in (('train', 'training'), ('val', 'validation')):
        image_names = {path.stem for path in (root / 'images' / split).glob('*.png')}
        label_names = {path.stem for path in (root / 'labels' / split).glob('*.txt')}
        expected = {f'{i:05d}' for i in range(source[key]['frames'])}
        if image_names != expected or label_names != expected:
            raise ValueError(f'{split}: exported frame or label files do not match the CNN benchmark')
    # JSON is valid YAML and avoids a dependency during dataset preparation.
    manifest.write_text(json.dumps(config, indent=2) + '\n')
    return manifest


def score_checkpoint(model, validation_dir, labels, device):
    paths = sorted(validation_dir.glob('*.png'))
    results = model.predict(source=str(validation_dir), stream=True, imgsz=640, rect=False,
                            batch=16, device=device, conf=0.0, max_det=MAX_BOXES,
                            verbose=False)
    predictions = []
    for path, result in zip(paths, results, strict=True):
        if path != Path(result.path).resolve() or result.orig_shape != (360, 640):
            raise ValueError(f'Unexpected validation frame {result.path} for {path}')
        scores = result.boxes.conf.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        order = np.argsort(-scores, kind='stable')
        predictions.append((scores[order], boxes[order]))
    return {name: average_precision(predictions, labels, overlap)
            for name, overlap in (('ap50', 0.5), ('ap75', 0.75))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True, help='custom CNN results.json')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    output = args.out.resolve()
    dataset = output / 'dataset'
    cache = args.baseline.resolve().parent.parent / 'fred-detection-cache'
    manifest = prepare_dataset(dataset, baseline, cache)
    if args.prepare_only:
        print(f'Prepared {manifest}', flush=True)
        return

    from ultralytics import YOLO, __version__ as ultralytics_version

    if (output / 'results.json').exists():
        raise FileExistsError(f'{output} already contains results')
    _, labels = load_detection_split(baseline['validation']['sequences'], cache)
    scores = []
    for run in baseline['runs']:
        seed = run['seed']
        name = f'yolo_seed_{seed}'
        if (output / name).exists():
            raise FileExistsError(f'{output / name} already exists')
        torch.manual_seed(seed)
        model = YOLO('yolo11n.yaml')
        model.train(data=str(manifest), epochs=baseline['epochs'], imgsz=640,
                    batch=baseline['batch_size'], workers=0, device=args.device,
                    seed=seed, pretrained=False, optimizer='AdamW', lr0=baseline['learning_rate'],
                    project=str(output), name=name, save=True, val=False, plots=False, rect=False,
                    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, translate=0.0, scale=0.0,
                    fliplr=0.0, mosaic=0.0)
        checkpoint = Path(model.trainer.last)
        trained = YOLO(str(checkpoint))
        result = score_checkpoint(trained, dataset / 'images' / 'val', labels, args.device)
        parameters = sum(parameter.numel() for parameter in trained.model.parameters())
        scores.append({'seed': seed, 'checkpoint': str(checkpoint.relative_to(output)), **result})
        print(f'YOLO seed {seed}: {result}', flush=True)
    summary = {}
    for metric in ('ap50', 'ap75'):
        values = [run[metric] for run in scores]
        mean = float(np.mean(values))
        cnn = baseline['validation_summary'][metric]
        summary[metric] = {'yolo_mean': mean, 'yolo_std': float(np.std(values, ddof=1)),
                           'cnn_mean': cnn['mean'], 'cnn_std': cnn['std'], 'mean_difference': mean - cnn['mean']}
    report = {'baseline': str(args.baseline.resolve()), 'model': 'YOLO11n', 'pretrained': False,
              'ultralytics_version': ultralytics_version, 'device': args.device,
              'epochs': baseline['epochs'], 'parameters': parameters, 'cnn_parameters': baseline['parameters'],
              'input': 'Two polarity-count channels and one zero channel, zero-loss PNG, letterboxed to 640 square',
              'training': baseline['training'], 'validation': baseline['validation'],
              'runs': scores, 'comparison': summary}
    (output / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
