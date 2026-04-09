
import argparse
import csv
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from skimage.feature import hog
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, average_precision_score, top_k_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import LinearSVC

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms

try:
    import timm
except ImportError:
    timm = None

try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    FlopCountAnalysis = None


DATASET_NAME = "Caltech-256"
DEFAULT_OUTPUT_CSV = "results/caltech256_results.csv"
SUPPORTED_MODELS = ("resnet50", "vit_b_16", "sift", "hog")


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


@dataclass
class RunResult:
    dataset_name: str
    model_name: str
    epochs: int
    train_count: int
    val_count: int
    test_count: int
    train_time_sec: float
    inference_time_sec: float
    val_top1: Optional[float]
    val_top5: Optional[float]
    val_macro_ap: Optional[float]
    test_top1: float
    test_top5: float
    test_macro_ap: float
    final_train_loss: Optional[float] = None
    final_val_loss: Optional[float] = None
    best_val_loss: Optional[float] = None
    params_total: Optional[int] = None
    params_trainable: Optional[int] = None
    flops: Optional[int] = None
    best_epoch: Optional[int] = None
    notes: str = ""


class IndexedDataset(Dataset):
    def __init__(self, base_dataset: Dataset, indices: Sequence[int]):
        self.base_dataset = base_dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.base_dataset[self.indices[idx]]

def convert_to_rgb(img):
    return img.convert("RGB")


def build_deep_base_dataset(data_dir: str) -> Dataset:
    transform = transforms.Compose([
    transforms.Lambda(convert_to_rgb),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])
    return datasets.Caltech256(root=data_dir, download=True, transform=transform)


def build_raw_base_dataset(data_dir: str) -> Dataset:
    return datasets.Caltech256(root=data_dir, download=True)


def infer_num_classes(raw_dataset: Dataset) -> int:
    labels = set()
    for i in range(len(raw_dataset)):
        _, label = raw_dataset[i]
        labels.add(int(label))
    return len(labels)


def resolve_split_counts(total_count: int, train_count: int, val_count: int, test_count: int) -> Tuple[int, int, int]:
    requested = train_count + val_count + test_count
    if requested > total_count:
        raise ValueError(
            f"Requested {requested} samples, but dataset only has {total_count}."
        )
    return train_count, val_count, test_count


def split_indices(total_count: int, train_count: int, val_count: int, test_count: int, seed: int) -> Tuple[List[int], List[int], List[int]]:
    indices = list(range(total_count))
    rng = random.Random(seed)
    rng.shuffle(indices)
    train_idx = indices[:train_count]
    val_idx = indices[train_count: train_count + val_count]
    test_idx = indices[train_count + val_count: train_count + val_count + test_count]
    return train_idx, val_idx, test_idx


def pad_tensor_batch(images: List[torch.Tensor], patch_multiple: int = 16) -> torch.Tensor:
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    max_h = int(math.ceil(max_h / patch_multiple) * patch_multiple)
    max_w = int(math.ceil(max_w / patch_multiple) * patch_multiple)

    batch = []
    for img in images:
        _, h, w = img.shape
        pad_h = max_h - h
        pad_w = max_w - w
        padded = F.pad(img, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
        batch.append(padded)
    return torch.stack(batch, dim=0)


def deep_collate_fn(batch):
    images, labels = zip(*batch)
    return pad_tensor_batch(list(images)), torch.tensor(labels, dtype=torch.long)


def create_deep_loaders(
    data_dir: str,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    test_idx: Sequence[int],
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    base_dataset = build_deep_base_dataset(data_dir)
    raw_dataset = build_raw_base_dataset(data_dir)
    num_classes = infer_num_classes(raw_dataset)

    train_loader = DataLoader(
        IndexedDataset(base_dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=deep_collate_fn,
    )
    val_loader = DataLoader(
        IndexedDataset(base_dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=deep_collate_fn,
    )
    test_loader = DataLoader(
        IndexedDataset(base_dataset, test_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=deep_collate_fn,
    )
    return train_loader, val_loader, test_loader, num_classes


def get_deep_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "vit_b_16":
        if timm is None:
            raise ImportError("timm is required for ViT-B/16. Install it with: pip install timm")
        model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=num_classes,
        )
        return model

    raise ValueError(f"Unsupported deep model: {model_name}")


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def maybe_compute_flops(model: nn.Module, loader: DataLoader, device: torch.device) -> Optional[int]:
    if FlopCountAnalysis is None:
        return None

    try:
        model.eval()
        images, _ = next(iter(loader))
        sample = images[:1].to(device)
        flops = FlopCountAnalysis(model, sample)
        return int(flops.total())
    except Exception:
        return None


def compute_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    num_classes: int,
) -> Tuple[float, float, float]:
    preds = y_score.argmax(axis=1)
    top1 = accuracy_score(y_true, preds)
    top5_k = min(5, num_classes)
    top5 = top_k_accuracy_score(y_true, y_score, k=top5_k, labels=np.arange(num_classes))
    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
    macro_ap = average_precision_score(y_true_bin, y_score, average="macro")
    return float(top1), float(top5), float(macro_ap)


def evaluate_deep(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    criterion: Optional[nn.Module] = None,
) -> Tuple[float, float, float, Optional[float], float]:
    model.eval()
    y_true: List[np.ndarray] = []
    y_score: List[np.ndarray] = []
    total_loss = 0.0
    total_samples = 0

    if device.type == "cuda":
        torch.cuda.synchronize()
    infer_start = time.perf_counter()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels) if criterion is not None else None
            probs = torch.softmax(logits, dim=1)
            y_true.append(labels.cpu().numpy())
            y_score.append(probs.cpu().numpy())
            if loss is not None:
                batch_size = labels.size(0)
                total_loss += float(loss.detach().item()) * batch_size
                total_samples += batch_size

    if device.type == "cuda":
        torch.cuda.synchronize()
    infer_time = time.perf_counter() - infer_start

    y_true_np = np.concatenate(y_true)
    y_score_np = np.concatenate(y_score)
    top1, top5, macro_ap = compute_classification_metrics(y_true_np, y_score_np, num_classes)
    mean_loss = (total_loss / total_samples) if criterion is not None and total_samples > 0 else None
    return top1, top5, macro_ap, mean_loss, infer_time


def train_deep_model(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    train_count: int,
    val_count: int,
    test_count: int,
) -> RunResult:
    model = get_deep_model(model_name, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    params_total, params_trainable = count_parameters(model)
    flops = maybe_compute_flops(model, val_loader, device)

    best_epoch = None
    best_val_top1 = -1.0
    best_val_metrics = (None, None, None)
    best_val_loss = None
    final_train_loss = None
    final_val_loss = None

    if device.type == "cuda":
        torch.cuda.synchronize()
    train_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        seen_train_samples = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = labels.size(0)
            running_train_loss += float(loss.detach().item()) * batch_size
            seen_train_samples += batch_size

        final_train_loss = running_train_loss / seen_train_samples if seen_train_samples > 0 else None
        val_top1, val_top5, val_macro_ap, val_loss, _ = evaluate_deep(
            model,
            val_loader,
            device,
            num_classes,
            criterion=criterion,
        )
        final_val_loss = val_loss
        print(
            f"[{model_name}] epoch {epoch}/{epochs} - "
            f"train_loss={final_train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"val_top1={val_top1:.4f}, val_top5={val_top5:.4f}, val_macro_ap={val_macro_ap:.4f}"
        )
        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            best_val_metrics = (val_top1, val_top5, val_macro_ap)
            best_val_loss = val_loss
            best_epoch = epoch

    if device.type == "cuda":
        torch.cuda.synchronize()
    train_time = time.perf_counter() - train_start

    test_top1, test_top5, test_macro_ap, _, infer_time = evaluate_deep(
        model, test_loader, device, num_classes, criterion=criterion
    )

    return RunResult(
        dataset_name=DATASET_NAME,
        model_name=model_name,
        epochs=epochs,
        train_count=train_count,
        val_count=val_count,
        test_count=test_count,
        train_time_sec=train_time,
        inference_time_sec=infer_time,
        val_top1=best_val_metrics[0],
        val_top5=best_val_metrics[1],
        val_macro_ap=best_val_metrics[2],
        test_top1=test_top1,
        test_top5=test_top5,
        test_macro_ap=test_macro_ap,
        final_train_loss=final_train_loss,
        final_val_loss=final_val_loss,
        best_val_loss=best_val_loss,
        params_total=params_total,
        params_trainable=params_trainable,
        flops=flops,
        best_epoch=best_epoch,
        notes="deep_model; resize_for_deep_models=224x224",
    )


def get_sift_extractor():
    return cv2.SIFT_create(nfeatures=300)


def build_classical_split_arrays(data_dir: str, train_idx: Sequence[int], val_idx: Sequence[int], test_idx: Sequence[int]):
    raw_dataset = build_raw_base_dataset(data_dir)

    def collect(indices: Sequence[int]):
        images = []
        labels = []
        for idx in indices:
            image, label = raw_dataset[idx]
            image = np.array(image)
            image = cv2.resize(image, (224, 224))
            if image.ndim == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            images.append(gray)
            labels.append(int(label))
        return images, np.array(labels, dtype=np.int64)

    x_train, y_train = collect(train_idx)
    x_val, y_val = collect(val_idx)
    x_test, y_test = collect(test_idx)
    num_classes = infer_num_classes(raw_dataset)
    return x_train, y_train, x_val, y_val, x_test, y_test, num_classes


def extract_sift_descriptors(images: List[np.ndarray]) -> List[np.ndarray]:
    extractor = get_sift_extractor()
    descs_per_image = []
    for idx, image in enumerate(images, start=1):
        keypoints, descriptors = extractor.detectAndCompute(image, None)
        if descriptors is None or len(keypoints) == 0:
            descriptors = np.zeros((1, extractor.descriptorSize()), dtype=np.float32)
        descs_per_image.append(descriptors.astype(np.float32))
        if idx % 1000 == 0 or idx == len(images):
            print(f"[sift] extracted descriptors for {idx}/{len(images)} images")
    return descs_per_image


def fit_bovw_kmeans(train_descs: List[np.ndarray], vocab_size: int, max_descriptor_samples: int) -> MiniBatchKMeans:
    stacked = np.vstack(train_descs)
    if len(stacked) > max_descriptor_samples:
        sampled_idx = np.random.choice(len(stacked), size=max_descriptor_samples, replace=False)
        stacked = stacked[sampled_idx]
    print(f"Fitting MiniBatchKMeans on {len(stacked)} local descriptors...")
    kmeans = MiniBatchKMeans(n_clusters=vocab_size, random_state=42, batch_size=4096, n_init=10)
    kmeans.fit(stacked)
    return kmeans


def build_bovw_histograms(descs_per_image: List[np.ndarray], kmeans: MiniBatchKMeans) -> np.ndarray:
    vocab_size = kmeans.n_clusters
    histograms = np.zeros((len(descs_per_image), vocab_size), dtype=np.float32)
    for i, descriptors in enumerate(descs_per_image):
        words = kmeans.predict(descriptors)
        hist = np.bincount(words, minlength=vocab_size).astype(np.float32)
        hist /= max(hist.sum(), 1.0)
        histograms[i] = hist
    return histograms


def determine_canvas_shape(all_images: List[np.ndarray]) -> Tuple[int, int]:
    max_h = max(img.shape[0] for img in all_images)
    max_w = max(img.shape[1] for img in all_images)
    return int(max_h), int(max_w)


def pad_image_to_shape(image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_shape
    h, w = image.shape[:2]
    pad_h = target_h - h
    pad_w = target_w - w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)


def extract_hog_features(images: List[np.ndarray], target_shape: Tuple[int, int]) -> np.ndarray:
    features = []
    for idx, image in enumerate(images, start=1):
        padded = pad_image_to_shape(image, target_shape)
        feat = hog(
            padded,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            transform_sqrt=True,
            feature_vector=True,
        )
        features.append(feat.astype(np.float32))
        if idx % 1000 == 0 or idx == len(images):
            print(f"[hog] extracted features for {idx}/{len(images)} images")
    return np.vstack(features)


def evaluate_classical_scores(
    y_true: np.ndarray,
    decision_scores: np.ndarray,
    num_classes: int,
) -> Tuple[float, float, float, int, int]:
    if decision_scores.ndim == 1:
        decision_scores = np.vstack([-decision_scores, decision_scores]).T
    return compute_classification_metrics(y_true, decision_scores, num_classes)


def train_sift_model(
    data_dir: str,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    test_idx: Sequence[int],
    vocab_size: int,
    max_descriptor_samples: int,
    train_count: int,
    val_count: int,
    test_count: int,
) -> RunResult:
    x_train, y_train, x_val, y_val, x_test, y_test, num_classes = build_classical_split_arrays(data_dir, train_idx, val_idx, test_idx)

    train_start = time.perf_counter()
    feat_start = time.perf_counter()
    train_descs = extract_sift_descriptors(x_train)
    val_descs = extract_sift_descriptors(x_val)
    test_descs = extract_sift_descriptors(x_test)
    feat_time = time.perf_counter() - feat_start

    bovw_start = time.perf_counter()
    kmeans = fit_bovw_kmeans(train_descs, vocab_size=vocab_size, max_descriptor_samples=max_descriptor_samples)
    x_train_bovw = build_bovw_histograms(train_descs, kmeans)
    x_val_bovw = build_bovw_histograms(val_descs, kmeans)
    x_test_bovw = build_bovw_histograms(test_descs, kmeans)
    bovw_time = time.perf_counter() - bovw_start

    svm_start = time.perf_counter()
    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("svm", LinearSVC(C=1.0, max_iter=5000, dual="auto")),
    ])
    clf.fit(x_train_bovw, y_train)
    svm_train_time = time.perf_counter() - svm_start
    train_time = time.perf_counter() - train_start

    val_scores = clf.decision_function(x_val_bovw)
    val_top1, val_top5, val_macro_ap = evaluate_classical_scores(y_val, val_scores, num_classes)

    infer_start = time.perf_counter()
    test_scores = clf.decision_function(x_test_bovw)
    _ = clf.predict(x_test_bovw)
    infer_time = time.perf_counter() - infer_start
    test_top1, test_top5, test_macro_ap = evaluate_classical_scores(y_test, test_scores, num_classes)

    notes = (
        f"classical_model=sift+svm; no_resize=True; vocab_size={vocab_size}; "
        f"max_descriptor_samples={max_descriptor_samples}; feature_extraction_time_sec={feat_time:.4f}; "
        f"bovw_build_time_sec={bovw_time:.4f}; svm_train_time_sec={svm_train_time:.4f}"
    )

    return RunResult(
        dataset_name=DATASET_NAME,
        model_name="sift+svm",
        epochs=1,
        train_count=train_count,
        val_count=val_count,
        test_count=test_count,
        train_time_sec=train_time,
        inference_time_sec=infer_time,
        val_top1=val_top1,
        val_top5=val_top5,
        val_macro_ap=val_macro_ap,
        test_top1=test_top1,
        test_top5=test_top5,
        test_macro_ap=test_macro_ap,
        best_epoch=None,
        notes=notes,
    )


def train_hog_model(
    data_dir: str,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    test_idx: Sequence[int],
    train_count: int,
    val_count: int,
    test_count: int,
) -> RunResult:
    x_train, y_train, x_val, y_val, x_test, y_test, num_classes = build_classical_split_arrays(data_dir, train_idx, val_idx, test_idx)
    canvas_shape = determine_canvas_shape(x_train + x_val + x_test)

    train_start = time.perf_counter()
    feat_start = time.perf_counter()
    x_train_hog = extract_hog_features(x_train, canvas_shape)
    x_val_hog = extract_hog_features(x_val, canvas_shape)
    x_test_hog = extract_hog_features(x_test, canvas_shape)
    feat_time = time.perf_counter() - feat_start

    svm_start = time.perf_counter()
    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("pca", PCA(n_components=512, random_state=42)),
        ("svm", LinearSVC(C=1.0, max_iter=3000, dual=False, verbose=1)),
    ])
    clf.fit(x_train_hog, y_train)
    svm_train_time = time.perf_counter() - svm_start
    train_time = time.perf_counter() - train_start

    val_scores = clf.decision_function(x_val_hog)
    val_top1, val_top5, val_macro_ap = evaluate_classical_scores(y_val, val_scores, num_classes)

    infer_start = time.perf_counter()
    test_scores = clf.decision_function(x_test_hog)
    _ = clf.predict(x_test_hog)
    infer_time = time.perf_counter() - infer_start
    test_top1, test_top5, test_macro_ap = evaluate_classical_scores(y_test, test_scores, num_classes)

    notes = (
        f"classical_model=hog+svm; no_resize=True; canvas_shape={canvas_shape}; "
        f"feature_extraction_time_sec={feat_time:.4f}; svm_train_time_sec={svm_train_time:.4f}"
    )

    return RunResult(
        dataset_name=DATASET_NAME,
        model_name="hog+svm",
        epochs=1,
        train_count=train_count,
        val_count=val_count,
        test_count=test_count,
        train_time_sec=train_time,
        inference_time_sec=infer_time,
        val_top1=val_top1,
        val_top5=val_top5,
        val_macro_ap=val_macro_ap,
        test_top1=test_top1,
        test_top5=test_top5,
        test_macro_ap=test_macro_ap,
        best_epoch=None,
        notes=notes,
    )


def save_results(results: List[RunResult], output_csv: str) -> None:
    if not results:
        return
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        if not file_exists:
            writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def pretty_print(results: List[RunResult]) -> None:
    print("\n=== FINAL RESULTS ===")
    for r in results:
        params = r.params_total if r.params_total is not None else "n/a"
        flops = r.flops if r.flops is not None else "n/a"
        print(
            f"{r.model_name:12s} | val_top1={r.val_top1 if r.val_top1 is not None else float('nan'):.4f} | "            f"test_top1={r.test_top1:.4f} | "
            f"test_top5={r.test_top5:.4f} | test_macro_ap={r.test_macro_ap:.4f} | "
            f"final_train_loss={r.final_train_loss if r.final_train_loss is not None else float('nan'):.4f} | "
            f"final_val_loss={r.final_val_loss if r.final_val_loss is not None else float('nan'):.4f} | "
            f"params={params} | flops={flops} | train_time={r.train_time_sec/60:.2f} min | infer_time={r.inference_time_sec:.2f} s"
        )


def parse_args():
    parser = argparse.ArgumentParser(description=f"{DATASET_NAME} benchmark: ResNet-50, ViT-B/16, SIFT+SVM, HoG+SVM")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--models", nargs="+", default=list(SUPPORTED_MODELS), choices=SUPPORTED_MODELS)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-count", type=int, default=18000)
    parser.add_argument("--val-count", type=int, default=6000)
    parser.add_argument("--test-count", type=int, default=4500)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--max-descriptor-samples", type=int, default=100000)
    parser.add_argument("--output-csv", type=str, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    args.seed = random.randint(1, 999)
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    raw_dataset = build_raw_base_dataset(args.data_dir)
    total_count = len(raw_dataset)
    train_count, val_count, test_count = resolve_split_counts(
        total_count=total_count,
        train_count=args.train_count,
        val_count=args.val_count,
        test_count=args.test_count,
    )
    train_idx, val_idx, test_idx = split_indices(
        total_count=total_count,
        train_count=train_count,
        val_count=val_count,
        test_count=test_count,
        seed=args.seed,
    )

    results: List[RunResult] = []

    deep_models = [m for m in args.models if m in {"resnet50", "vit_b_16"}]
    classical_models = [m for m in args.models if m in {"sift", "hog"}]

    if deep_models:
        train_loader, val_loader, test_loader, num_classes = create_deep_loaders(
            data_dir=args.data_dir,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        for model_name in deep_models:
            result = train_deep_model(
                model_name=model_name,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                num_classes=num_classes,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                train_count=train_count,
                val_count=val_count,
                test_count=test_count,
            )
            results.append(result)

    for model_name in classical_models:
        if model_name == "sift":
            result = train_sift_model(
                data_dir=args.data_dir,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                vocab_size=args.vocab_size,
                max_descriptor_samples=args.max_descriptor_samples,
                train_count=train_count,
                val_count=val_count,
                test_count=test_count,
            )
        elif model_name == "hog":
            result = train_hog_model(
                data_dir=args.data_dir,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                train_count=train_count,
                val_count=val_count,
                test_count=test_count,
            )
        else:
            raise ValueError(f"Unsupported classical model: {model_name}")
        results.append(result)

    if not results:
        raise RuntimeError("No models were run.")

    pretty_print(results)
    save_results(results, args.output_csv)
    print(f"\nSaved results to: {args.output_csv}")


if __name__ == "__main__":
    main()
