"""기존 ResNet 실습 연결과 과제 데이터의 읽기와 저장."""

import csv
import json
import os
from pathlib import Path


# 다른 TensorFlow 연산 전에 공통 환경을 초기화한다.
from shared.common import DATA, WEEK4, keras, np, tf
from shared.data import read_images
from shared.models import load_base, make_breed_model


SOURCE_MANIFEST = DATA / "base" / "manifest.csv"
TASK_DATA = WEEK4 / "subtask" / "data"
RUNS = WEEK4 / "subtask" / "runs"


def load_frozen_encoder():
    """배포된 A의 전처리 -> ResNet -> GAP를 가져온다. 추가 다운로드는 없다."""
    _, encoder, _, _ = load_base()
    encoder.trainable = False
    return encoder


def make_empty_encoder():
    """전체 checkpoint 복원용 구조. ImageNet/A 가중치를 읽지 않는다."""
    _, encoder, _, _ = make_breed_model(pretrained=False)
    encoder.trainable = False
    return encoder


def read_source_manifest():
    with SOURCE_MANIFEST.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_species_manifest(source_rows, labels):
    """학생이 만든 label을 원본 경로 및 split과 묶어 새 CSV에 저장한다."""
    labels = np.asarray(labels)
    if labels.shape != (len(source_rows),) or not np.isin(labels, [0, 1]).all():
        raise ValueError("각 원본 이미지에 대응하는 0 또는 1 label을 만들어주세요.")
    TASK_DATA.mkdir(parents=True, exist_ok=True)
    manifest = TASK_DATA / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "path", "split", "breed_id", "label"])
        writer.writeheader()
        for row, label in zip(source_rows, labels):
            image = SOURCE_MANIFEST.parent / row["path"]
            writer.writerow({
                "image_id": row["image_id"],
                "path": Path(os.path.relpath(image, TASK_DATA)).as_posix(),
                "split": row["split"],
                "breed_id": row["breed_id"],
                "label": int(label),
            })
    (TASK_DATA / "classes.json").write_text(
        json.dumps(["cat", "dog"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def read_species_data(manifest):
    """이미지는 RGB 0~255로 읽는다. 흑백 변환은 학생 함수에서 처리한다."""
    images, splits, rows = read_images(Path(manifest))
    labels = np.array([int(row["label"]) for row in rows], dtype=np.int64)
    return images, labels, splits, rows


def show_data_summary(labels, splits):
    print("split       cat  dog  total")
    for split in ["train", "validation", "test"]:
        selected = labels[splits == split]
        counts = np.bincount(selected, minlength=2)
        print(f"{split:10s} {counts[0]:4d} {counts[1]:4d} {len(selected):5d}")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def create_run(config):
    """실행마다 새 폴더를 만들어 이전 결과와 제공 가중치를 보존한다."""
    RUNS.mkdir(parents=True, exist_ok=True)
    number = 1
    while (RUNS / f"run_{number:03d}").exists():
        number += 1
    run_dir = RUNS / f"run_{number:03d}"
    run_dir.mkdir()
    write_json(run_dir / "config.json", config)
    return run_dir


def show_predictions(gray_images, labels, logits):
    import matplotlib.pyplot as plt

    predicted = np.asarray(logits).argmax(axis=1)
    correct = np.flatnonzero(predicted == labels)
    wrong = np.flatnonzero(predicted != labels)
    candidates = list(correct[:1]) + list(wrong[:2])
    candidates += [i for i in range(len(labels)) if i not in candidates]
    figure, axes = plt.subplots(1, 3, figsize=(10, 3))
    for ax, i in zip(axes, candidates[:3]):
        ax.imshow(gray_images[i, ..., 0], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"Actual: {['cat', 'dog'][labels[i]]}\nPredicted: {['cat', 'dog'][predicted[i]]}")
        ax.axis("off")
    figure.tight_layout()
    plt.show()
