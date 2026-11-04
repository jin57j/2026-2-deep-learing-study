"""제공된 mini 이미지와 CSV를 읽는다. 원본 다운로드나 재가공은 필요 없다."""

import csv

from PIL import Image
from .common import np, tf


def read_images(csv_path):
    with csv_path.open() as handle:
        rows = list(csv.DictReader(handle))
    images = np.stack([
        np.asarray(Image.open(csv_path.parent / row["path"]).convert("RGB"))
        for row in rows
    ])
    splits = np.array([row["split"] for row in rows])
    return images, splits, rows


def read_classification(csv_path):
    images, splits, rows = read_images(csv_path)
    labels = np.array([int(row["breed_id"]) for row in rows])
    return images, labels, splits, rows


def read_localization(csv_path):
    images, splits, rows = read_images(csv_path)
    boxes = np.array([
        [float(row[key]) for key in ["xmin", "ymin", "xmax", "ymax"]]
        for row in rows
    ], dtype=np.float32)
    return images, boxes, splits, rows


def make_dataset(images, labels, batch_size, shuffle=False, augmentation=None):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        dataset = dataset.shuffle(len(images), seed=42, reshuffle_each_iteration=True)
    if augmentation is not None:
        # batch 전에 변형해 사진마다 다른 증강을 적용한다.
        dataset = dataset.map(lambda x, y: (augmentation(x, training=True), y))
    return dataset.batch(batch_size)
