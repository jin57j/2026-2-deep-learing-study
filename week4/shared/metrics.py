"""분류 정확도, 교차 엔트로피, 사각형 IoU."""

from .common import keras, np


def classification_score(actual, logits):
    accuracy = float(np.mean(actual == logits.argmax(axis=1)))
    ce = float(np.mean(keras.losses.sparse_categorical_crossentropy(
        actual, logits, from_logits=True).numpy()))
    return accuracy, ce


def box_iou(actual, predicted):
    overlap = np.maximum(0, np.minimum(actual[:, 2:], predicted[:, 2:])
                         - np.maximum(actual[:, :2], predicted[:, :2]))
    intersection = np.prod(overlap, axis=1)
    actual_area = np.prod(actual[:, 2:] - actual[:, :2], axis=1)
    predicted_area = np.prod(predicted[:, 2:] - predicted[:, :2], axis=1)
    return intersection / np.maximum(actual_area + predicted_area - intersection, 1e-8)


