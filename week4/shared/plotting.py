"""같은 형식으로 품종 예측과 머리 위치를 보여준다."""

from .common import np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def show_classification(images, actual, logits, names, title):
    predicted = logits.argmax(axis=1)
    indices = np.linspace(0, len(images) - 1, 3, dtype=int)
    figure, axes = plt.subplots(1, 3, figsize=(10, 3))
    figure.suptitle(title)
    for ax, i in zip(axes, indices):
        ax.imshow(images[i])
        ax.set_title(f"Actual: {names[actual[i]]}\nPredicted: {names[predicted[i]]}",
                     fontsize=9, color="green" if actual[i] == predicted[i] else "red")
        ax.axis("off")
    figure.tight_layout()


def show_boxes(images, actual, predicted):
    indices = np.linspace(0, len(images) - 1, 3, dtype=int)
    figure, axes = plt.subplots(1, 3, figsize=(10, 3))
    figure.suptitle("E: Head localization / Green: actual / Red: predicted")
    for ax, i in zip(axes, indices):
        ax.imshow(images[i])
        height, width = images[i].shape[:2]
        scale = np.array([width, height, width, height])
        for box, color in [(actual[i], "green"), (predicted[i], "red")]:
            xmin, ymin, xmax, ymax = box * scale
            ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, edgecolor=color, linewidth=2))
        ax.axis("off")
    figure.tight_layout()
