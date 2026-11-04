"""A의 공간 특징에 연결한 머리 위치 Head를 확인한다."""

from shared.common import DATA, ROOT, keras
from shared.data import read_localization
from shared.metrics import box_iou
from shared.models import load_base
from e_localization.model import make_localization_head, make_spatial_encoder
from shared.plotting import plt, show_boxes


def main():
    keras.backend.clear_session()
    _, _, backbone, _ = load_base()
    images, boxes, splits, _ = read_localization(DATA / "base/manifest.csv")
    test = splits == "test"
    encoder = make_spatial_encoder(backbone)
    features = encoder.predict(images[test], batch_size=16, verbose=0)
    head = make_localization_head()
    head.load_weights(ROOT / "e_localization/final.weights.h5")
    predicted = head.predict(features, batch_size=16, verbose=0)
    mean_iou = float(box_iou(boxes[test], predicted).mean())
    print(f"E / head localization Test: mean IoU {mean_iou:.3f}")
    show_boxes(images[test], boxes[test], predicted)


if __name__ == "__main__":
    main()
    plt.show()
