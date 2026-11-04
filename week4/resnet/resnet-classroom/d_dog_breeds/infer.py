"""A의 특징에 연결한 개 25품종 Head를 확인한다."""

import json

from shared.common import DATA, ROOT, keras
from shared.data import read_classification
from shared.metrics import classification_score
from shared.models import load_base, make_classification_head
from shared.plotting import plt, show_classification


def main():
    keras.backend.clear_session()
    _, encoder, _, _ = load_base()
    images, labels, splits, _ = read_classification(DATA / "dog_breeds/manifest.csv")
    test = splits == "test"
    features = encoder.predict(images[test], batch_size=32, verbose=0)
    head = make_classification_head(25)
    head.load_weights(ROOT / "d_dog_breeds/final.weights.h5")
    logits = head.predict(features, batch_size=32, verbose=0)
    accuracy, ce = classification_score(labels[test], logits)
    print(f"D / dog 25-breed Test: accuracy {accuracy:.1%}, CE {ce:.4f}")
    names = json.loads((DATA / "dog_breeds/classes.json").read_text())
    show_classification(images[test], labels[test], logits, names, "D: Dog breeds")


if __name__ == "__main__":
    main()
    plt.show()
