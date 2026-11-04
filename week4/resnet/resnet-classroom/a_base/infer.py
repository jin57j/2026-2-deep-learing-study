"""A의 기본 6품종 성능과 고양이 도메인 성능을 확인한다."""

from shared.common import BREEDS, DATA, keras
from shared.data import read_classification
from shared.metrics import classification_score
from shared.models import load_base
from shared.plotting import plt, show_classification


def main():
    keras.backend.clear_session()
    model, _, _, _ = load_base()
    for dataset in ["base", "domain"]:
        images, labels, splits, _ = read_classification(DATA / dataset / "manifest.csv")
        test = splits == "test"
        logits = model.predict(images[test], batch_size=8, verbose=0)
        accuracy, ce = classification_score(labels[test], logits)
        print(f"A / {dataset} Test: accuracy {accuracy:.1%}, CE {ce:.4f}")
        show_classification(images[test], labels[test], logits, BREEDS, f"A: {dataset}")


if __name__ == "__main__":
    main()
    plt.show()
