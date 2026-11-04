"""원본과 밝기·채도/shift 변형본으로 추가 학습한 C를 확인한다."""

from shared.common import BREEDS, DATA, ROOT, keras
from shared.data import read_classification
from shared.metrics import classification_score
from shared.models import make_breed_model
from shared.plotting import plt, show_classification


def main():
    keras.backend.clear_session()
    model, _, _, _ = make_breed_model()
    model.load_weights(ROOT / "c_augmentation/final.weights.h5")
    images, labels, splits, _ = read_classification(DATA / "domain/manifest.csv")
    for split in ["validation", "test"]:
        selected = splits == split
        logits = model.predict(images[selected], batch_size=8, verbose=0)
        accuracy, ce = classification_score(labels[selected], logits)
        print(f"C / domain {split}: accuracy {accuracy:.1%}, CE {ce:.4f}")
        if split == "test":
            show_classification(images[selected], labels[selected], logits, BREEDS, "C: Domain + Augmentation")


if __name__ == "__main__":
    main()
    plt.show()
