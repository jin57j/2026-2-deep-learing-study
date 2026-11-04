"""D: A의 고정 특징에서 새 개 25품종 Head를 학습한다."""

from shared.common import DATA, ROOT, keras
from shared.data import read_classification
from shared.models import load_base, make_classification_head, save_head_weights


def main():
    keras.backend.clear_session()
    keras.utils.set_random_seed(42)
    images, labels, splits, _ = read_classification(DATA / "dog_breeds/manifest.csv")
    train, validation = splits == "train", splits == "validation"

    _, encoder, _, _ = load_base()
    encoder.trainable = False
    features = encoder.predict(images, batch_size=32, verbose=0)
    head = make_classification_head(25)

    head.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    head.fit(features[train], labels[train],
             validation_data=(features[validation], labels[validation]),
             batch_size=32, epochs=10, verbose=0)
    save_head_weights(head, ROOT / "d_dog_breeds/final.weights.h5")


if __name__ == "__main__":
    main()
