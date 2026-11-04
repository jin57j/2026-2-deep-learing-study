"""A: ImageNet ResNet 특징을 고정하고 6품종 Head를 새로 학습한다."""

from shared.common import DATA, ROOT, keras
from shared.data import read_classification
from shared.models import make_breed_model, save_breed_weights


def main():
    keras.backend.clear_session()
    keras.utils.set_random_seed(42)
    images, labels, splits, _ = read_classification(DATA / "base/classification.csv")
    train, validation = splits == "train", splits == "validation"

    model, encoder, backbone, head = make_breed_model(pretrained=True)
    backbone.trainable = False
    features = encoder.predict(images, batch_size=32, verbose=0)

    head.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    head.fit(features[train], labels[train],
             validation_data=(features[validation], labels[validation]),
             batch_size=32, epochs=3, verbose=0)
    save_breed_weights(model, ROOT / "a_base/final.weights.h5")


if __name__ == "__main__":
    main()
