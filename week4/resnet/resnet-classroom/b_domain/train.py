"""B: Train 30장의 원본으로 conv4~5 + 기존 Head를 학습한다."""

from shared.common import DATA, ROOT, keras
from shared.data import make_dataset, read_classification
from shared.models import load_base, save_breed_weights


def main():
    keras.backend.clear_session()
    keras.utils.set_random_seed(42)
    images, labels, splits, _ = read_classification(DATA / "domain/manifest.csv")
    train, validation = splits == "train", splits == "validation"

    model, _, backbone, head = load_base()
    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = (layer.name.startswith(("conv4_", "conv5_"))
                           and isinstance(layer, keras.layers.Conv2D))
    head.trainable = True

    train_data = make_dataset(images[train], labels[train], batch_size=8, shuffle=True)
    validation_data = make_dataset(images[validation], labels[validation], batch_size=8)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    model.fit(train_data, validation_data=validation_data,
              epochs=4, shuffle=False, verbose=0)
    output = ROOT / "b_domain/final.weights.h5"
    save_breed_weights(model, output)


if __name__ == "__main__":
    main()
