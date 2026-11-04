"""C: 원본 30장 + 밝기·채도 15장 + shift(±8%) 15장으로 학습한다."""

from shared.common import DATA, ROOT, keras, np
from shared.augmentation import make_brightness_saturation
from shared.data import make_dataset, read_classification
from shared.models import load_base, save_breed_weights


def main():
    keras.backend.clear_session()
    keras.utils.set_random_seed(42)
    images, labels, splits, _ = read_classification(DATA / "domain/manifest.csv")
    train, validation = splits == "train", splits == "validation"
    original_images, original_labels = images[train], labels[train]

    augmentation = make_brightness_saturation()
    augmented_images = np.stack([
        augmentation(image, training=True).numpy() for image in original_images
    ])

    # 품종마다 밝기·채도 5장, shift 5장의 변형본을 만든다.
    group_rng = np.random.default_rng(141)
    shift = keras.layers.RandomTranslation(
        height_factor=0.08, width_factor=0.08,
        fill_mode="constant", fill_value=128,
        interpolation="bilinear", seed=143,
    )
    for breed in [0, 1, 2]:
        candidates = np.flatnonzero(original_labels == breed)
        for i in group_rng.choice(candidates, size=5, replace=False):
            augmented_images[i] = shift(original_images[i], training=True).numpy()

    train_images = np.concatenate([original_images, augmented_images])
    train_labels = np.concatenate([original_labels, original_labels])

    model, _, backbone, head = load_base()
    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = (layer.name.startswith(("conv4_", "conv5_"))
                           and isinstance(layer, keras.layers.Conv2D))
    head.trainable = True

    train_data = make_dataset(train_images, train_labels, batch_size=8, shuffle=True)
    validation_data = make_dataset(images[validation], labels[validation], batch_size=8)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    model.fit(train_data, validation_data=validation_data,
              epochs=4, shuffle=False, verbose=0)
    output = ROOT / "c_augmentation/final.weights.h5"
    save_breed_weights(model, output)


if __name__ == "__main__":
    main()
