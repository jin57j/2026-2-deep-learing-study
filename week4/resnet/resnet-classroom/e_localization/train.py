"""E: A의 공간 특징에서 동물의 머리를 둘러싸는 좌표를 학습한다."""

from shared.common import DATA, ROOT, keras, np
from shared.data import read_localization
from shared.models import load_base, save_head_weights
from e_localization.model import make_localization_head, make_spatial_encoder


def main():
    keras.backend.clear_session()
    keras.utils.set_random_seed(42)
    images, boxes, splits, _ = read_localization(DATA / "base/manifest.csv")
    train, validation = splits == "train", splits == "validation"

    _, _, backbone, _ = load_base()
    encoder = make_spatial_encoder(backbone)
    features = encoder.predict(images, batch_size=16, verbose=0)

    # Train 평균 box에서 시작하도록 OrderedBox의 sigmoid 입력을 계산한다.
    mean = boxes[train].mean(axis=0)
    p = np.array([mean[0], mean[1], (mean[2] - mean[0]) / (1 - mean[0]),
                  (mean[3] - mean[1]) / (1 - mean[1])])
    head = make_localization_head(initial_bias=np.log(p / (1 - p)))
    head.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1),
        loss=keras.losses.Huber(delta=0.1),
    )
    head.fit(features[train], boxes[train],
             validation_data=(features[validation], boxes[validation]),
             batch_size=16, epochs=27, verbose=0)
    save_head_weights(head, ROOT / "e_localization/final.weights.h5")


if __name__ == "__main__":
    main()
