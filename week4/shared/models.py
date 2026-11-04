"""ResNet 특징, Task별 Head, 최종 가중치 읽기와 저장."""

from .common import ROOT, keras, tf


@keras.saving.register_keras_serializable(package="student")
class ResNetPreprocess(keras.layers.Layer):
    def call(self, rgb):
        return keras.applications.resnet.preprocess_input(tf.cast(rgb, tf.float32))


def make_classification_head(classes):
    return keras.Sequential([
        keras.Input((2048,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(classes),
    ], name=f"head_{classes}")


def make_breed_model(pretrained=False):
    backbone = keras.applications.ResNet50(
        include_top=False, weights="imagenet" if pretrained else None,
        input_shape=(160, 160, 3))
    backbone.trainable = False
    rgb = keras.Input((160, 160, 3))
    spatial = backbone(ResNetPreprocess()(rgb), training=False)
    pooled = keras.layers.GlobalAveragePooling2D()(spatial)
    encoder = keras.Model(rgb, pooled, name="feature_encoder")
    head = make_classification_head(6)
    model = keras.Model(rgb, head(encoder.output), name="breed_classifier")
    return model, encoder, backbone, head


def load_base():
    model, encoder, backbone, head = make_breed_model()
    model.load_weights(ROOT / "a_base/final.weights.h5")
    return model, encoder, backbone, head


def save_breed_weights(model, path):
    # 새 추론용 구조에 값만 복사해 optimizer 상태를 제외한다.
    final, _, _, _ = make_breed_model()
    final.set_weights(model.get_weights())
    path.parent.mkdir(parents=True, exist_ok=True)
    final.save_weights(path)


def save_head_weights(head, path):
    final = keras.Sequential.from_config(head.get_config())
    final.set_weights(head.get_weights())
    path.parent.mkdir(parents=True, exist_ok=True)
    final.save_weights(path)
