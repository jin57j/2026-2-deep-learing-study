"""머리 위치 회귀에 사용하는 공간 특징과 좌표 Head."""

from shared.common import keras, np, tf
from shared.models import ResNetPreprocess


@keras.saving.register_keras_serializable(package="student")
class OrderedBox(keras.layers.Layer):
    def call(self, values):
        p = tf.nn.sigmoid(values)
        # 왼쪽 위와 오른쪽 아래를 만들어 0 <= xmin <= xmax <= 1을 유지한다.
        return tf.stack([p[:, 0], p[:, 1],
                         p[:, 0] + (1 - p[:, 0]) * p[:, 2],
                         p[:, 1] + (1 - p[:, 1]) * p[:, 3]], axis=-1)


def make_spatial_encoder(backbone):
    core = keras.Model(backbone.input, backbone.get_layer("conv4_block6_out").output)
    core.trainable = False
    rgb = keras.Input((160, 160, 3))
    return keras.Model(rgb, core(ResNetPreprocess()(rgb), training=False), name="spatial_encoder")


def make_localization_head(initial_bias=None):
    if initial_bias is None:
        initial_bias = np.zeros(4)
    return keras.Sequential([
        keras.Input((10, 10, 1024)),
        keras.layers.LayerNormalization(axis=-1),
        keras.layers.Conv2D(32, 1, activation="relu"),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(4, kernel_initializer="zeros",
                           bias_initializer=keras.initializers.Constant(initial_bias)),
        OrderedBox(),
    ], name="localization_head")
