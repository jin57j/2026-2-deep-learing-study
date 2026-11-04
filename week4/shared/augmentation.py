"""Train 변형본을 만들 때 밝기와 채도를 조금 바꾼다."""

from .common import tf


def make_brightness_saturation():
    rng = tf.random.Generator.from_seed(142)

    def augment(image, training=True):
        image = tf.cast(image, tf.float32)
        if not training:
            return image
        image = image / 255.0
        image = tf.image.adjust_brightness(image, rng.uniform((), -0.05, 0.05))
        image = tf.clip_by_value(image, 0.0, 1.0)
        image = tf.image.adjust_saturation(image, rng.uniform((), 0.9, 1.1))
        return tf.clip_by_value(image, 0.0, 1.0) * 255.0

    return augment
