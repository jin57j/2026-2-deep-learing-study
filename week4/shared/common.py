"""공통 경로와 CPU 학습과 추론에 필요한 TensorFlow/Keras 설정."""

import os
import warnings
from pathlib import Path

WEEK4 = Path(__file__).resolve().parents[1]
ROOT = WEEK4 / "resnet" / "resnet-classroom"
DATA = ROOT / "data"
BREEDS = ["Birman", "Ragdoll", "Siamese", "Beagle", "Pug", "Shiba Inu"]

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ.setdefault("KERAS_HOME", str(ROOT / ".cache/keras"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

# TensorFlow 초기화 전에는 CPU와 스레드 수를 설정한다.
# 노트북에서 이미 연산했다면 변경할 수 없는 실행 설정을 유지한다.
try:
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except RuntimeError:
    warnings.warn(
        "TensorFlow가 이미 초기화되어 기존 실행 설정을 사용합니다. "
        "이 실습의 CPU와 스레드 설정을 적용하려면 커널을 재시작하고 "
        "TensorFlow 연산 전에 shared.common을 불러오세요.",
        RuntimeWarning,
        stacklevel=2,
    )
