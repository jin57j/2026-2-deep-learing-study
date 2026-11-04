"""학생 구현을 확인하는 셀프채점. 이 파일의 내부 구현은 필수 범위가 아니다."""

from .runtime import keras, np, tf


def _check_block(builder, filters, kernels, expected_parameters):
    model = builder()
    assert isinstance(model, keras.Model), "Keras Model을 반환해주세요. TODO를 먼저 완성했나요?"
    assert model.input_shape == (None, 16, 16, 64), "입력 shape는 (16, 16, 64)입니다."
    assert model.output_shape == (None, 16, 16, 64), "shortcut과 더할 수 있도록 출력 shape를 유지해주세요."
    convs = [layer for layer in model.layers if isinstance(layer, keras.layers.Conv2D)]
    assert [layer.filters for layer in convs] == filters, "Conv의 개수와 필터 수를 확인해주세요."
    assert [layer.kernel_size for layer in convs] == kernels, "Conv의 kernel 크기를 확인해주세요."
    assert all(layer.strides == (1, 1) and not layer.use_bias for layer in convs), "stride=1, use_bias=False 조건을 확인해주세요."
    assert all(layer.padding == "same" and layer.activation == keras.activations.linear for layer in convs), "padding='same'으로 두고 ReLU는 Conv 밖에 연결해주세요."
    expected_types = [keras.layers.InputLayer]
    for index in range(len(convs)):
        expected_types += [keras.layers.Conv2D, keras.layers.BatchNormalization]
        if index < len(convs) - 1:
            expected_types += [keras.layers.ReLU]
    expected_types += [keras.layers.Add, keras.layers.ReLU]
    assert [type(layer) for layer in model.layers] == expected_types, "Conv -> BN -> ReLU 순서와 마지막 Add -> ReLU를 확인해주세요."
    assert model.count_params() == expected_parameters, "BN의 scale/center와 Conv 설정을 확인해주세요."
    sample = np.linspace(-2, 2, 16 * 16 * 64, dtype=np.float32).reshape(1, 16, 16, 64)
    assert np.isfinite(model(sample, training=False).numpy()).all()
    # F(x)=0을 만들었을 때 shortcut이 살아 있어야 한다.
    for conv in convs:
        conv.kernel.assign(tf.zeros_like(conv.kernel))
    actual = model(sample, training=False).numpy()
    np.testing.assert_allclose(actual, np.maximum(sample, 0), rtol=1e-5, atol=1e-6,
                               err_msg="F(x)=0일 때 출력은 ReLU(x)여야 합니다. shortcut을 확인해주세요.")
    print("🎉 블록 채점 통과: shape, 구성, 파라미터 수, shortcut을 확인했습니다.")


def check_basic_block(builder):
    _check_block(builder, [64, 64], [(3, 3), (3, 3)], 74240)


def check_bottleneck_block(builder):
    _check_block(builder, [16, 16, 64], [(1, 1), (3, 3), (1, 1)], 4736)


def check_species_labels(function):
    sample = np.array([0, 1, 2, 3, 4, 5, 5, 0], dtype=np.int64)
    original = sample.copy()
    result = np.asarray(function(sample))
    assert result.shape == sample.shape, "입력 label마다 종 label 하나를 반환해주세요."
    assert np.issubdtype(result.dtype, np.integer), "label은 정수 배열이어야 합니다."
    np.testing.assert_array_equal(result, [0, 0, 0, 1, 1, 1, 1, 0], err_msg="품종 0~2는 cat(0), 3~5는 dog(1)입니다.")
    np.testing.assert_array_equal(sample, original, err_msg="원본 품종 배열은 변경하지 마세요.")
    print("종 label 채점 통과")


def check_grayscale(function):
    rgb = np.array([[[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]]], dtype=np.uint8)
    result = np.asarray(function(rgb))
    assert result.shape == (1, 2, 2, 1), "(batch, height, width, 1)을 유지해주세요."
    assert result.dtype == np.float32, "float32로 변환한 뒤 흑백으로 바꿔주세요."
    expected = tf.image.rgb_to_grayscale(tf.cast(rgb, tf.float32)).numpy()
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-4,
                               err_msg="0~255 RGB를 float32로 바꾼 뒤 tf.image.rgb_to_grayscale을 사용해주세요.")
    print("흑백 변환 채점 통과")


def check_dataset(rows, labels, splits):
    assert len(rows) == 540 and len(labels) == 540, "기본 원본 540장을 사용해주세요."
    per_source = {}
    for row in rows:
        per_source.setdefault(row["image_id"], set()).add(row["split"])
        expected = 0 if int(row["breed_id"]) < 3 else 1
        assert int(row["label"]) == expected, "품종과 종 label의 대응이 다릅니다."
    assert all(len(values) == 1 for values in per_source.values()), "원본 사진 하나가 여러 split에 들어갔습니다."
    for split, count in [("train", 360), ("validation", 90), ("test", 90)]:
        selected = labels[splits == split]
        assert len(selected) == count, "원본 split을 유지해주세요."
        np.testing.assert_array_equal(np.bincount(selected, minlength=2), [count // 2, count // 2])
    print("데이터 확인 통과: 원본 split, 종 label, 클래스별 개수가 맞습니다.")


def check_task_model(model, encoder):
    assert isinstance(model, keras.Model), "Keras Model을 반환해주세요."
    assert model.input_shape == (None, 160, 160, 1), "모델 입력은 흑백 한 채널입니다."
    assert model.output_shape == (None, 2), "cat/dog의 logit 두 개를 출력해주세요."
    assert encoder in model.layers and not encoder.trainable, "제공 encoder를 실제 경로에 연결하고 고정해주세요."
    adapter = model.get_layer("input_adapter")
    assert isinstance(adapter, keras.layers.Conv2D), "input_adapter는 Conv2D입니다."
    assert adapter.filters == 3 and adapter.kernel_size == (1, 1), "1x1 Conv로 3채널을 만들어주세요."
    assert adapter.trainable and adapter.count_params() == 6, "adapter의 kernel 3개와 bias 3개를 학습해야 합니다."
    np.testing.assert_array_equal(adapter.kernel.numpy(), np.ones((1, 1, 1, 3)))
    np.testing.assert_array_equal(adapter.bias.numpy(), np.zeros(3))
    dense = [layer for layer in model.layers if isinstance(layer, keras.layers.Dense)]
    assert [layer.units for layer in dense] == [64, 2], "새 head는 Dense(64) -> Dense(2)입니다."
    assert dense[0].activation == keras.activations.relu and dense[-1].activation == keras.activations.linear
    assert sum(int(np.prod(weight.shape)) for weight in model.trainable_weights) == 131272, "학습 범위에 encoder가 포함되지 않았는지 확인해주세요."
    sample = tf.reshape(tf.linspace(0.0, 255.0, 160 * 160), (1, 160, 160, 1))
    with tf.GradientTape() as tape:
        output = model(sample, training=False)
        objective = tf.reduce_sum(output)
    gradient = tape.gradient(objective, adapter.kernel)
    assert gradient is not None and bool(tf.reduce_all(tf.math.is_finite(gradient))), "adapter까지 gradient가 이어져야 합니다. 특징을 NumPy로 미리 고정하지 마세요."
    print("🎉 모델 확인 통과: adapter, head 학습 / encoder 고정 / gradient 연결")
