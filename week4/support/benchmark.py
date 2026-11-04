"""두 블록을 같은 CPU 조건에서 간단히 비교하는 제공 코드."""

from time import perf_counter

from .runtime import keras, np, tf


def conv_macs_per_image(model):
    """Conv의 곱셈, 누산 한 쌍을 1 MAC으로 센다. BN/ReLU/Add는 제외한다."""
    total = 0
    for layer in model.layers:
        if isinstance(layer, keras.layers.Conv2D):
            height, width = map(int, layer.output.shape[1:3])
            total += height * width * int(np.prod(layer.kernel.shape))
    return total


def compare_blocks(models, batch_size=8, warmup=10, repeats=30):
    """생성, tracing은 제외하고 출력 동기화까지 포함한 batch 추론 시간을 잰다."""
    shapes = {tuple(model.input_shape[1:]) for model in models.values()}
    if len(shapes) != 1:
        raise ValueError("같은 입력 shape의 블록을 비교해주세요.")
    input_shape = shapes.pop()
    rows = []
    with tf.device("/CPU:0"):
        sample = tf.random.stateless_uniform((batch_size, *input_shape), seed=(42, 4), dtype=tf.float32)
        for name, model in models.items():
            infer = tf.function(lambda x: model(x, training=False), jit_compile=False)
            for _ in range(warmup):
                infer(sample).numpy()
            durations = []
            for _ in range(repeats):
                start = perf_counter()
                infer(sample).numpy()  # 출력 계산이 끝날 때까지 기다린다.
                durations.append((perf_counter() - start) * 1000)
            rows.append({
                "block": name,
                "parameters": int(model.count_params()),
                "median_ms_per_batch": float(np.median(durations)),
                "conv_macs_per_image": conv_macs_per_image(model),
            })
    print(f"CPU / float32 / batch={batch_size} / warmup={warmup} / repeats={repeats}")
    print("파라미터 수는 BN의 비학습 통계도 포함합니다. 시간은 같은 환경 안에서 비교하세요.")
    print(f"{'block':14s} {'parameters':>12s} {'batch ms':>12s} {'Conv MACs/image':>18s}")
    for row in rows:
        print(f"{row['block']:14s} {row['parameters']:12,d} {row['median_ms_per_batch']:12.3f} {row['conv_macs_per_image']:18,d}")
    return rows
