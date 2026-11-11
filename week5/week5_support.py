"""5주차 제공 코드. 데이터·자기점검·동일 조건 비교를 담당합니다."""
from collections import Counter, defaultdict
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import html
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from IPython.display import HTML, display


TOKENS = ("빨강", "파랑", "구름", "나무", "책", "물", "길", "바람", "별", "돌", "질문")
MODEL_NAMES = ("RNN", "LSTM", "GRU")
COLORS = {"RNN": "#3973ac", "LSTM": "#d38b28", "GRU": "#3c946d"}
DEFAULT_CONFIG = {
    "sequence_length": 96, "gaps": [2, 24, 72],
    "train_groups_per_gap": 200, "validation_groups_per_gap": 50,
    "test_groups_per_gap": 100,
    "data_seeds": {"train": 101, "validation": 202, "test": 303},
    "embedding_dim": 8, "units": 32, "epochs": 8,
    "batch_size": 64, "learning_rate": 1e-3, "clipnorm": 1.0,
    "model_seed": 42, "cpu_only": True,
}


def configure_runtime():
    """작은 실험의 CPU 설정. 초기화 이후에는 각 함수의 device scope를 사용."""
    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        pass
    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        pass
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def _scope(config):
    return tf.device("/CPU:0") if config["cpu_only"] else nullcontext()


def _context_key(row):
    # 색 지시만 가려서 같은 배경에서 만든 네 예시를 한 그룹으로 취급한다.
    context = row.copy()
    context[context < 2] = -1
    return hashlib.sha256(context.tobytes()).hexdigest()


def make_split(groups_per_gap, seed, config):
    rng = np.random.default_rng(seed)
    length = config["sequence_length"]
    xs, ys, gaps = [], [], []
    for gap in config["gaps"]:
        cue_index = length - gap - 2
        if not 6 <= cue_index < length - 1:
            raise ValueError("gap은 마지막 지시 앞에 네 지시를 놓을 수 있는 범위여야 합니다.")
        backgrounds = rng.integers(2, 10, (groups_per_gap, length), dtype=np.int32)
        backgrounds[:, -1] = 10
        middle_indices = (cue_index // 3, 2 * cue_index // 3)
        for label in (0, 1):
            for first_label in (0, 1):
                rows = backgrounds.copy()
                rows[:, 0] = first_label
                rows[:, cue_index] = label
                if first_label == label:
                    rows[:, middle_indices[0]] = 1 - label
                    rows[:, middle_indices[1]] = 1 - label
                else:
                    middle = rng.integers(0, 2, groups_per_gap, dtype=np.int32)
                    rows[:, middle_indices[0]] = middle
                    rows[:, middle_indices[1]] = 1 - middle
                xs.append(rows)
                ys.append(np.full(groups_per_gap, label, dtype=np.int32))
                gaps.append(np.full(groups_per_gap, gap, dtype=np.int32))
    x, y, gap = np.concatenate(xs), np.concatenate(ys), np.concatenate(gaps)
    order = rng.permutation(len(y))
    return {"x": x[order], "y": y[order], "gap": gap[order]}


def make_data(config):
    return {
        split: make_split(config[f"{split}_groups_per_gap"], config["data_seeds"][split], config)
        for split in ("train", "validation", "test")
    }


def check_data(data, config):
    contexts = {}
    for name, split in data.items():
        x, y, gaps = split["x"], split["y"], split["gap"]
        assert x.ndim == 2 and x.shape[1] == config["sequence_length"]
        assert len(x) == len(y) == len(gaps) and np.issubdtype(x.dtype, np.integer)
        assert np.all((0 <= x) & (x < len(TOKENS)))
        assert np.all(x[:, -1] == 10), "마지막 입력은 질문이어야 합니다."
        assert np.all((x == 0).sum(axis=1) == 2) and np.all((x == 1).sum(axis=1) == 2)
        marker_positions = np.where(x < 2, np.arange(x.shape[1]), -1).max(axis=1)
        assert np.array_equal(x[np.arange(len(x)), marker_positions], y), "정답은 마지막 색 지시입니다."
        assert np.array_equal(x.shape[1] - 2 - marker_positions, gaps)
        assert set(np.unique(gaps)) == set(config["gaps"])
        for gap in config["gaps"]:
            mask = gaps == gap
            counts = np.bincount(y[mask], minlength=2)
            assert counts[0] == counts[1] > 0
            for first in (0, 1):
                counts = np.bincount(y[mask & (x[:, 0] == first)], minlength=2)
                assert counts[0] == counts[1] > 0, "첫 지시만으로 정답을 알 수 없어야 합니다."
        groups = defaultdict(list)
        for row, target in zip(x, y):
            groups[np.bincount(row, minlength=len(TOKENS)).tobytes()].append(int(target))
        assert all(labels.count(0) == labels.count(1) for labels in groups.values())
        contexts[name] = {_context_key(row) for row in x}
    assert not contexts["train"] & contexts["validation"], "같은 배경 그룹이 train/validation에 중복됩니다."
    assert not contexts["train"] & contexts["test"], "같은 배경 그룹이 train/test에 중복됩니다."
    assert not contexts["validation"] & contexts["test"], "같은 배경 그룹이 validation/test에 중복됩니다."
    print("데이터 확인 완료: 정답 규칙·균형·그룹 분리·빈도 및 첫 지시 단서 점검")


def _table(headers, rows):
    head = "".join(f"<th style='padding:6px 12px;text-align:left'>{html.escape(str(h))}</th>" for h in headers)
    body = "".join("<tr>" + "".join(
        f"<td style='padding:6px 12px;border-top:1px solid #ddd'>{html.escape(str(value))}</td>"
        for value in row) + "</tr>" for row in rows)
    display(HTML(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"))


def show_data_summary(data):
    rows = []
    for name, split in data.items():
        counts = np.bincount(split["y"], minlength=2)
        rows.append((name, split["x"].shape, split["y"].shape, counts[0], counts[1]))
    _table(["split", "입력 shape", "정답 shape", "빨강(0)", "파랑(1)"], rows)


def _paired_indices(split, gap):
    candidates = np.flatnonzero(split["gap"] == gap)
    if not len(candidates):
        raise ValueError(f"없는 gap입니다: {gap}")
    first = int(candidates[0])
    key = _context_key(split["x"][first])
    first_color = split["x"][first, 0]
    matching = [int(i) for i in candidates if _context_key(split["x"][i]) == key
                and split["x"][i, 0] == first_color]
    assert len(matching) == 2 and set(split["y"][matching]) == {0, 1}
    return sorted(matching, key=lambda i: split["y"][i])


def show_sequence_pair(split, gap):
    indices = _paired_indices(split, gap)
    palette = ["#d45265", "#3479b7", "#e7ebef", "#aa8acc"]
    view = np.full((2, split["x"].shape[1]), 2)
    for j, index in enumerate(indices):
        row = split["x"][index]
        view[j, row == 0] = 0
        view[j, row == 1] = 1
        view[j, row == 10] = 3
        important = np.flatnonzero((row < 2) | (row == 10))
        print(f"예시 {j + 1}:", " | ".join(f"t={i}: {TOKENS[row[i]]}" for i in important),
              f"→ 정답 {TOKENS[split['y'][index]]} / 사이 토큰 수 {gap}")
    fig, ax = plt.subplots(figsize=(12, 2.3))
    ax.imshow(view, cmap=ListedColormap(palette), vmin=0, vmax=3, aspect="auto", interpolation="nearest")
    ax.set_yticks([0, 1], ["case 1: RED", "case 2: BLUE"])
    ax.set_xticks(sorted(set([0, 16, 32, 48, 64, 80, view.shape[1] - 1])))
    ax.set_xlabel("input position (0-based)")
    ax.set_title(f"Same background and token counts; gap = {gap}")
    ax.legend(handles=[Patch(facecolor=palette[i], label=label)
                       for i, label in enumerate(["RED", "BLUE", "other token", "QUERY"])],
              loc="upper center", bbox_to_anchor=(0.5, -0.38), ncol=4, frameon=False)
    fig.tight_layout()
    plt.show()


def check_model_builder(builder, config):
    try:
        classes = {"RNN": keras.layers.SimpleRNN, "LSTM": keras.layers.LSTM, "GRU": keras.layers.GRU}
        for name in MODEL_NAMES:
            keras.utils.set_random_seed(config["model_seed"])
            with _scope(config):
                model = builder(name, config)
                assert isinstance(model, keras.Model), f"{name}: Q9의 두 TODO를 완성하고 Model을 반환해주세요."
                assert model.input_shape == (None, config["sequence_length"]), f"{name}: 입력 shape를 확인해주세요."
                assert model.output_shape == (None, 2), f"{name}: 시퀀스 하나마다 확률 두 개를 출력해야 합니다."
                cells = [layer for layer in model.layers if isinstance(layer, (keras.layers.SimpleRNN, keras.layers.LSTM, keras.layers.GRU))]
                assert len(cells) == 1 and type(cells[0]) is classes[name], f"{name}: 지정한 순환층 한 개를 사용해주세요."
                assert cells[0].units == config["units"], f"{name}: hidden 크기를 공통 설정에 맞춰주세요."
                assert not cells[0].return_sequences and not cells[0].stateful
                embedding = next(layer for layer in model.layers if isinstance(layer, keras.layers.Embedding))
                assert not embedding.mask_zero, "ID 0은 빨강입니다. padding으로 가리면 안 됩니다."
                assert embedding.output_dim == config["embedding_dim"]
                sample = np.full((2, config["sequence_length"]), 2, dtype=np.int32)
                sample[:, -1] = 10
                sample[0, 0], sample[1, 0] = 0, 1
                probs = model(sample, training=False).numpy()
                assert np.all(np.isfinite(probs)) and np.all(probs >= 0), f"{name}: 확률 출력을 확인해주세요."
                assert np.allclose(probs.sum(axis=1), 1, atol=1e-5), f"{name}: 마지막 softmax를 확인해주세요."
        print("🎉 세 모델의 순환층·입출력·확률 출력을 확인했습니다. 정확도는 채점하지 않습니다.")
        return True
    except (AssertionError, TypeError, ValueError, AttributeError, StopIteration) as error:
        print("다시 확인해주세요:", error or "Embedding 연결을 확인해주세요.")
        return False


def _dataset(split, config):
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    # make_split의 동일한 셔플 결과를 매 모델에 동일한 순서로 제공한다.
    return tf.data.Dataset.from_tensor_slices((split["x"], split["y"])).batch(
        config["batch_size"]).with_options(options)


def train_comparison(builder, data, config, verbose=2):
    config = deepcopy(config)
    run = {"config": config, "models": {}, "histories": {}, "fit_seconds": {}, "parameters": {}}
    for name in MODEL_NAMES:
        keras.utils.set_random_seed(config["model_seed"])
        with _scope(config):
            model = builder(name, config)
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=config["learning_rate"], clipnorm=config["clipnorm"]),
                loss="sparse_categorical_crossentropy", metrics=["accuracy"],
            )
            if verbose:
                print(f"\n===== {name} / seed {config['model_seed']} =====")
            started = perf_counter()
            history = model.fit(_dataset(data["train"], config),
                                validation_data=_dataset(data["validation"], config),
                                epochs=config["epochs"], shuffle=False, verbose=verbose)
            run["fit_seconds"][name] = perf_counter() - started
        run["models"][name] = model
        run["parameters"][name] = model.count_params()
        run["histories"][name] = {key: [float(v) for v in values] for key, values in history.history.items()}
    return run


def plot_learning_curves(run):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name in MODEL_NAMES:
        h = run["histories"][name]
        epochs = np.arange(1, len(h["loss"]) + 1)
        for ax, key in zip(axes, ["loss", "accuracy"]):
            ax.plot(epochs, h[key], "--", color=COLORS[name], alpha=0.65, label=f"{name} train")
            ax.plot(epochs, h[f"val_{key}"], "-o", markersize=3, color=COLORS[name], label=f"{name} validation")
    for ax, title in zip(axes, ["Cross-entropy loss", "Accuracy"]):
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_xticks(epochs)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[1].set_ylim(0.45, 1.03)
    fig.tight_layout()
    plt.show()


def show_training_summary(run):
    rows = []
    for name in MODEL_NAMES:
        h = run["histories"][name]
        rows.append([name, f"{run['parameters'][name]:,}", f"{run['fit_seconds'][name]:.2f}",
                     f"{h['val_accuracy'][-1]:.3f}", f"{h['val_loss'][-1]:.4f}"])
    _table(["모델", "parameter 수", "학습 실행 시간(초)", "마지막 validation accuracy", "마지막 validation loss"], rows)


def evaluate_comparison(run, test):
    results = {}
    for name in MODEL_NAMES:
        with _scope(run["config"]):
            probabilities = run["models"][name].predict(_dataset(test, run["config"]), verbose=0)
        predicted = probabilities.argmax(axis=1)
        loss = float(np.mean(keras.losses.sparse_categorical_crossentropy(test["y"], probabilities).numpy()))
        results[name] = {"accuracy": float(np.mean(predicted == test["y"])), "loss": loss,
                         "probabilities": probabilities, "by_gap": {}}
        for gap in run["config"]["gaps"]:
            mask = test["gap"] == gap
            results[name]["by_gap"][int(gap)] = {"accuracy": float(np.mean(predicted[mask] == test["y"][mask])),
                                                "count": int(mask.sum())}
    return results


def show_test_summary(results, gaps):
    rows = [[name, f"{results[name]['accuracy']:.3f}", f"{results[name]['loss']:.4f}"]
            + [f"{results[name]['by_gap'][int(gap)]['accuracy']:.3f}" for gap in gaps] for name in MODEL_NAMES]
    _table(["모델", "test accuracy", "test loss"] + [f"gap={gap} accuracy" for gap in gaps], rows)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    positions = np.arange(len(gaps))
    for j, name in enumerate(MODEL_NAMES):
        ax.bar(positions + (j - 1) * 0.23, [results[name]["by_gap"][int(g)]["accuracy"] for g in gaps],
               width=0.23, label=name, color=COLORS[name])
    ax.axhline(0.5, color="#888888", linestyle="--", linewidth=1, label="balanced chance")
    ax.set_xticks(positions, [str(g) for g in gaps])
    ax.set_xlabel("tokens between last instruction and query")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1.08)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    plt.show()


def show_prediction_pair(results, test, gap):
    indices = _paired_indices(test, gap)
    rows = []
    for j, i in enumerate(indices):
        rows.append([f"예시 {j + 1}", TOKENS[test["y"][i]]] + [
            f"P(빨강)={results[name]['probabilities'][i, 0]:.5f}, P(파랑)={results[name]['probabilities'][i, 1]:.5f}"
            for name in MODEL_NAMES])
    _table(["입력", "정답"] + list(MODEL_NAMES), rows)


def serializable_result(run, test_results):
    return {
        "config": run["config"], "tensorflow": tf.__version__, "keras": keras.__version__,
        "device": "CPU" if run["config"]["cpu_only"] else "runtime default",
        "models": {name: {
            "parameters": run["parameters"][name], "fit_seconds": run["fit_seconds"][name],
            "history": run["histories"][name], "test_accuracy": test_results[name]["accuracy"],
            "test_loss": test_results[name]["loss"], "accuracy_by_gap": {
                str(g): row["accuracy"] for g, row in test_results[name]["by_gap"].items()},
        } for name in MODEL_NAMES},
    }


def show_reference_summary(path):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for run in document["runs"]:
        for name in MODEL_NAMES:
            result = run["models"][name]
            rows.append([run["config"]["model_seed"], name, result["parameters"],
                         f"{result['history']['val_accuracy'][-1]:.3f}",
                         f"{result['test_accuracy']:.3f}",
                         ", ".join(f"{g}:{a:.3f}" for g, a in result["accuracy_by_gap"].items())])
    _table(["seed", "모델", "parameter", "validation accuracy", "test accuracy", "gap: accuracy"], rows)
    print("제작자 환경:", document["environment"], "\n이 결과의 순위나 수치에 맞출 필요는 없습니다.")
