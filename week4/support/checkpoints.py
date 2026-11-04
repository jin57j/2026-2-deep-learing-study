"""epoch별 저장과 validation 기준 상위 두 checkpoint의 관리."""

import json

from .runtime import keras, np, write_json


class EpochCheckpoints(keras.callbacks.Callback):
    """전체 흑백 입력 모델을 저장한다. adapter와 encoder도 함께 들어간다."""

    def __init__(self, run_dir, probe_images, probe_ids):
        super().__init__()
        self.run_dir = run_dir
        self.probe_images = probe_images
        self.probe_ids = list(probe_ids)
        self.records = []

    def on_epoch_end(self, epoch, logs=None):
        metrics = {key: float(value) for key, value in (logs or {}).items()}
        if not {"val_accuracy", "val_loss"} <= metrics.keys():
            raise ValueError("validation_data와 이름이 accuracy인 metric을 지정해주세요.")
        if not all(np.isfinite(value) for value in metrics.values()):
            raise ValueError("loss 또는 metric에 NaN/Inf가 있습니다. 학습 설정을 확인해주세요.")
        filename = f"epoch_{epoch + 1:03d}.weights.h5"
        self.model.save_weights(self.run_dir / filename)
        # 저장 시점의 예측을 남겨 커널 재시작 후 복원을 확인한다.
        logits = self.model(self.probe_images, training=False).numpy().tolist()
        self.records.append({"epoch": epoch + 1, "file": filename, **metrics,
                             "probe_ids": self.probe_ids, "probe_logits": logits})
        write_json(self.run_dir / "history.json", [
            {key: value for key, value in record.items() if key not in {"file", "probe_ids", "probe_logits"}}
            for record in self.records
        ])

    def keep_best(self, count=2):
        if len(self.records) < count:
            raise ValueError(f"상위 {count}개를 보관하려면 {count} epoch 이상 학습해주세요.")
        ordered = sorted(self.records, key=lambda row: (-row["val_accuracy"], row["val_loss"], row["epoch"]))
        selected = [{"rank": rank, **record} for rank, record in enumerate(ordered[:count], start=1)]
        write_json(self.run_dir / "checkpoints.json", selected)
        retained = {row["file"] for row in selected}
        for record in self.records:
            if record["file"] not in retained:
                (self.run_dir / record["file"]).unlink()
        print("validation 기준 checkpoint:")
        for row in selected:
            print(f"  {row['rank']}위: {row['file']} / accuracy={row['val_accuracy']:.4f} / loss={row['val_loss']:.4f}")
        return selected


def read_checkpoints(run_dir):
    with (run_dir / "checkpoints.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def check_restored_predictions(model, probe_images, checkpoint):
    actual = model(probe_images, training=False).numpy()
    expected = np.asarray(checkpoint["probe_logits"])
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5,
                               err_msg="저장 시점과 복원 후 예측이 다릅니다. 구조, 전처리, checkpoint를 확인해주세요.")
    maximum = float(np.max(np.abs(actual - expected)))
    print(f"복원 확인 통과 / 저장 시점과의 최대 차이: {maximum:.3g}")
    return maximum
