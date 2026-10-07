import json

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from psychic.data.cache import load_waveform_cache
from psychic.data.preprocessing import (
    build_preprocess_config,
    write_manifest,
    write_preprocess_config,
    write_splits,
)
from psychic.inference.model import load_model
from psychic.labels import EMOTION_LABELS
from psychic.training.engine import build_class_weights, evaluate, train
from psychic.training.metrics import (
    classification_metrics,
    format_classification_report,
)
from psychic.training.model import CNN, MODELS
from psychic.training.preprocessing import (
    time_mask,
    time_shift,
    waveforms_to_log_mel,
)


@pytest.fixture
def waveform_cache(tmp_path):
    cache_dir = tmp_path / "cache"
    (cache_dir / "tensors").mkdir(parents=True)
    records = []
    splits = {"split_id": "tiny_v1", "strategy": "speaker_disjoint"}
    time_sec = torch.arange(48000, dtype=torch.float32) / 16000
    for name, count in (("train", 9), ("val", 3), ("test", 5)):
        splits[name] = []
        splits[f"{name}_speakers"] = [name]
        for index in range(count):
            emotion_id = min(index % 4, 2)
            sample_id = f"{name}_{index}"
            tensor_path = f"tensors/{sample_id}.pt"
            waveform = 0.1 * torch.sin(
                2 * torch.pi * (220 + 220 * emotion_id) * time_sec
            )
            torch.save(waveform, cache_dir / tensor_path)
            records.append(
                {
                    "sample_id": sample_id,
                    "dataset": "simulation",
                    "sample_path": "original_audio_is_not_needed.wav",
                    "tensor_path": tensor_path,
                    "emotion": EMOTION_LABELS[emotion_id],
                    "emotion_id": emotion_id,
                    "speaker_id": name,
                    "duration_sec": 3.0,
                    "metadata": {},
                }
            )
            splits[name].append(sample_id)
    write_preprocess_config(cache_dir)
    write_manifest(cache_dir, records)
    write_splits(cache_dir, splits)
    return cache_dir


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("kernel_size, pool_dim", [(4, (3, 5)), (3, (4, 1))])
def test_cnn_custom_dimensions_pool_by_position_and_backpropagate(
    device, kernel_size, pool_dim
):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")
    torch.manual_seed(7)
    model = CNN(
        conv_out_channels=(4, 4, 4, 4),
        dropout_p=0.0,
        hidden_dim1=12,
        hidden_dim2=6,
        output_dim=3,
        kernel_size=kernel_size,
        avg_pool_dim=pool_dim,
    ).to(device)
    features = model.preprocess_data(torch.randn(2, 48000))
    assert features.device.type == "cpu"
    x = features.to(device).requires_grad_()
    logits = model(x)
    assert logits.shape == (2, 3)
    assert model.features[-1].output_size == pool_dim
    assert (
        sum(isinstance(layer, nn.MaxPool2d) for layer in model.features) == 3
    )
    logits.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert model.features[0].weight.grad.abs().sum().item() > 0


def test_checkpoint_reconstructs_custom_cnn_settings(tmp_path):
    model = CNN(
        conv_out_channels=(4, 4, 8),
        dropout_p=0.2,
        hidden_dim1=12,
        hidden_dim2=6,
        avg_pool_dim=(2, 4),
        kernel_size=3,
    ).eval()
    config = {
        "model": model.build_model_config(),
        "waveform_preprocessing": build_preprocess_config(),
    }
    checkpoint = {
        "config": config,
        "model_state_dict": model.state_dict(),
    }
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)
    restored, restored_config = load_model(path)
    assert not restored.training
    assert restored_config == config
    assert restored_config["model"]["version"] == 1
    x = torch.randn(2, 1, 64, 301)
    with torch.no_grad():
        torch.testing.assert_close(restored(x), model(x))

    for field, invalid_value in (
        ("version", 2),
        ("labels", list(reversed(EMOTION_LABELS))),
        ("num_classes", 9),
        ("preprocessing", {"name": "raw_waveform"}),
    ):
        checkpoint["config"]["model"] = {
            **model.build_model_config(),
            field: invalid_value,
        }
        torch.save(checkpoint, path)
        with pytest.raises(AssertionError, match="model config mismatch"):
            load_model(path)


def test_log_mel_silence_and_batch_independence():
    generator = torch.Generator().manual_seed(7)
    waveforms = torch.stack(
        [
            torch.zeros(48000),
            torch.randn(48000, generator=generator),
        ]
    )
    features = waveforms_to_log_mel(waveforms)
    assert features.shape == (2, 1, 64, 301)
    assert features.dtype == torch.float32
    assert torch.count_nonzero(features[0]) == 0
    torch.testing.assert_close(
        features[1:], waveforms_to_log_mel(waveforms[1:])
    )
    assert features[1].mean().item() == pytest.approx(0, abs=1e-6)
    assert features[1].std(correction=0).item() == pytest.approx(1)
    with pytest.raises(AssertionError):
        waveforms_to_log_mel(waveforms.double())


def test_time_shift_pads_without_wrapping_or_changing_inputs():
    waveforms = torch.arange(1, 48001, dtype=torch.float32).repeat(32, 1)
    original = waveforms.clone()
    torch.manual_seed(7)
    shifted = time_shift(waveforms)
    assert shifted.shape == waveforms.shape
    assert torch.equal(waveforms, original)
    offsets = []
    for source, result in zip(waveforms, shifted, strict=True):
        nonzero = result.nonzero().flatten()
        assert 48000 - 800 <= len(nonzero) <= 48000
        # Ramps expose any wraparound, internal gaps, or changed samples.
        assert torch.all(result[nonzero].diff() == 1)
        assert torch.all(nonzero.diff() == 1)
        offset = int(nonzero[0] - (result[nonzero[0]] - 1))
        offsets.append(offset)
        source_indices = nonzero - offset
        assert torch.equal(result[nonzero], source[source_indices])
        assert nonzero[0] == 0 or nonzero[-1] == 47999
    assert min(offsets) < 0 < max(offsets)
    assert len(set(offsets)) > 1
    state = torch.get_rng_state().clone()
    assert torch.equal(time_shift(waveforms, max_shift_ms=0), waveforms)
    assert torch.equal(state, torch.get_rng_state())


def test_time_mask_hides_one_interval_per_clip_without_mutation():
    features = torch.ones(32, 1, 64, 301)
    original = features.clone()
    torch.manual_seed(7)
    masked = time_mask(features)
    assert torch.equal(features, original)
    assert masked.shape == features.shape
    mask = masked[:, 0, 0] == 0
    assert torch.equal(masked == 0, mask[:, None, None, :].expand_as(masked))
    assert 0 < mask.any(dim=1).sum() < len(features)
    assert len(torch.unique(mask, dim=0)) > 2
    for interval in mask:
        indices = interval.nonzero().flatten()
        assert len(indices) <= 10
        assert torch.all(indices.diff() == 1)
    assert torch.all(masked[masked != 0] == 1)
    assert (time_mask(features, max_frames=1, probability=1) == 0).sum() == (
        len(features) * 64
    )
    state = torch.get_rng_state().clone()
    assert torch.equal(time_mask(features, probability=0), features)
    assert torch.equal(time_mask(features, max_frames=0), features)
    assert torch.equal(state, torch.get_rng_state())


def test_preprocess_augmentation_flag_is_explicit_and_seeded():
    model = CNN()
    waveforms = torch.randn(4, 48000)
    original = waveforms.clone()
    expected = waveforms_to_log_mel(waveforms)
    for training in (True, False):
        model.train(training)
        state = torch.get_rng_state().clone()
        assert torch.equal(model.preprocess_data(waveforms), expected)
        assert torch.equal(
            model.preprocess_data(waveforms, augment=False), expected
        )
        assert torch.equal(state, torch.get_rng_state())
        torch.manual_seed(42)
        augmented = model.preprocess_data(waveforms, augment=True)
        torch.manual_seed(42)
        assert torch.equal(
            augmented, model.preprocess_data(waveforms, augment=True)
        )
        assert augmented.shape == (4, 1, 64, 301)
        assert torch.isfinite(augmented).all()
        assert not torch.equal(augmented, expected)
    assert torch.equal(waveforms, original)


def test_macro_f1_includes_absent_classes_and_rejects_invalid_ids():
    labels = torch.tensor([0, 0, 1, 1, 2])
    predictions = torch.tensor([0, 1, 1, 1, 1])
    metrics = classification_metrics(labels, predictions)
    assert metrics["accuracy"] == 3 / 5
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 2 / 3) / 8)
    assert metrics["labels"] == list(EMOTION_LABELS)
    matrix = torch.tensor(metrics["confusion_matrix"])
    assert matrix.shape == (8, 8)
    assert matrix.sum().item() == 5
    assert matrix[:3, :3].tolist() == [[1, 1, 0], [0, 2, 0], [0, 1, 0]]
    assert matrix[3:].sum().item() == matrix[:, 3:].sum().item() == 0
    assert metrics["per_emotion"]["neutral"] == pytest.approx(
        {"precision": 1, "recall": 0.5, "f1": 2 / 3, "support": 2}
    )
    assert metrics["per_emotion"]["calm"] == pytest.approx(
        {"precision": 0.5, "recall": 1, "f1": 2 / 3, "support": 2}
    )
    assert metrics["per_emotion"]["happy"] == {
        "precision": 0,
        "recall": 0,
        "f1": 0,
        "support": 1,
    }
    assert metrics["per_emotion"]["sad"] == {
        "precision": 0,
        "recall": 0,
        "f1": 0,
        "support": 0,
    }
    predicted_only = classification_metrics(
        torch.tensor([0]), torch.tensor([1])
    )
    assert predicted_only["per_emotion"]["calm"] == {
        "precision": 0,
        "recall": 0,
        "f1": 0,
        "support": 0,
    }
    formatted = format_classification_report(metrics)
    assert "rows=true, columns=predicted; counts" in formatted
    assert "Precision" in formatted and "Support" in formatted
    with pytest.raises(AssertionError):
        classification_metrics(labels, torch.tensor([0, 1, 1, 1, 8]))


def test_class_weights_balance_training_labels_and_handle_missing_classes():
    weights = build_class_weights([0, 7, 7, 7])
    torch.testing.assert_close(
        weights, torch.tensor([2.0, 1, 1, 1, 1, 1, 1, 2 / 3])
    )


@pytest.mark.parametrize(
    "class_weights", [None, torch.tensor([2.0, 0.5, 1, 1, 1, 1, 1, 1])]
)
def test_evaluate_weights_partial_batches_and_preserves_model_state(
    class_weights,
):
    model = CNN()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.classifier[-1].bias[0] = 2
    before = {
        name: value.clone() for name, value in model.state_dict().items()
    }
    labels = torch.tensor([0, 1, 1, 1, 1])
    dataset = TensorDataset(torch.zeros(5, 48000), labels)
    metrics = evaluate(
        model,
        DataLoader(dataset, batch_size=2),
        torch.device("cpu"),
        class_weights,
    )
    assert not model.training
    assert all(parameter.grad is None for parameter in model.parameters())
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])
    logits = torch.zeros(5, 8)
    logits[:, 0] = 2
    assert metrics["loss"] == pytest.approx(
        nn.functional.cross_entropy(
            logits, labels, weight=class_weights
        ).item()
    )
    assert metrics["accuracy"] == 1 / 5
    assert metrics["macro_f1"] == pytest.approx((2 / 6) / 8)
    assert metrics["confusion_matrix"][0][0] == 1
    assert metrics["confusion_matrix"][1][0] == 4
    assert metrics["per_emotion"]["calm"]["support"] == 4


@pytest.mark.parametrize(
    "problem, message",
    [
        ("label", "emotion_id must match"),
        ("overlap", "sample ids overlap"),
        ("speaker", "speakers overlap"),
        ("path", "inside cache_dir"),
        ("missing", "missing cached tensor"),
        ("contract", "unsupported waveform cache"),
    ],
)
def test_cache_rejects_invalid_metadata(waveform_cache, problem, message):
    manifest_path = waveform_cache / "manifest.jsonl"
    records = list(map(json.loads, manifest_path.read_text().splitlines()))
    splits = json.loads((waveform_cache / "splits.json").read_text())
    if problem == "label":
        records[0]["emotion_id"] = 7
    elif problem == "overlap":
        splits["val"].append(splits["train"][0])
    elif problem == "speaker":
        for record in records:
            if record["speaker_id"] == "val":
                record["speaker_id"] = "train"
        splits["val_speakers"] = ["train"]
    elif problem == "path":
        records[0]["tensor_path"] = "../outside.pt"
    elif problem == "missing":
        records[0]["tensor_path"] = "tensors/missing.pt"
    elif problem == "contract":
        path = waveform_cache / "preprocess_config.json"
        config = json.loads(path.read_text())
        config["labels"].reverse()
        path.write_text(json.dumps(config))
    write_manifest(waveform_cache, records)
    write_splits(waveform_cache, splits)
    with pytest.raises(AssertionError, match=message):
        load_waveform_cache(waveform_cache)


@pytest.mark.parametrize(
    "waveform",
    [
        torch.zeros(12),
        torch.zeros(48000).double(),
        torch.full((48000,), float("nan")),
    ],
)
def test_cache_checks_tensor_contents_when_loading(waveform_cache, waveform):
    torch.save(waveform, waveform_cache / "tensors/train_0.pt")
    datasets, _, _ = load_waveform_cache(waveform_cache)
    with pytest.raises(AssertionError):
        datasets["train"][0]


def test_training_selects_best_reloads_and_repeats_with_seed(
    waveform_cache, caplog
):
    caplog.set_level("INFO", logger="psychic.training.engine")
    kwargs = {
        "cache_dir": waveform_cache,
        "models_dir": waveform_cache.parent / "models",
        "epochs": 2,
        "batch_size": 4,
        "seed": 42,
        "device": "cpu",
    }
    run_dir = train(**kwargs)
    checkpoint = torch.load(run_dir / "checkpoint.pt", weights_only=True)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    report = json.loads((run_dir / "validation_report.json").read_text())
    test_report = json.loads((run_dir / "test_report.json").read_text())
    config = json.loads((run_dir / "config.json").read_text())
    best = max(metrics["history"], key=lambda row: row["val_macro_f1"])
    assert checkpoint["epoch"] == metrics["best_epoch"] == best["epoch"]
    assert checkpoint["val_macro_f1"] == metrics["val_macro_f1"]
    assert report["epoch"] == metrics["best_epoch"]
    assert report["split"] == "val"
    assert caplog.text.count("Best checkpoint validation report") == 1
    assert format_classification_report(report) in caplog.text
    assert test_report["epoch"] == metrics["best_epoch"]
    assert test_report["split"] == "test"
    assert caplog.text.count("Best checkpoint test report") == 1
    assert format_classification_report(test_report) in caplog.text
    assert checkpoint["config"] == config
    assert "format_version" not in checkpoint
    # Training counts are [3, 2, 4], whereas validation counts are [1, 1, 1].
    assert config["training"]["class_weights"] == [1, 1.5, 0.75, 1, 1, 1, 1, 1]
    assert config["training"]["augment"] is True
    assert config["training"]["augmentation"] == CNN().augmentation_config
    assert "test_accuracy" not in metrics
    model, loaded_config = load_model(run_dir / "checkpoint.pt")
    assert not model.training
    assert loaded_config == config
    datasets, _, _ = load_waveform_cache(waveform_cache)
    reloaded_metrics = evaluate(
        model,
        DataLoader(datasets["val"], batch_size=2),
        torch.device("cpu"),
        torch.tensor(config["training"]["class_weights"]),
    )
    for key in ("loss", "accuracy", "macro_f1"):
        assert reloaded_metrics[key] == pytest.approx(
            metrics[f"val_{key}"], abs=1e-6
        )
        assert report[key] == metrics[f"val_{key}"]
    for key in ("labels", "confusion_matrix", "per_emotion"):
        assert report[key] == reloaded_metrics[key]
    reloaded_test_metrics = evaluate(
        model,
        DataLoader(datasets["test"], batch_size=2),
        torch.device("cpu"),
        torch.tensor(config["training"]["class_weights"]),
    )
    for key in ("loss", "accuracy", "macro_f1"):
        assert test_report[key] == pytest.approx(
            reloaded_test_metrics[key], abs=1e-6
        )
    for key in ("labels", "confusion_matrix", "per_emotion"):
        assert test_report[key] == reloaded_test_metrics[key]
    assert sum(map(sum, test_report["confusion_matrix"])) == 5
    with torch.no_grad():
        features = model.preprocess_data(datasets["val"][0][0][None])
        assert model(features).shape == (1, 8)
    torch.manual_seed(42)
    initial = CNN()
    assert not torch.equal(
        model.features[0].weight, initial.features[0].weight
    )

    repeat_dir = train(**kwargs)
    assert repeat_dir != run_dir
    assert json.loads((repeat_dir / "metrics.json").read_text()) == metrics
    assert (
        json.loads((repeat_dir / "validation_report.json").read_text())
        == report
    )
    assert (
        json.loads((repeat_dir / "test_report.json").read_text())
        == test_report
    )
    repeated, _ = load_model(repeat_dir / "checkpoint.pt")
    for name, value in model.state_dict().items():
        assert torch.equal(value, repeated.state_dict()[name])

    checkpoint["config"]["model"]["labels"].reverse()
    torch.save(checkpoint, run_dir / "incompatible.pt")
    with pytest.raises(AssertionError, match="model config mismatch"):
        load_model(run_dir / "incompatible.pt")


@pytest.mark.parametrize("augment", [False, True])
def test_other_architecture_early_stops_and_reloads_its_best_model(
    waveform_cache,
    caplog,
    augment,
):
    caplog.set_level("INFO", logger="psychic.training.engine")
    expected_training_augmentation = augment

    class WaveformClassifier(nn.Module):
        """Tiny real model using waveform bins instead of spectrograms."""

        def __init__(self, bins=12):
            super().__init__()
            self.bins = bins
            self.classifier = nn.Linear(bins, len(EMOTION_LABELS))
            # Keep F1 tied while loss changes so best and last weights differ.
            with torch.no_grad():
                self.classifier.bias.zero_()
                self.classifier.bias[-1] = 2

        def preprocess_data(self, waveforms, augment=False):
            assert augment == (
                self.training and expected_training_augmentation
            )
            return waveforms.reshape(len(waveforms), self.bins, -1).mean(2)

        def forward(self, x):
            return self.classifier(x)

        def build_model_config(self):
            return {
                "name": "waveform_classifier",
                "version": 1,
                "labels": list(EMOTION_LABELS),
                "num_classes": len(EMOTION_LABELS),
                "init_args": {"bins": self.bins},
                "preprocessing": {"name": "waveform_bin_means"},
            }

    MODELS["waveform_classifier"] = WaveformClassifier
    try:
        run_dir = train(
            waveform_cache,
            waveform_cache.parent / "models",
            model_name="waveform_classifier",
            model_kwargs={"bins": 6},
            epochs=20,
            early_stopping_patience=2,
            batch_size=4,
            learning_rate=0.01,
            device="cpu",
            augment=augment,
        )
        model, config = load_model(run_dir / "checkpoint.pt")
    finally:
        del MODELS["waveform_classifier"]

    assert run_dir.name.endswith("_waveform_classifier")
    assert isinstance(model, WaveformClassifier)
    assert model.bins == 6
    assert config["model"]["preprocessing"]["name"] == "waveform_bin_means"
    datasets, _, _ = load_waveform_cache(waveform_cache)
    metrics = evaluate(
        model,
        DataLoader(datasets["val"], batch_size=2),
        torch.device("cpu"),
        torch.tensor(config["training"]["class_weights"]),
    )
    saved_metrics = json.loads((run_dir / "metrics.json").read_text())
    report = json.loads((run_dir / "validation_report.json").read_text())
    assert saved_metrics["best_epoch"] == 1
    assert report["epoch"] == 1
    assert len(saved_metrics["history"]) == 3
    assert (
        saved_metrics["history"][-1]["val_loss"]
        < saved_metrics["history"][0]["val_loss"]
    )
    assert config["training"]["early_stopping_patience"] == 2
    assert config["training"]["augment"] is augment
    assert list(run_dir.glob("*.pt")) == [run_dir / "checkpoint.pt"]
    assert not (run_dir / "checkpoint.tmp").exists()
    for key in ("loss", "accuracy", "macro_f1"):
        assert metrics[key] == pytest.approx(
            saved_metrics[f"val_{key}"], abs=1e-6
        )
    for key in ("labels", "confusion_matrix", "per_emotion"):
        assert report[key] == metrics[key]
    test_report = json.loads((run_dir / "test_report.json").read_text())
    assert test_report == {
        "epoch": 1,
        "split": "test",
        **evaluate(
            model,
            DataLoader(datasets["test"], batch_size=4),
            torch.device("cpu"),
            torch.tensor(config["training"]["class_weights"]),
        ),
    }
    (summary,) = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("Evaluation summary")
    ]
    assert summary.splitlines()[0] == "Evaluation summary (best epoch=1):"
    summary_rows = {}
    for line in summary.splitlines()[2:]:
        split, *values = line.split()
        summary_rows[split] = [float(value) for value in values]
    best_epoch = saved_metrics["history"][0]
    last_epoch = saved_metrics["history"][-1]
    for split in ("train", "val"):
        assert summary_rows[split] == pytest.approx(
            [
                best_epoch[f"{split}_{key}"]
                for key in ("loss", "accuracy", "macro_f1")
            ],
            abs=0.00005,
        )
        assert summary_rows[split][0] != round(last_epoch[f"{split}_loss"], 4)
    assert summary_rows["test"] == pytest.approx(
        [test_report[key] for key in ("loss", "accuracy", "macro_f1")],
        abs=0.00005,
    )


def test_final_test_failure_preserves_completed_training(waveform_cache):
    torch.save(torch.zeros(12), waveform_cache / "tensors/test_0.pt")
    models_dir = waveform_cache.parent / "models"
    with pytest.raises(AssertionError, match="waveform shape must match"):
        train(
            waveform_cache,
            models_dir,
            model_kwargs={"conv_out_channels": (4, 4)},
            epochs=2,
            batch_size=4,
            device="cpu",
        )

    (run_dir,) = models_dir.iterdir()
    metrics = json.loads((run_dir / "metrics.json").read_text())
    report = json.loads((run_dir / "validation_report.json").read_text())
    assert len(metrics["history"]) == 2
    assert report["epoch"] == metrics["best_epoch"]
    model, config = load_model(run_dir / "checkpoint.pt")
    assert not model.training
    assert config == json.loads((run_dir / "config.json").read_text())
    assert not (run_dir / "test_report.json").exists()
