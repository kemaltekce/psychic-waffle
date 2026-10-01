<h1 align="center">
  <img src="img/icon.png" style="height: 250px">
  <br/>
  🔮 psychic-waffle 🧇
</h1>

Speech emotion recognition project with PyTorch and RAVDESS. The project
provides deterministic waveform caching, a training pipeline, and inference pipeline.
The eight output labels are `neutral`, `calm`, `happy`, `sad`, `angry`,
`fearful`, `disgust`, and `surprised`.

## 🚀 Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run
from the repository root:

```bash
uv sync
uv run psy --help
```

The project pins Python 3.12.4 and dependencies in `uv.lock`.

## &#x1F50A; Prepare the data

Download `Audio_Speech_Actors_01-24.zip` from the
[RAVDESS dataset](https://zenodo.org/records/1188976) and extract it so the actor
folders are directly under `data/original/ravdess/`:

```text
data/original/ravdess/
  Actor_01/03-01-01-01-01-01-01.wav
  ...
  Actor_24/
```

```bash
uv run psy preprocess
```

This rebuilds `data/preprocessed/waveform_16khz_3s_v1/`, replacing any previous
cache there. It stores mono float32 waveforms at 16 kHz, center padded/cropped
to three seconds (`[48000]`), plus the preprocessing contract, sample manifest,
and speaker-disjoint splits. For the complete dataset, actors 1–18 train,
19–21 validate, and 22–24 are held out for testing.

## Train, validate, and test

```bash
uv run psy train
```

Training reads cached tensors; it does not require the original audio files.
Every run validates after each epoch, selects the best checkpoint by validation
macro F1, then reloads that checkpoint and evaluates the test split once.
This final test evaluation also runs after early stopping.
Epochs, directories, learning rate, batch size, seed, and device are defined in
the `train` function in `psychic/training/engine.py`.

To try another architecture, add its class to `MODELS` in
`psychic/training/model.py` and set `CURRENT_MODEL` at the top of that file.
`psy train` uses that selection; Python calls can override it with `model_name`.
Each class provides
`preprocess_data(waveforms, augment=False)`, `forward(x)`, and
`build_model_config()`. The config
records its name, version, constructor `init_args`, labels, and preprocessing
settings. Models choose their own transforms from `preprocessing.py` and
return eight emotion logits in the canonical label order. Training and loading
use the same mapping, so neither needs architecture-specific branches.

Training defaults to `train(augment=True)`; use `train(augment=False)` for a
run without augmentation. The same flag reaches `preprocess_data` and is
saved in `config["training"]["augment"]`. Validation and test always pass
`augment=False`. Inference callers should also use `False` (the default).
These settings are recorded under `config["training"]["augmentation"]`.

Python experiments can adjust the current model's constructor arguments:

```python
from psychic.training.engine import train

train(model_kwargs={"dropout_p": 0.2}, epochs=10)
```

Every run creates a new folder:

```text
models/<timestamp>_<model_name>/
  checkpoint.pt
  config.json
  metrics.json
  validation_report.json
  test_report.json
```

- `checkpoint.pt` holds the best model state dict, epoch, score, and load-critical
  configuration. It can be reloaded on CPU or an available accelerator with
  `psychic.inference.model.load_model`, which selects the saved architecture from
  `MODELS` and verifies its config before loading weights. This rebuild is still
  model version 1. Preprocessing settings live inside `config["model"]`.
- `config.json` records model/feature settings, labels, waveform contract,
  seed, device, optimizer settings, class weights, early-stopping patience,
  and the split assignments used for the run. When calling `evaluate` directly,
  pass these class weights on the model's device to reproduce validation or
  test loss.
- `metrics.json` contains the selected epoch's metrics and every epoch's history.
  Training metrics describe the training pass with the chosen augmentation
  setting, when dropout is active and weights change between batches.
  Validation metrics describe the saved checkpoint on unaugmented inputs.
  The final evaluation summary shows train and validation metrics from this
  selected epoch, alongside test metrics from the reloaded best checkpoint.
- `validation_report.json` records the best checkpoint's epoch, validation loss,
  accuracy, macro F1, confusion matrix, and per-emotion precision, recall, F1,
  and support. Matrix rows are true emotions and columns are predictions, in
  the recorded label order; counts and per-emotion scores are unweighted.
  Undefined precision/recall/F1 are zero. Training also logs this report once
  at the end, using the selected epoch rather than the last epoch.
- `test_report.json` records the same fields for the held-out test split,
  using the reloaded best checkpoint. Test loss uses the training-derived
  class weights; accuracy, macro F1, confusion counts, and per-emotion scores
  are unweighted. The pipeline logs the test report once at the end.

Test tensors are only read after training and validation finish; test scores
do not affect checkpoint selection or early stopping. If final evaluation
fails, the checkpoint and training/validation reports remain saved.
To score an existing checkpoint without retraining, use
`psychic.inference.model.load_model` and `psychic.training.engine.evaluate`
from Python. `psy predict-file` is still pending. Checkpoints support
evaluation/inference loading; optimizer-state resumption is not implemented.
Data and generated model folders stay out of Git.

## Live microphone inference

From the repository root, with a trained checkpoint available:

```bash
uv run psy inference
```

The command loads `checkpoint.pt` from the newest timestamped run under
`models/` once, on CPU. It captures the system's default microphone at its
default sample rate, then resamples to the model's 16 kHz mono input. Allow
microphone access for your terminal when the operating system requests it.
An interactive terminal is required. Press Ctrl+C to stop.

After the initial three-second buffer fills, the display updates about every
250 ms with all eight emotions in fixed order. It always processes the newest
window, even when inference runs slowly. The audio callback only copies input
blocks; preprocessing, inference, and rendering run in the main loop.

Scores use an exponential moving average with `alpha = 0.3`, initialized from
the first prediction. A window is active when at least a tenth of its blocks
(15 of 150, 20 ms each) exceed an RMS volume threshold of -40 dBFS. These blocks
need not be consecutive. Below that count, the header reads
`Listening... but no speech`, all bars become zero, and smoothing resets.
This is a volume gate: background noise can activate it. Active windows show
`Listening... analysing...`; scores are model outputs, not calibrated emotion
certainty. Audio is held only in memory.

The volume threshold and smoothing factor are constants in
`psychic/inference/live.py`. Tune the threshold for your microphone and room:
a more negative value admits quieter sounds. The command has no additional
options and stops with an error if the newest checkpoint is incompatible or
microphone capture fails; it does not switch models or devices automatically.

Microphone capture uses [sounddevice](https://python-sounddevice.readthedocs.io/).
Its pip wheels include PortAudio on macOS and Windows. On Linux, install the
system PortAudio library if it is missing (for example, `libportaudio2` on
Debian/Ubuntu).

## Development

```bash
uv run ruff check .
uv run pytest
```

Use `uv add PACKAGE` for dependencies and `uv add --dev PACKAGE` for development
tools. Format changed Python files with `uv run ruff format PATH`.

`legacy/` contains the previous implementation for reference. The human
performance script in `scripts/` still needs migration to the rebuilt loader.
