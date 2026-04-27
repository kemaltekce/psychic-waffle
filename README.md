# psychic-waffle
Speech Emotion Research - Building Speech Emotion Recognition (SER) models

## Installation

1. Create an environment, e.g. with conda for a M-chip Mac with
  ```
  CONDA_SUBDIR=osx-arm64 conda create -n psychicwaffle python=3.12.4 -c conda-forge
  conda activate psychicwaffle
  conda config --env --set subdir osx-arm64
  ```
2. Install the project with
  ```
  pip install -e .
  ```
3. If you want to use the frozen/pinned package versions install them with
  ```
  pip install -r requirements.txt
  ```

## Data

This project uses the Ravdess dataset. Load the zip file `Audio_Speech_Actors_01-24.zip` from the [Affective Data Sience Lab](https://zenodo.org/records/1188976?preview_file=Audio_Speech_Actors_01-24.zip), save it in the `data/` directory and unzip it there. You should now have a folder in `data/` called `Ravdess_Audio_Speech_Actors_01-24` with other folders inside for each actor.


## Development

### Running the project

Run the project with `psy` after installing.

### Dependency management

If you want to install new dependencies, follow the instructions below:

1. Add the new package to `pyproject.toml`

2. Install it with `pip install -e .`

3. Freeze/pin versions in `requirements.txt` by running
  ```
  pip-compile pyproject.toml --resolver=backtracking
  ```

