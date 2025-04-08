# Fighting Label Noise: Empirical Study of Robust Classification Methods

This is an impementation of three papers dealing with label noise:
- Giorgio Patrini et al. “_Making deep neural networks robust to label noise: A loss correction
approach_” (**Backward correction**)
- Bo Han et al. “_Co-teaching: Robust training of deep neural networks with extremely noisy
labels_” (**Co-teaching**)
- Jiacheng Cheng et al. “_Learning with bounded instance and label-dependent label noise_” (**Anchor points**)

The repository was used to performe the experiments described [here](https://github.com/nagonch/noisy_label_classification/blob/main/fighting_label_noise.pdf).

## How to run the code
Install dependencies: `pip install -r requirements.txt`. We expect
the datasets to be located in folder named datasets .

### Backward correction
- Script to run backward correction: `backward_correction.py`
- Purpose: trains 10 models for the chosen dataset
- Required arguments:
  - `--dataset-name` — must be one of:
    - `FashionMNIST5`
    - `FashionMNIST6`
    - `CIFAR`
  - `--exp-name` — any string (experiment name)
- Example command: `python backward_correction.py --dataset-name CIFAR --exp-name test`


### Co-teaching
- Script to run co-teaching: `co_teaching.py`
- Purpose: trains 10 models for the chosen dataset
- Required arguments:
  - `--dataset-name` — must be one of:
    - `FashionMNIST5`
    - `FashionMNIST6`
    - `CIFAR`
  - `--exp-name` — any string (experiment name)
- Example command: `python co_teaching.py --dataset-name CIFAR --exp-name test`

### Backward correction and co-teaching evaluation
- Script to run model evaluation: `eval.py`
- Required arguments:
  - `--dataset-name` — must be one of:
    - `FashionMNIST5`
    - `FashionMNIST6`
    - `CIFAR`
  - `--model-folder` — folder containing multiple model weight files produced by `backward_correction.py` or `co_teaching.py`
- Example command: `python eval.py --dataset-name CIFAR --model-folder my_folder`
- Output: Evaluation metrics averaged over all models in the folder will be printed to the scree


### Anchor points
- Script to run estimation of all T matrices: `estimate_T.py`
- Purpose: estimates T matrices on all datasets using the anchor points method
- Example command: `python estimate_T.py`
- Output: Information about the matrices will be printed to the screen after a while


