# noisy_label_classification

How to run the code Install dependencies: pip install -r requirements.txt . We expect
the datasets to be located in folder named datasets .

• **Backward correction**. The file to run backward correction is `backward correction.py`.
The file trains 10 models for the chosen dataset. Two required arguments
are: `--dataset-name` (must be one of: ”FashionMNIST5”, ”FashionM-
NIST6”, ”CIFAR”), `--exp-name` (anything you like). An example command:
`python backward_correction.py --dataset-name CIFAR --exp-name test` .
**WARNING**: the models are only saved if you specify `--save-model`, which doesn’t
happen by default!

• **Co-teaching**. The file to run backward correction is `co_teaching.py`. The
file trains 10 models for the chosen dataset. Two required arguments
are: `--dataset-name` (must be one of: ”FashionMNIST5”, ”FashionM-
NIST6”, ”CIFAR”), `--exp-name` (anything you like). An example command:
`python co_teaching.py --dataset-name CIFAR --exp-name test`. **WARNING**: the models are only saved if you specify `--save-model` , which doesn’t happen by
default!

• **Backward correction and co-teaching evaluation**. The file to run model evaluation is called `eval.py`. Two required arguments are: `--dataset-name`
(must be one of: ”FashionMNIST5”, ”FashionMNIST6”, ”CIFAR”),
`--model-folder` (the folder that contains multiple files with weight mod-
els, produced by either of the previous to scripts). An example command:
`python eval.py --dataset-name CIFAR --model-folder my folder`. The
evaluation metrics averaged over all the models in the folder will be printed to your screen.

• **Anchor points**. To run the estimation of all the T matrices (on all the datasets) through the
method of anchor points: `python estimate_T.py`. After a while, the information about
the matrices will be printed on your screen
