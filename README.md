# Smallest Maximizer Criterion
Context tree estimation using Smallest Maximizer Criterion (SMC)

[![Python application](https://github.com/neuromat/smallest_maximizer_criterion/actions/workflows/python-app.yml/badge.svg)](https://github.com/neuromat/smallest_maximizer_criterion/actions/workflows/python-app.yml)

## Requirements
	python 3.7


## Installation
### Using pip

	`pip install g4l-smc`


### Direct download and run
	Download and extract the zip package to a folder
	Run `pip install -r requirements.txt`



## Command-line executables


### CTM

The following command can be used in order to estimate a context tree using CTM algorithm with the BIC criteria:

`python ctm.py -s fixtures/sample20000.txt -c 0.5 -d 6  ./my_model.tree`

Use `python ctm.py --help` for more information.

### SMC

The following command can be used in order to estimate the optimal context tree using SMC algorithm:

```
python smc.py -d 4 \
    -s examples/linguistic_case_study/folha.txt \
    -f ../test/results/bp \
    -A '0 1 2 3 4' \
    -p 4 \
    --split \> \
    --num_cores 4 \
    bic

```

Use `python smc --help` for more information.

#### Reports

A full html report will be available in the results folder passed as `-f` argument

### Sample Generation

To generate a sample given a context tree, use the following command:

Use CTM to estimate a model:

`python ctm.py -s tests/files/lipsum.txt -c 0 -d 6  ../lorem_ipsum.tree`

Generate a sample from the model

`python sample_gen.py -t ../lorem_ipsum.tree  -s 5000 ../lipsum_sample.txt`


## Citing

Please cite the following publication when using this algorithm:

Galves, Antonio & Galves, Charlotte & Garcia, Jesus & Garcia, Nancy & Leonardi, Florencia. (2009). Context tree selection and linguistic rhythm retrieval from written texts. The Annals of Applied Statistics. 6. 10.1214/11-AOAS511.


Bibtex version:

```
@article{Galves2012,
archivePrefix = {arXiv},
arxivId = {0902.3619},
author = {Galves, Antonio and Galves, Charlotte and Garc{\'{i}}a, Jes{\'{u}}s E. and Garcia, Nancy L. and Leonardi, Florencia},
doi = {10.1214/11-AOAS511},
eprint = {0902.3619},
issn = {19326157},
journal = {Annals of Applied Statistics},
keywords = {BIC,European and Brazilian Portuguese,Linguistic rhythm,Model selection,Smallest maximizer criterion,Variable length Markov chains},
mendeley-groups = {Neuromat},
number = {1},
pages = {186--209},
title = {{Context tree selection and linguistic rhythm retrieval from written texts}},
volume = {4},
year = {2012}
}

```

## Running tests

Run `pytest --cov=g4l -s tests/test_lcb.py`


## License

* The code in this repository is licensed under [GNU General Public License v3.0](LICENSE)


## Acknowledgement

This implementation was produced as part of the activities of FAPESP Research, Innovation and Dissemination Center for Neuromathematics (grant # 2020/04807-0, S.Paulo Research Foundation).


Universidade de São Paulo

Instituto de Matemática e Estatística

Research, Innovation and Dissemination Center for Neuromathematics - [NeuroMat](https://neuromat.numec.prp.usp.br/)

2020

