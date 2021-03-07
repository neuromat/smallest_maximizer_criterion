# Smallest Maximizer Criterion

We introduce a new criterion to select in a consistent way the probabilistic context tree generating a sample. The basic idea is to construct a totally ordered set of candidate trees. This set is composed by the "champion trees", the ones that maximize the likelihood of the sample for each number of degrees of freedom. The smallest maximizer criterion selects the infimum of the subset of champion trees whose gain in likelihood is negligible. In addition, we propose a new algorithm based on resampling to implement this criterion.

This study was motivated by the linguistic challenge of retrieving rhythmic features from written texts. Applied to a data set consisting of texts extracted from daily newspapers, our algorithm identifies different context trees for European Portuguese and Brazilian Portuguese. This is compatible with the long standing conjecture that European Portuguese and Brazilian Portuguese belong to different rhythmic classes. Moreover, these context trees have several interesting properties which are linguistically meaningful.


## Requirements
	python 3.8


## Installation
	`pip install -r requirements.txt`



## Examples

#### Estimation by pruning

run `python3 examples/estimation_by_pruning.py`


## Citing

Please cite the following publication when using this algorithm:

Galves, Antonio & Galves, Charlotte & Garcia, Jesus & Garcia, Nancy & Leonardi, Florencia. (2009). Context tree selection and linguistic rhythm retrieval from written texts. The Annals of Applied Statistics. 6. 10.1214/11-AOAS511.


Bibtex version:

```
@article{article,
author = {Galves, Antonio and Galves, Charlotte and Garcia,
          Jesus and Garcia, Nancy and Leonardi, Florencia},
year = {2009},
month = {02},
pages = {},
title = {Context tree selection and linguistic rhythm retrieval from written
texts},
volume = {6},
journal = {The Annals of Applied Statistics},
doi = {10.1214/11-AOAS511}
}
```

## Running tests

Run `pytest -s`

## License

* The code in this repository is licensed under [GNU General Public License v3.0](LICENSE)


## Acknowledgement

This implementation was produced as part of the activities of FAPESP Research, Innovation and Dissemination Center for Neuromathematics (grant # 2020/04807-0, S.Paulo Research Foundation).


Universidade de São Paulo

Instituto de Matemática e Estatística

Research, Innovation and Dissemination Center for Neuromathematics - [NeuroMat](https://neuromat.numec.prp.usp.br/)

2020

