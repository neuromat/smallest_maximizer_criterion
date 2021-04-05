# Smallest Maximizer Criterion

We introduce a new criterion to select in a consistent way the probabilistic context tree generating a sample. The basic idea is to construct a totally ordered set of candidate trees. This set is composed by the "champion trees", the ones that maximize the likelihood of the sample for each number of degrees of freedom. The smallest maximizer criterion selects the infimum of the subset of champion trees whose gain in likelihood is negligible. In addition, we propose a new algorithm based on resampling to implement this criterion.

This study was motivated by the linguistic challenge of retrieving rhythmic features from written texts. Applied to a data set consisting of texts extracted from daily newspapers, our algorithm identifies different context trees for European Portuguese and Brazilian Portuguese. This is compatible with the long standing conjecture that European Portuguese and Brazilian Portuguese belong to different rhythmic classes. Moreover, these context trees have several interesting properties which are linguistically meaningful.


## Requirements
	python 3.7


## Installation
	`pip install g4l-smc`



## Command-line executables


### CTM

The following command can be used in order to estimate a context tree using CTM algorithm with the BIC criteria:

`python ctm.py -s fixtures/sample20000.txt - 0.5 -d 6  ./my_model.tree`

Use `python ctm.py --help` for more information.

### SMC

The following command can be used in order to estimate the optimal context tree using SMC algorithm:

```
python smc.py -d 4 \
    -s examples/linguistic_case_study/folha.txt \
    -f ../test/results \
    -p 4 \
    --num_cores 4 \
    bic

```

Use `python smc --help` for more information.



### Sample Generation

To generate a sample given a context tree, use the following command:

```
samplegen -t my_model.tree --size 5000 new_sample.txt

```

Use `smc --help` for more information.

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

