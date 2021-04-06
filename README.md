# Smallest Maximizer Criterion

This algorithm implements the Smallest Maximizer Criterion, a consistent and constant free model selection procedure in the class of variable length Markov chains (VLMC), presented originally in the paper **Context tree selection and linguistic rhythm retrieval from written texts** [[1]](#1).

The package also provides an implementation of the Context Tree Maximizer (CTM) algorithm, which estimates context tree models consistently in linear time using the Bayesian Information Criteria (BIC) [[2]](#2). This approach relies on the specification of a constant, whereas its choice results on the estimation of different models. The SMC algorithm, in turn, is a constant-free procedure: given a list of consistent candidate models (__champion trees__), the procedure selects the smallest model that maximizes the likelihood of a sample. Two methods are available in the package for performing the champion trees set selection. The first method uses the CTM algorithm as presented in [[1]](#1), where the candidate models are located by scanning the model space for different constant values, and this method is proven to be consistent. The second method composes the set of champion trees by starting from the full tree, moving towards the empty tree by iteratively pruning the least contributive branches (LCB).

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

`python ctm.py -s fixtures/sample20000.txt - 0.5 -d 6  ./my_model.tree`

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


## References

<a id="1">[1]</a>

Galves, A., Galves, C., García, J. E., Garcia, N. L., & Leonardi, F. (2012). Context tree selection and linguistic rhythm retrieval from written texts. Annals of Applied Statistics, 4(1), 186–209. [https://doi.org/10.1214/11-AOAS511](https://doi.org/10.1214/11-AOAS511)


<a id="2">[2]</a>

Csiszar, I., & Talata, Z. (2006). Context tree estimation for not necessarily finite memory processes, via BIC and MDL. IEEE Transactions on Information Theory, 52(3), 1007–1016. [https://doi.org/10.1109/TIT.2005.864431](https://doi.org/10.1109/TIT.2005.864431)


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

