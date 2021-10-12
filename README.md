# turbowsr
Minimal implementation of BOWSR using TuRBO.

The `BOWSR` algorithm combines a surrogate structure-based energy model and bayesian optimisation to adjust the free parameters of symmetrised prototype structures obtain quasi-relaxed structures. In this implementation we use a pretrained `MEGNet` message passing neural network model as the surrogate energy model and carry out the bayesian optimisation using the trust region approach implemented in `TuRBO`. In order to get the free parameters of the symmetrised prototype structures we use the `aflowsym` symmetry finder implemented in `aflow`.

## Setup
Setup tbc using conda for everything if possible

## References

1. Zuo, Yunxing, et al. "Accelerating Materials Discovery with Bayesian Optimization and Graph Deep Learning." arXiv preprint arXiv:2104.10242 (2021). [arXiv](https://arxiv.org/abs/2104.10242)

2. Eriksson, David, et al. "Scalable global optimization via local bayesian optimization." Advances in Neural Information Processing Systems 32 (2019): 5496-5507. [arXiv](https://arxiv.org/abs/1910.01739)

## TODO

- [ ] Design a benchmark suit to compare against BOWSR and Wren
- [ ] Add tests
- [ ] Add docstrings and notes to code
- [ ] Implement TuRBO natively using botorch to keep code length minimal (this is for ability to parallelise more easily and reproducibility as TuRBO pip package unsupported)
- [ ] resolve how to handle errors when MEGNet fails
