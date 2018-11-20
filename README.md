# Attentive Guidance

This repository contains the implementation of *Attentive Guidance*, a technique that demonstrates that vanilla seq2seq models with attention can achieve compositional solutions without the need to explicitly build this into their architecture [1](https://arxiv.org/abs/1905.09657).

The repositary builds on top of the seq2seq library [machine](https://github.com/i-machine-think/machine) and adds to this library an attention loss and data field.
The file `train_model.py` is based on the example training script provided in machine, but adds command line arguments to facilitate training a model with attention loss.
The file `test.sh` contains several (toy) examples of how to invoke these parameters, for a complete overview, use the *help* function of the `train_model.py` script.


\[1\] [Hupkes D., Singh A.K., Korrel K., Kruszewski G. and, Bruni E. (2018) Learning compositionally through attentive guidance](https://arxiv.org/abs/1805.09657).
