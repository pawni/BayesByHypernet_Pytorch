# Implicit Weight Uncertainty in Neural Networks
This repository contains a reimplementation in Pytorch of some of the code for the paper Implicit Weight
Uncertainty in Neural Networks
([arXiv](https://arxiv.org/abs/1711.01297)).

## Abstract
Modern neural networks tend to be overconfident on unseen, noisy or
incorrectly labelled data and do not produce meaningful uncertainty
measures. Bayesian deep learning aims to address this shortcoming with
variational approximations (such as Bayes by Backprop or Multiplicative
Normalising Flows). However, current approaches have limitations
regarding flexibility and scalability. We introduce Bayes by Hypernet
(BbH), a new method of variational approximation that interprets
hypernetworks as implicit distributions. It naturally uses neural
networks to model arbitrarily complex distributions and scales to
modern deep learning architectures. In our experiments, we demonstrate
that our method achieves competitive accuracies and predictive
uncertainties on MNIST and a CIFAR5 task, while being the most robust
against adversarial attacks.

## Contact
For discussion, suggestions or questions don't hesitate to contact n.pawlowski16@imperial.ac.uk .