## A General Analysis of Example-Selection for Stochastic Gradient Descent
---
[Yucheng Lu](https://www.cs.cornell.edu/~yucheng/)\*, [Si Yi Meng](https://www.cs.cornell.edu/~siyimeng/)\* and [Christopher De Sa](http://www.cs.cornell.edu/~cdesa/)

In the [Tenth International Conference on Learning Representations (ICLR) 2022](https://iclr.cc/Conferences/2022).

**Note:**
This repository contains the source code for the empirical results in our [ICLR'22 paper](https://openreview.net/pdf?id=7gWSJrP3opB) on QMC-based data ordering analysis. It provides implementation on two Quasi-Monte-Carlo (QMC) related methods:
* Greedy sorting to construct better example ordering at the beginning of each epoch.
* QMC-based data augmentation.
  
## 1. Example Ordering with Greedy Sorting
One of the key insights from our [paper](https://openreview.net/pdf?id=7gWSJrP3opB) is that: if the examples are ordered in a way such that the averages of consecutive example gradients converge faster to the full gradient, then running SGD with such ordering will enjoy a faster convergence rate. Given this insight, we propose a greedy selection algorithm that can minimizes a metric named *Averaged Gradient Error*, which allows us to use better example ordering at the beginning of each epoch. Informally, we store the gradients computed for each mini-batch from the previous epoch, and then launch a sorting process over these gradients. To optimize the space/time complexity for the sorting, we additionally provide the optimization with random projection and QR decomposition. We provide an example script in [commands](https://github.com/EugeneLYC/qmc-ordering/tree/main/commands) with logistic regression on MNIST. One can run it with
```
bash commands/lg_mnist.sh
```

## 2. QMC-based Data Augmentation
The rationale for data augmentation is that by performing some reasonable random transformation on a given example, we assume the output would be another example that is identically distributed, and the expected value models an infinitely-large training set consisting of such transformed examples. Leveraging our insight from the greedy algorithm, we apply QMC points in data augmentation and expect the optimizer would converge faster to the population distribution (i.e., better generalization). We provide an example script in [commands](https://github.com/EugeneLYC/qmc-ordering/tree/main/commands) with Resnet20 on CIFAR10. One can run it with
```
bash commands/resnet_cifar10.sh
```


## 3. Citation
To cite this paper/code base, please use the following bibtex:
```
@inproceedings{
  lu2022a,
  title={A General Analysis of Example-Selection for Stochastic Gradient Descent},
  author={Yucheng Lu and Si Yi Meng and Christopher De Sa},
  booktitle={International Conference on Learning Representations},
  year={2022},
  url={https://openreview.net/forum?id=7gWSJrP3opB}
}

```