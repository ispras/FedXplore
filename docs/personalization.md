# Federated Learning Personalisation

Personalisation in Federated Learning (FL) aims to adapt a global model to heterogeneous client distributions so that each client achieves better local performance without centralising private data.


## List of supported methods

* [Ditto](https://arxiv.org/abs/2012.04221)
* [FedRep](https://arxiv.org/abs/2102.07078)
* [FedAMP](https://arxiv.org/abs/2007.03797)  
* [pFedMe](https://arxiv.org/abs/2006.08848)


## Example results

**Experimental setup**

* Dataset: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).  
* Model: [ResNet-18](https://arxiv.org/abs/1512.03385).  
* Topology: 10 clusters × 10 clients (100 clients total).  
* Local skew: each client is highly imbalanced — **90% of samples belong to a single dominant class**.  
* Metrics:
  * **Personalised Accuracy** — accuracy on a test split matching the client’s local distribution.
  * **Generalisation Accuracy** — accuracy on a global test set disjoint from all clients.

> **Reproduce this table:**
>
> ```bash
> python scripts/pers_cifar10_script.py > personalization_log.txt &
> ```

The results illustrate the trade-off between Finetune’s strong adaptation to local samples at the cost of generalization, FedAMP’s balanced performance on both metrics, and FedAvg’s baseline non-personalized performance.

| Method   | Personalised Accuracy | Generalisation Accuracy |
|---------:|---------------------:|-----------------------:|
| FedAvg   | 0.801                | 0.794                  |
| Finetune | 0.905                | 0.703                  |
| FedAMP   | 0.840                | 0.782                  |


*Finetune achieved strong personalization but at the cost of reduced generalization. FedAMP, in contrast, outperformed FedAvg in local adaptation while preserving generalization. These insights highlight the core trade-off in federated learning personalization between model specialization and generalization. Our framework supports a structured investigation of this balance, offering tools for reproducible and systematic experimentation.*

---

<!-- ## Custom personalisation method

TODO -->
