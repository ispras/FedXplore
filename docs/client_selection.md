# Client Selection

Client selection in Federated Learning (FL) determines which subset of clients participate in each communication round. Intelligent strategies can reduce communication overhead and mitigate non-i.i.d. harms. 

## List of supported Strategies

* [Power-of-Choice (POW-D)](https://arxiv.org/pdf/2010.01243)  
* [FedCor](https://arxiv.org/abs/2103.13822)  
* [Fed-CBS](https://arxiv.org/abs/2209.15245)  
* [DELTA](https://arxiv.org/abs/2205.13925)

---

## Experimental setup

* Dataset: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)  
* Model: [ResNet-18](https://arxiv.org/abs/1512.03385)  
* Clients: 100 clients total.  
* Data partitioning: Dirichlet distribution with `alpha = 0.1` (highly non-IID).  
* Participation per round: 25 clients are sampled each round.  
* Reproduce:  
  ```bash
  python scripts/cs_cifar10_script.py > cs_cifar10_log.txt &
  ```

| Method          | Communications         | Test Loss         | Accuracy         |
|-----------------|------------------------|-------------------|------------------|
| Uniform         | 17,767 ± 1,937         | 0.521 ± 0.009     | 0.822 ± 0.004    |
| POW-D           | **10,347 ± 493**       | 0.573 ± 0.012     | 0.812 ± 0.009    |
| FedCor          | 19,360 ± 557           | **0.449 ± 0.017**     | **0.848 ± 0.006**    |
| FedCBS          | 19,207 ± 837           | 0.507 ± 0.018     | 0.830 ± 0.007    |
| DELTA           | 15,700 ± 191           | 0.816 ± 0.019     | 0.721 ± 0.007    |

* Adversarial client strategies such as POW-D and DELTA provide lower performance while primarily addressing communication overhead.
* In contrast, FedCor and FedCBS aim to balance performance by selecting the most informative clients, which results in better accuracy but slightly higher communication costs.
* *These differences illustrate the communication-quality trade-off that arises in non-i.i.d. FL.*

<!-- ## Custom CS method

TODO -->