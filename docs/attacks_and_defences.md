# Federated Learning Defence with Byzantine Attacks

Federated learning (FL) enables training across many clients without centralizing raw data, but the this setup makes it vulnerable to *Byzantine* clients that send poisoned updates. Robust aggregation and proactive defences aim to limit the influence of such clients and represents a research direction. This framework provides the ability to evaluate SoTA defence techniques against various Byzantine attacks in a variety of federated learning scenarios.

## List of supported Defences and attacks

**Defences**

* [FLTrust](https://arxiv.org/abs/2012.13995)
* [Zeno](https://arxiv.org/abs/1805.10032)
* [Central Clipping (CC)](https://arxiv.org/abs/2012.10333)
* [Safeguard](https://arxiv.org/abs/2012.13995)
* [Recess](https://arxiv.org/abs/2310.05431)
* [BANT](https://arxiv.org/abs/2505.07614)

**Pre-aggregation strategies**

* [Bucketing](https://arxiv.org/pdf/2006.09365)
* [Fixing By Mixing (FBM)](https://proceedings.mlr.press/v206/allouah23a/allouah23a.pdf)

**Attacks**

* Label Flip — data-poisoning by changing class labels on malicious clients.
* Random Grad. and Sign Flip — gradient-poisoning by noise or flipping direction of the gradient.
* [A Little Is Enough (ALIE)](https://arxiv.org/pdf/1902.06156)
* [Inner Product Manipulation (IPM)](https://arxiv.org/abs/1903.03936)

---

## Example results

We demonstrate the capabilities of FedXplore through small-scale experiment on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) using [ResNet-18](https://arxiv.org/abs/1512.03385) with multiple defense techniques under a range of adversarial attacks.

> **Reproduce this table:**
>
> ```bash
> python scripts/byz_cifar10_script.py > byz_cifar10_log_script.txt &
> ```

Test accuracy for Byzantine tolerance techniques under Various Attacks. The percentage defines the number of Byzantine clients. The table shows that existing defenses often excel against specific attacks but lack consistent protection overall.

| Defence        | No Attack | Label Flip (50%) | Sign Flip (60%) | IPM (50%) |
| -------------- | --------: | ---------------: | --------------: | --------: |
| FedAvg         |     0.902 |            0.207 |           0.100 |     0.832 |
| FLTrust        |     0.767 |            0.694 |           0.100 |     0.519 |
| Recess         |     0.887 |            0.633 |           0.100 |     0.774 |
| Zeno           |     0.910 |            0.156 |       **0.410** |     0.100 |
| CC             |     0.911 |            0.603 |           0.102 |     0.864 |
| CC + FBM       |     0.915 |            0.818 |           0.098 | **0.923** |
| CC + Bucketing |     0.845 |        **0.887** |           0.089 |     0.100 |
| Safeguard      | **0.918** |            0.102 |           0.100 |     0.112 |

*The table shows that many defences are effective only against a subset of attacks. For instance, Zeno performs well against sign-flip in our setup, while CC+FBM mitigates IPM strongly. No single method provides consistent protection across all attack types and attack proportions — highlighting the need for standardized, extensible benchmarks and combination strategies (e.g., pre-aggregation + aggregator)*.

---


<!-- ## Custom defences

TODO


## Custom attack

TODO -->
