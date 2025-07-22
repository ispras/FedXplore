## ⚙️ Federated Method

### 📋 Federated Method is represented by triples:
1. Federated learning algorithm: see [Base Method](C4.md) for structure details.
2. Client algorithm: see [Base Client](C4.md) for structure details.
3. Server algorithm: see [Base Server](C4.md) for structure details.

### 🔩 Implementation a FedaAvg with [Proximal Term](https://arxiv.org/pdf/1812.06127)

1. The main difference from the [Federated Averaging](https://arxiv.org/pdf/1602.05629) algorithm is the introduction of a regularizer into the client's local loss function:
$$
\ell_i(w, x, y) \rightarrow \ell_i(w, x, y) + \dfrac{\lambda}{2}\|w_i - w_g\|^2
$$

2. To do this, you need to redefine the `Client Algorithm`: `client.py --> fedprox_client.py`
```python
loss = super().get_loss_value(outputs, targets)
proximity = (
    0.5
    * self.fed_prox_lambda
    * sum(
        [
            (p.float() - q.float()).norm() ** 2
            for (_, p), (_, q) in zip(
                self.model.state_dict().items(),
                self.server_model_state.items(),
            )
        ]
    )
)
loss += proximity
```

3. We redefine `Client Algorithm`, so, we need to update `client_cls`:
```python
def _init_client_cls(self):
    super()._init_client_cls()
    self.client_cls = FedProxClient
    self.client_kwargs["client_cls"] = self.client_cls
    self.client_args.extend([self.fed_prox_lambda])
```

4. Let's also add a warmup parameter that specifies the round at which the proximal term is added. 
    - To do this, we need to pass it to clients
    ```python
    def get_communication_content(self, rank):
        # In fedprox we need additionaly send current round to warmup
        content = super().get_communication_content(rank)
        content["current_round"] = self.cur_round
        return content
    ```
    - And process it on the client side.
    ```python
    def create_pipe_commands(self):
        pipe_commands_map = super().create_pipe_commands()
        pipe_commands_map["current_round"] = self.set_cur_round
        return pipe_commands_map

    def set_cur_round(self, round):
        self.cur_com_round = round
    ```
    - So, change proximity term with condition
    ```python
    loss = super().get_loss_value(outputs, targets)
    if self.cur_com_round > self.num_fedavg_rounds - 1:
        proximity = (
            0.5
            * self.fed_prox_lambda
            * sum(
                [
                    (p.float() - q.float()).norm() ** 2
                    for (_, p), (_, q) in zip(
                        self.model.state_dict().items(),
                        self.server_model_state.items(),
                    )
                ]
            )
        )
        loss += proximity
    ``` 