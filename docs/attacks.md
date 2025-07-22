## Attack Functionality

- Attack mechanisms enable Byzantine client scenarios in federated learning
- Attack parameters in `config.federated_params`:
    - `clients_attack_types`: Attacker types (`no_attack`, `label_flip`, `sign_flip`). Supports multiple types via list input: `federated_params.clients_attack_types=['label_flip','sign_flip']`
    - `prop_attack_clients`: Proportion of attacking clients (from 0 to 1)
    - `attack_scheme`: 
        - `constant`: Fixed attackers every round
        - `random_rounds`: Fixed attackers at random rounds (number controlled by `prop_attack_rounds`)
        - `random_clients`: Random attackers every round
        - `random_rounds_random_clients`: Random attackers at random rounds

Client attack mapping during initialization:

```python
from utils.attack_utils import map_attack_clients

client_map = map_attack_clients(cfg.federated_params.clients_attack_types,
                                cfg.federated_params.prop_attack_clients,
                                cfg.federated_params.amount_of_clients,
                                )
Output:
{
    0: 'no_attack',
    1: 'no_attack',
    2: 'label_flip',
    3: 'sign_flip',
    4: 'no_attack'
}
```

Dynamic mapping updates for random schemes:

```python
from utils.attack_utils import set_client_map_round

for round in range(self.rounds):
    self.client_map_round = set_client_map_round(
        self.client_map, self.attack_rounds, self.attack_scheme, round
    )
```

Attack type transmission via pipes:

```python

def get_communication_content(self, rank):
    return {
        "update_model": {
            k: v.cpu() for k, v in self.server.global_model.state_dict().items()
        },
        "attack_type": self.client_map_round[rank],
    }
```

Client-side attack activation:

```python

def _set_attack_type(self, attack_type):
    self.attack_type = attack_type
```

Add attack functionality

```python
from utils.attack_utils import add_attack_functionality

if client.attack_type != "no_attack":
    client = add_attack_functionality(client, client.attack_type)
```

#### Adding New Attacks

- Base attack client class:

```python
class AttackClient:
    def apply_attack(self, client_instance):
        """Apply attack functionality to existing client instance

        Args:
            client_instance (Client): Client instance to modify

        Returns:
            client_instance (Client): Modified client instance
        """
        raise NotImplementedError
```

This class contains attack implementation logic and `apply_attack` method. The `add_attack_functionality` method instantiates attack classes from configs (`configs/attacks/`). 
Adding new attacks requires implementing core functionality and connecting it to base client.

- Label flipping attack implementation example:
    1. Create config: `configs/attacks/label_flip.yaml`
    2. Implement label corruption:
```python
def _change_client_labels(self, train_df, data_name, rank):
    # Seed randomization in accordance with client rank
    rng = np.random.RandomState(rank)
    labels = np.array(train_df["target"].tolist())
    attacked_labels = rng.choice(
        np.prod(labels.shape),
        int(self.percent_of_changed_labels * np.prod(labels.shape)),
        replace=False,
    )
    corrupted_labels = rng.randint(0, 10, size=attacked_labels.size)
    labels.flat[attacked_labels] = corrupted_labels
    train_df.loc[train_df.index, "target"] = np.abs(labels)

    return train_df
```
3. Apply modifications
```python
def apply_attack(self, client_instance):
    client_instance.train_df = self._change_client_labels(
        client_instance.train_df,
        client_instance.cfg.dataset.data_name,
        client_instance.rank,
    )
    client_instance.train_loader = get_dataset_loader(
        client_instance.train_df, client_instance.cfg
    )
    return client_instance
```
- For method overrides, use `types.MethodType`:
```python
def apply_attack(self, client_instance):
    client_instance.percent_of_changed_grads = self.percent_of_changed_grads
    client_instance.get_grad = MethodType(
        SignFlipClient.get_grad_with_flipping, client_instance
    )
    return client_instance
```