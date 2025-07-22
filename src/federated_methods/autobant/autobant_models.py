from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate


class AutoBANTModel(nn.Module):
    def __init__(
        self, cfg, server_model_weights, client_updates, device, init_trust_scores=None
    ):
        super(AutoBANTModel, self).__init__()

        self.amount_of_clients = len(client_updates)
        self.device = device
        if init_trust_scores is None:
            initial_values = torch.tensor(
                [1.0 / self.amount_of_clients] * self.amount_of_clients
            )
        else:
            initial_values = init_trust_scores
        self.trust_scores = nn.Parameter(
            initial_values.clone().detach(), requires_grad=True
        )

        client_states = self.get_client_states(server_model_weights, client_updates)
        self.client_models = self.get_client_models(client_states, cfg)
        self.freeze_client_models_weights()

    def get_client_states(self, server_model_weights, client_updates):
        states = []
        for update in client_updates:
            state = OrderedDict()
            for key, weights1 in update.items():
                state[key] = weights1.to(self.device) + server_model_weights[key]
            states.append(state)
        return states

    def get_client_models(self, client_states, cfg):
        client_models = [
            instantiate(cfg.model).to(self.device)
            for i in range(self.amount_of_clients)
        ]
        for model, state in zip(client_models, client_states):
            model.load_state_dict(state)
        return client_models

    def freeze_client_models_weights(self):
        for model in self.client_models:
            for param in model.parameters():
                param.requires_grad = False


class AutoBANTModel2d(AutoBANTModel):
    def calc_basic_block(self, layers, ts, x):
        # First
        out = sum([ts[i] * layers[i].conv1(x) for i in range(self.amount_of_clients)])
        out = F.relu(
            sum([ts[i] * layers[i].bn1(out) for i in range(self.amount_of_clients)])
        )
        # Second
        out = sum([ts[i] * layers[i].conv2(out) for i in range(self.amount_of_clients)])
        out = sum([ts[i] * layers[i].bn2(out) for i in range(self.amount_of_clients)])
        # Shortcut
        out += self.calc_shortcut(
            [layers[i].shortcut for i in range(self.amount_of_clients)], ts, x
        )
        return F.relu(out)

    def calc_shortcut(self, layers, ts, x):
        if len(layers[0]) == 0:
            return x
        else:
            out = sum([ts[i] * layers[i][0](x) for i in range(self.amount_of_clients)])
            out = sum(
                [ts[i] * layers[i][1](out) for i in range(self.amount_of_clients)]
            )
            return out

    def forward(self, x):
        # to unit simplex
        # trust_scores = torch.softmax(self.trust_scores, dim=0)
        trust_scores = self.trust_scores
        # Init layers
        x = sum(
            [
                trust_scores[i] * self.client_models[i].conv1(x)
                for i in range(self.amount_of_clients)
            ]
        )
        x = F.relu(
            sum(
                [
                    trust_scores[i] * self.client_models[i].bn1(x)
                    for i in range(self.amount_of_clients)
                ]
            )
        )
        # Basic blocks
        for j in range(len(self.client_models[0].layer1)):
            x = self.calc_basic_block(
                [
                    self.client_models[i].layer1[j]
                    for i in range(self.amount_of_clients)
                ],
                trust_scores,
                x,
            )
        for j in range(len(self.client_models[0].layer2)):
            x = self.calc_basic_block(
                [
                    self.client_models[i].layer2[j]
                    for i in range(self.amount_of_clients)
                ],
                trust_scores,
                x,
            )
        for j in range(len(self.client_models[0].layer3)):
            x = self.calc_basic_block(
                [
                    self.client_models[i].layer3[j]
                    for i in range(self.amount_of_clients)
                ],
                trust_scores,
                x,
            )
        for j in range(len(self.client_models[0].layer4)):
            x = self.calc_basic_block(
                [
                    self.client_models[i].layer4[j]
                    for i in range(self.amount_of_clients)
                ],
                trust_scores,
                x,
            )
        # Final layers
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = sum(
            [
                trust_scores[i] * self.client_models[i].linear(x)
                for i in range(self.amount_of_clients)
            ]
        )
        return x


class AutoBANTModel1d(AutoBANTModel):
    def calc_stem(self, layers, ts, x):
        # MyConv
        out = sum([ts[i] * layers[i][0](x) for i in range(self.amount_of_clients)])
        # Bn
        out = sum([ts[i] * layers[i][1](out) for i in range(self.amount_of_clients)])
        # ReLU
        out = layers[0][2](out)
        # MyMaxPool
        out = sum([ts[i] * layers[i][3](out) for i in range(self.amount_of_clients)])
        return out

    def calc_backbone(self, layers, ts, x):
        for i in range(len(layers[0])):
            for j in range(len(layers[0][0])):
                x = self.calc_basic_block(
                    [layers[p][i][j] for p in range(self.amount_of_clients)], ts, x
                )
        return x

    def calc_basic_block(self, layers, ts, x):
        # print(f"Init shape: {x.size()}")
        identity = x
        # First
        out = sum([ts[i] * layers[i].conv1(x) for i in range(self.amount_of_clients)])
        out = F.relu(
            sum([ts[i] * layers[i].bn1(out) for i in range(self.amount_of_clients)])
        )
        # Check dropout if don't work
        out = sum([ts[i] * layers[i].do1(out) for i in range(self.amount_of_clients)])

        # Second
        out = sum([ts[i] * layers[i].conv2(out) for i in range(self.amount_of_clients)])
        out = F.relu(
            sum([ts[i] * layers[i].bn2(out) for i in range(self.amount_of_clients)])
        )
        # Check dropout if don't work
        out = sum([ts[i] * layers[i].do2(out) for i in range(self.amount_of_clients)])

        # print(f"Out size: {out.size()}")
        # Downsample
        if layers[0].downsample is not None:
            # print("Downsampling...")
            identity = self.calc_downsample(
                [layers[i].downsample for i in range(self.amount_of_clients)],
                ts,
                identity,
            )
            # print(f"Identity size: {identity.size()}")
        # shortcut
        out = out + identity
        return out

    def calc_downsample(self, layers, ts, x):
        out = sum([ts[i] * layers[i][0](x) for i in range(self.amount_of_clients)])
        out = sum([ts[i] * layers[i][1](out) for i in range(self.amount_of_clients)])
        return out

    def calc_head(self, layers, ts, x):
        x = self.calc_pooling_adapter(
            [layers[i].pooling_adapter_head for i in range(self.amount_of_clients)],
            ts,
            x,
        )
        x = self.calc_lin_bn_drop(
            [layers[i].lin_bn_drop_head_final for i in range(self.amount_of_clients)],
            ts,
            x,
        )
        return x

    def calc_pooling_adapter(self, layers, ts, x):
        avg_out = sum(
            [ts[i] * layers[i][0].ap(x) for i in range(self.amount_of_clients)]
        )
        max_out = sum(
            [ts[i] * layers[i][0].mp(x) for i in range(self.amount_of_clients)]
        )
        out = torch.cat([max_out, avg_out], 1)

        out = layers[0][1](out)
        return out

    def calc_lin_bn_drop(self, layers, ts, x):
        out = sum([ts[i] * layers[i][0](x) for i in range(self.amount_of_clients)])
        out = sum([ts[i] * layers[i][1](out) for i in range(self.amount_of_clients)])
        return out

    def forward(self, x):
        trust_scores = self.trust_scores
        x = self.calc_stem(
            [self.client_models[i].stem for i in range(self.amount_of_clients)],
            trust_scores,
            x,
        )
        x = self.calc_backbone(
            [self.client_models[i].backbone for i in range(self.amount_of_clients)],
            trust_scores,
            x,
        )
        x = self.calc_head(
            [self.client_models[i].head for i in range(self.amount_of_clients)],
            trust_scores,
            x,
        )
        return x
