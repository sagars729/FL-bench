import os
from copy import deepcopy
from collections import OrderedDict
from typing import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from fedavg import FedAvgServer
from src.config.args import get_pfedla_argparser
from src.config.utils import TEMP_DIR, trainable_params


class pFedLAServer(FedAvgServer):
    def __init__(self):
        super(pFedLAServer, self).__init__(
            "pFedLA", get_pfedla_argparser().parse_args(), unique_model=True
        )
        self.hypernet = HyperNetwork(
            embedding_dim=self.args.embedding_dim,
            client_num=self.client_num_in_total,
            hidden_dim=self.args.hidden_dim,
            backbone=self.model,
            K=self.args.k,
            gpu=self.args.server_cuda,
        )
        self.hn_optimizer = torch.optim.SGD(
            self.hypernet.parameters(),
            lr=self.args.hn_lr,
            momentum=self.args.hn_momentum,
        )
        self.test_flag = False
        self.layers_name = [
            name
            for name, module in self.model.named_modules()
            if isinstance(module, nn.Conv2d)
            or isinstance(module, nn.Linear)
            or isinstance(module, nn.BatchNorm2d)
        ]

    def train(self) -> None:
        for E in self.train_progress_bar:
            self.current_epoch = E
            if (E + 1) % self.args.verbose_gap == 0:
                self.logger.log(" " * 30, f"TRAINING EPOCH: {E + 1}", " " * 30)
            if (E + 1) % self.args.test_gap == 0:
                self.test_flag = True
                self.test()
                self.test_flag = False
            self.selected_clients = self.client_sample_stream[E]
            for client_id in self.selected_clients:
                client_local_params = self.generate_client_params(client_id)
                delta, _, self.clients_metrics[client_id][E] = self.trainer.train(
                    client_id=client_id,
                    new_parameters=client_local_params,
                    verbose=((E + 1) % self.args.verbose_gap) == 0,
                )

                self.update_hn(client_id, delta)
                self.update_client_params(client_id, delta)

            self.log_info()

    @torch.no_grad()
    def update_client_params(self, client_id, delta):
        new_params = []
        for param, diff in zip(
            self.client_trainable_params[client_id], trainable_params(delta)
        ):
            new_params.append((param - diff.to(self.device)).detach())
        self.client_trainable_params[client_id] = new_params

    def generate_client_params(self, client_id: int) -> OrderedDict[str, torch.Tensor]:
        layer_params_dict = dict(
            zip(self.trainable_params_name, list(zip(*self.client_trainable_params)))
        )

        if self.test_flag:
            with torch.no_grad():
                alpha = self.hypernet(client_id)
        else:
            alpha = self.hypernet(client_id)

        aggregated_params = OrderedDict(
            zip(self.trainable_params_name, self.client_trainable_params[client_id])
        )
        default_weight = torch.zeros(
            self.client_num_in_total, dtype=torch.float, device=self.device
        )
        default_weight[client_id] = 1.0

        for name, params in layer_params_dict.items():
            a = alpha[".".join(name.split(".")[:-1])]
            if a.sum() == 0:
                a = default_weight
            aggregated_params[name] = torch.sum(
                (a / a.sum()) * torch.stack(params, dim=-1).to(self.device), dim=-1
            )
        self.client_trainable_params[client_id] = list(aggregated_params.values())
        return aggregated_params

    def update_hn(self, client_id: int, delta: OrderedDict[str, torch.Tensor]) -> None:
        # calculate gradients
        self.hn_optimizer.zero_grad()
        hn_grads = torch.autograd.grad(
            outputs=self.client_trainable_params[client_id],
            inputs=self.hypernet.parameters(),
            grad_outputs=list(
                map(lambda diff: (-diff).clone().detach(), list(delta.values()))
            ),
            allow_unused=True,
        )
        for param, grad in zip(self.hypernet.parameters(), hn_grads):
            if grad is not None:
                param.grad = grad
        self.hn_optimizer.step()
        self.hypernet.save_hn()

    def run(self):
        super().run()
        self.hypernet.clean()  # clean out all HNs


class HyperNetwork(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        client_num: int,
        hidden_dim: int,
        backbone: nn.Module,
        K: int,
        gpu=True,
    ):
        super(HyperNetwork, self).__init__()
        self.device = torch.device(
            "cuda" if gpu and torch.cuda.is_available() else "cpu"
        )
        self.K = K
        self.client_num = client_num
        self.embedding = nn.Embedding(client_num, embedding_dim, device=self.device)
        # for tracking the current client's hn parameters
        self.client_id: int = None
        self.cache_dir = TEMP_DIR / "pfedla_hn_weight"
        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        def uniform_init_linear(in_features, out_features):
            linear = nn.Linear(in_features, out_features)
            torch.nn.init.uniform_(linear.weight)
            torch.nn.init.zeros_(linear.bias)
            return linear

        self.layers_name = [
            name
            for name, module in backbone.named_modules()
            if isinstance(module, nn.Conv2d)
            or isinstance(module, nn.Linear)
            or isinstance(module, nn.BatchNorm2d)
        ]
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        ).to(self.device)
        self.fc_layers = nn.ParameterList(
            [
                uniform_init_linear(hidden_dim, client_num).to(self.device)
                for _ in range(len(self.layers_name))
            ]
        )
        if os.listdir(self.cache_dir) == []:
            for client_id in range(client_num):
                torch.save(
                    {
                        "mlp": deepcopy(self.mlp.state_dict()),
                        "fc": deepcopy(self.fc_layers.state_dict()),
                    },
                    self.cache_dir / f"{client_id}.pt",
                )

    def forward(self, client_id: int):
        self.client_id = client_id
        emd = self.embedding(
            torch.tensor(client_id, dtype=torch.long, device=self.device)
        )
        self.load_hn()
        feature = self.mlp(emd)
        alpha = [F.relu(fc(feature)) for fc in self.fc_layers]

        if self.K > 0:  # HeurpFedLA
            default_weight = torch.zeros(
                self.client_num, dtype=torch.float, device=self.device
            )
            default_weight[client_id] = 1.0

            self_weights = torch.zeros(len(alpha), device=self.device)
            for i, weight in enumerate(alpha):
                self_weights[i] = weight[client_id].data

            topk_weights_idx = torch.topk(self_weights, self.K, sorted=False)[1]

            for i in topk_weights_idx:
                alpha[i] = (alpha[i] * default_weight).detach().requires_grad_(True)

        return {layer: a for layer, a in zip(self.layers_name, alpha)}

    def save_hn(self):
        torch.save(
            {
                "mlp": deepcopy(self.mlp.state_dict()),
                "fc": deepcopy(self.fc_layers.state_dict()),
            },
            self.cache_dir / f"{self.client_id}.pt",
        )
        self.client_id = None

    def load_hn(self):
        weights = torch.load(self.cache_dir / f"{self.client_id}.pt")
        self.mlp.load_state_dict(weights["mlp"])
        self.fc_layers.load_state_dict(weights["fc"])

    def clean(self):
        if os.path.isdir(self.cache_dir):
            os.system(f"rm -rf {self.cache_dir}")


if __name__ == "__main__":
    server = pFedLAServer()
    server.run()
