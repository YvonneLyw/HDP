import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.init as init


def _get_activation(activation_type: str) -> nn.Module:
    if activation_type == "sigmoid":
        return nn.Sigmoid()
    if activation_type == "tanh":
        return nn.Tanh()
    if activation_type == "ReLU":
        return nn.ReLU()
    if activation_type == "PReLU":
        return nn.PReLU()
    if activation_type == "softmax":
        return nn.Softmax(dim=-1)
    if activation_type == "Mish":
        return nn.Mish()
    return nn.PReLU()


class TwoLayerPreActivationResNetLinear(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 100,
        activation: str = "ReLU",
        dropout_rate: float = 0.0,
        use_spectral_norm: bool = False,
        use_norm: bool = False,
        norm_style: str = "BatchNorm",
    ) -> None:
        super().__init__()
        linear = spectral_norm if use_spectral_norm else (lambda layer: layer)
        self.l1 = linear(nn.Linear(hidden_dim, hidden_dim))
        self.l2 = linear(nn.Linear(hidden_dim, hidden_dim))
        self.dropout = nn.Dropout(dropout_rate)
        self.use_norm = use_norm
        self.act = _get_activation(activation)

        if use_norm:
            if norm_style == "BatchNorm":
                self.normalizer = nn.BatchNorm1d(hidden_dim)
            elif norm_style == "LayerNorm":
                self.normalizer = nn.LayerNorm(hidden_dim, eps=1e-6)
            else:
                raise ValueError(f"Unsupported norm_style: {norm_style}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_input = x
        if self.use_norm:
            x = self.normalizer(x)
        x = self.l1(self.dropout(self.act(x)))
        if self.use_norm:
            x = self.normalizer(x)
        x = self.l2(self.dropout(self.act(x)))
        return x + x_input


class ResidualMLPNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 100,
        num_hidden_layers: int = 4,
        output_dim: int = 1,
        dropout: float = 0.0,
        activation: str = "ReLU",
        use_spectral_norm: bool = False,
        use_norm: bool = False,
        norm_style: str = "BatchNorm",
    ):
        super().__init__()
        if num_hidden_layers % 2 != 0:
            raise ValueError("num_hidden_layers must be even for residual blocks.")

        linear = spectral_norm if use_spectral_norm else (lambda layer: layer)
        layers = [linear(nn.Linear(input_dim, hidden_dim))]
        layers.extend(
            [
                TwoLayerPreActivationResNetLinear(
                    hidden_dim=hidden_dim,
                    activation=activation,
                    dropout_rate=dropout,
                    use_spectral_norm=use_spectral_norm,
                    use_norm=use_norm,
                    norm_style=norm_style,
                )
                for _ in range(1, num_hidden_layers, 2)
            ]
        )
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class DeepKoopmanModule(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        latent_act_dim: int,
        hidden_dim: int = 128,
        num_hidden_layers: int = 4,
        dropout: float = 0.0,
        activation: str = "ReLU",
        use_spectral_norm: bool = False,
        use_norm: bool = False,
        norm_style: str = "BatchNorm",
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.latent_act_dim = int(latent_act_dim)

        self.K = nn.Linear(self.obs_dim, self.obs_dim, bias=False)
        self.V = nn.Linear(self.latent_act_dim, self.obs_dim, bias=False)
        self._initialize_weights(self.K)
        self._initialize_weights(self.V)

        ##obs feature 预测 latent action
        self.latent_policy = ResidualMLPNetwork(
            input_dim=self.obs_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            output_dim=self.latent_act_dim,
            dropout=dropout,
            activation=activation,
            use_spectral_norm=use_spectral_norm,
            use_norm=use_norm,
            norm_style=norm_style,
        )

    def _initialize_weights(self, module: nn.Linear):
        init.xavier_uniform_(module.weight)
        if module.bias is not None:
            init.zeros_(module.bias)

    def enable_kv_grad(self, enable_flag: bool):    ##（逐 weight 设置 requires_grad）
        self.K.requires_grad_(enable_flag)
        self.V.requires_grad_(enable_flag)

    def forward(self, current_obs: torch.Tensor, latent_act: torch.Tensor = None):
        ## 原始 Koopman consistency：学K,V,latent_policy,也允许 encoder 从latent-policy 路径训练
        if latent_act is None:
            latent_act = self.get_latent_act(current_obs)
            self.enable_kv_grad(True)
            ## 不让 K 把“原始 current feature”那条输入分支直接拉着 encoder 走，但允许 encoder 通过“生成 latent action”这件事被训练
            next_obs = self.K(current_obs.detach() + self.V(latent_act))
            return next_obs, latent_act
        ##增强一致性，希望图像增强后的 encoder 输出 与之前的动力学一致，只想更新 obs_encoder
        self.enable_kv_grad(False)
        next_obs = self.K(current_obs + self.V(latent_act.detach()))    ##只更新 obs_encoder，不更新KV
        return next_obs

    def get_latent_act(self, enc_obs: torch.Tensor) -> torch.Tensor:
        return self.latent_policy(enc_obs)

    def get_latnet_act(self, enc_obs: torch.Tensor) -> torch.Tensor:
        return self.get_latent_act(enc_obs)
