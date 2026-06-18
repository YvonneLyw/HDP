from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _flatten_batch(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1)


class ScalarEmbedding(nn.Module):
    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.embed_dim = int(embed_dim)

    def forward(self, scalar: torch.Tensor) -> torch.Tensor:
        scalar = scalar.reshape(-1).float()
        half_dim = max(self.embed_dim // 2, 1)
        freq = torch.exp(
            -torch.log(torch.tensor(10000.0, device=scalar.device))
            * torch.arange(half_dim, device=scalar.device, dtype=scalar.dtype)
            / max(half_dim - 1, 1)
        )
        emb = scalar[:, None] * freq[None]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if emb.shape[-1] < self.embed_dim:
            emb = F.pad(emb, (0, self.embed_dim - emb.shape[-1]))
        return emb[:, :self.embed_dim]


class DenoisingMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        hidden_dim: int = 256,
        time_embed_dim: int = 32,
    ):
        super().__init__()
        self.time_embed = ScalarEmbedding(time_embed_dim)
        total_dim = int(input_dim) + int(cond_dim) + int(time_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, int(input_dim)),
        )

    def forward(self, noisy_x: torch.Tensor, cond: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(sigma)
        return self.net(torch.cat((noisy_x, cond, t_emb), dim=-1))


class LatentValueNet(nn.Module):
    """
    Standalone value network for DOSER-style branch arbitration.

    It intentionally lives outside Critic2net. The intended first-version use is:
      - dynamics predicts a latent successor z'
      - state support detector evaluates the same z'
      - this value net evaluates the same z'

    First-version training happens in pretrain_doser_selector_components.py with
    a frozen critic target. Later versions can move this into the critic/value
    stage, but it should not be trained in the actor BC stage.
    """
    def __init__(self, latent_dim: int, subgoal_dim: int = 0, hidden_dim: int = 256):
        super().__init__()
        input_dim = int(latent_dim) + int(subgoal_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.subgoal_dim = int(subgoal_dim)

    def forward(self, latent: torch.Tensor, subgoal: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.subgoal_dim > 0:
            if subgoal is None:
                raise RuntimeError("LatentValueNet requires subgoal but got None.")
            x = torch.cat((latent, subgoal), dim=-1)
        else:
            x = latent
        return self.net(x).squeeze(-1)

    @staticmethod
    def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
        weight = torch.where(diff > 0, expectile, 1.0 - expectile)
        return (weight * diff.pow(2)).mean()


class EmpiricalPercentileMixin:
    def _init_reference_errors(self) -> None:
        self.register_buffer("reference_errors", torch.empty(0))

    def has_reference_errors(self) -> bool:
        return self.reference_errors.numel() > 0

    @torch.no_grad()
    def set_reference_errors(self, errors: torch.Tensor) -> None:
        ref = torch.as_tensor(errors, dtype=torch.float32).reshape(-1)
        if ref.numel() == 0:
            self.reference_errors = torch.empty(0, device=self.reference_errors.device)
            return
        self.reference_errors = torch.sort(ref.to(self.reference_errors.device)).values

    @torch.no_grad()
    def percentile(self, errors: torch.Tensor) -> torch.Tensor:
        if not self.has_reference_errors():
            raise RuntimeError("Detector has no reference_errors. Run calibration first.")
        ref = self.reference_errors.to(device=errors.device, dtype=errors.dtype)
        flat = errors.reshape(-1)
        ranks = torch.searchsorted(ref, flat, right=True).to(dtype=errors.dtype)
        return (ranks / float(ref.numel())).reshape_as(errors)


class CurrentImageEncoder(nn.Module):
    def __init__(
        self,
        image_shape: Tuple[int, int, int, int],
        output_dim: int = 64,
    ):
        super().__init__()
        views, channels, _, _ = image_shape
        self.image_shape = tuple(int(x) for x in image_shape)
        self.output_dim = int(output_dim)
        self.net = nn.Sequential(
            nn.Conv2d(int(views) * int(channels), 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, self.output_dim),
            nn.ReLU(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() == 6:
            current_images = images[:, 0]
        elif images.dim() == 5:
            current_images = images
        else:
            raise RuntimeError(f"images must be (B,2,2,C,H,W) or (B,2,C,H,W), got {tuple(images.shape)}")
        if tuple(current_images.shape[1:]) != self.image_shape:
            raise RuntimeError(
                f"current image shape mismatch: expected {self.image_shape}, "
                f"got {tuple(current_images.shape[1:])}"
            )
        B, views, channels, height, width = current_images.shape
        x = current_images.reshape(B, views * channels, height, width)
        return self.net(x)


class FullStateActionDetector(nn.Module, EmpiricalPercentileMixin):
    """
    Shared full-state action support detector.

    score(common, action_eval) returns:
      - error: denoised reconstruction MSE on action_eval
      - percentile: empirical rank against offline calibration errors

    It does not use branch_condition_encoder / DKO. A and B candidate actions
    are scored by the same detector under the same raw/current observation.
    """
    def __init__(
        self,
        state_dim: int,
        subgoal_dim: int,
        qpos_dim: int,
        action_eval_dim: int,
        image_shape: Optional[Tuple[int, int, int, int]] = None,
        image_feat_dim: int = 64,
        hidden_dim: int = 256,
        time_embed_dim: int = 32,
        min_sigma: float = 0.01,
        max_sigma: float = 1.0,
        score_samples: int = 4,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.subgoal_dim = int(subgoal_dim)
        self.qpos_dim = int(qpos_dim)
        self.action_eval_dim = int(action_eval_dim)
        self.image_shape = tuple(image_shape) if image_shape is not None else None
        self.image_feat_dim = int(image_feat_dim) if self.image_shape is not None else 0
        self.min_sigma = float(min_sigma)
        self.max_sigma = float(max_sigma)
        self.score_samples = int(score_samples)
        self.image_encoder = (
            CurrentImageEncoder(self.image_shape, self.image_feat_dim)
            if self.image_shape is not None and self.image_feat_dim > 0
            else None
        )
        self.denoiser = DenoisingMLP(
            input_dim=self.action_eval_dim,
            cond_dim=self.state_dim + self.subgoal_dim + self.qpos_dim + self.image_feat_dim,
            hidden_dim=hidden_dim,
            time_embed_dim=time_embed_dim,
        )
        self._init_reference_errors()

    def _obs_feature(self, common: Dict[str, Optional[torch.Tensor]], like: torch.Tensor) -> torch.Tensor:
        parts = []
        B = like.shape[0]
        for key, dim in (("state", self.state_dim), ("subgoal", self.subgoal_dim)):
            if dim <= 0:
                continue
            value = common.get(key, None)
            if value is None:
                value = torch.zeros((B, dim), device=like.device, dtype=like.dtype)
            parts.append(value.reshape(B, -1))

        if self.qpos_dim > 0:
            qpos_pair = common.get("qpos_pair", None)
            if qpos_pair is None:
                raise RuntimeError("FullStateActionDetector was trained with qpos, but common['qpos_pair'] is missing.")
            qpos = qpos_pair[:, 0] if qpos_pair.dim() == 3 else qpos_pair
            parts.append(qpos.reshape(B, -1).to(device=like.device, dtype=like.dtype))

        if self.image_encoder is not None:
            images = common.get("image_pair", common.get("doser_image_pair", None))
            if images is None:
                raise RuntimeError(
                    "FullStateActionDetector was trained with images, but common['image_pair'] "
                    "or common['doser_image_pair'] is missing."
                )
            image_feat = self.image_encoder(images.to(device=like.device, dtype=like.dtype))
            parts.append(image_feat)

        return torch.cat(parts, dim=-1) if parts else like.new_zeros((B, 0))

    def _sample_sigma(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        sigma = torch.rand(batch_size, device=device, dtype=dtype)
        return sigma * (self.max_sigma - self.min_sigma) + self.min_sigma

    def denoising_loss(self, common: Dict[str, Optional[torch.Tensor]], action_eval: torch.Tensor) -> torch.Tensor:
        x = _flatten_batch(action_eval)
        cond_feat = self._obs_feature(common, x)
        sigma = self._sample_sigma(x.shape[0], x.device, x.dtype)
        noise = torch.randn_like(x)
        noisy_x = x + sigma[:, None] * noise
        pred_noise = self.denoiser(noisy_x, cond_feat, sigma)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def reconstruction_error(
        self,
        common: Dict[str, Optional[torch.Tensor]],
        action_eval: torch.Tensor,
        num_samples: Optional[int] = None,
    ) -> torch.Tensor:
        x = _flatten_batch(action_eval)
        cond_feat = self._obs_feature(common, x)
        n = self.score_samples if num_samples is None else int(num_samples)
        errors = []
        for _ in range(max(n, 1)):
            sigma = self._sample_sigma(x.shape[0], x.device, x.dtype)
            noise = torch.randn_like(x)
            noisy_x = x + sigma[:, None] * noise
            pred_noise = self.denoiser(noisy_x, cond_feat, sigma)
            denoised_x = noisy_x - sigma[:, None] * pred_noise
            errors.append(F.mse_loss(denoised_x, x, reduction="none").mean(dim=-1))
        return torch.stack(errors, dim=0).mean(dim=0)

    @torch.no_grad()
    def score(self, common: Dict[str, Optional[torch.Tensor]], action_eval: torch.Tensor) -> Dict[str, torch.Tensor]:
        error = self.reconstruction_error(common, action_eval)
        out = {"error": error}
        if self.has_reference_errors():
            out["percentile"] = self.percentile(error)
        return out


class LatentDynamicsModel(nn.Module):
    """
    Latent dynamics over normalized state/subgoal plus optional current images.

    The decoder is used only during pretraining to keep the latent from
    collapsing. Rollout only calls forward(...)->next_latent.
    """
    def __init__(
        self,
        state_dim: int,
        subgoal_dim: int,
        action_eval_dim: int,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        image_shape: Optional[Tuple[int, int, int, int]] = None,
        image_feat_dim: int = 64,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.subgoal_dim = int(subgoal_dim)
        self.action_eval_dim = int(action_eval_dim)
        self.latent_dim = int(latent_dim)
        self.image_shape = tuple(image_shape) if image_shape is not None else None
        self.image_feat_dim = int(image_feat_dim) if self.image_shape is not None else 0
        self.image_encoder = (
            CurrentImageEncoder(self.image_shape, self.image_feat_dim)
            if self.image_shape is not None and self.image_feat_dim > 0
            else None
        )
        obs_dim = self.state_dim + self.subgoal_dim + self.image_feat_dim
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, self.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, obs_dim),
        )
        self.transition = nn.Sequential(
            nn.Linear(self.latent_dim + self.action_eval_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, self.latent_dim),
        )

    def _obs_feature(
        self,
        state: torch.Tensor,
        subgoal: Optional[torch.Tensor],
        image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        state = state.reshape(state.shape[0], -1)
        parts = [state]
        if subgoal is None:
            subgoal = torch.zeros((state.shape[0], self.subgoal_dim), device=state.device, dtype=state.dtype)
        if self.subgoal_dim > 0:
            parts.append(subgoal.reshape(state.shape[0], -1).to(device=state.device, dtype=state.dtype))
        if self.image_encoder is not None:
            if image is None:
                raise RuntimeError("LatentDynamicsModel was trained with images, but image is missing.")
            image_feat = self.image_encoder(image.to(device=state.device, dtype=state.dtype))
            parts.append(image_feat)
        return torch.cat(parts, dim=-1)

    def encode(
        self,
        state: Optional[torch.Tensor] = None,
        subgoal: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        common: Optional[Dict[str, Optional[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        if common is not None:
            state = common["state"]
            subgoal = common.get("subgoal", None)
            image = common.get("doser_image_pair", common.get("image_pair", None))
        if state is None:
            raise RuntimeError("LatentDynamicsModel.encode requires state or common.")
        return self.encoder(self._obs_feature(state, subgoal, image))

    def reconstruction_loss(
        self,
        state: torch.Tensor,
        subgoal: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        obs = self._obs_feature(state, subgoal, image)
        recon = self.decoder(self.encoder(obs))
        return F.mse_loss(recon, obs)

    def forward(self, common: Dict[str, Optional[torch.Tensor]], action_eval: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encode(common=common)
        action_flat = _flatten_batch(action_eval)
        next_latent = self.transition(torch.cat((z, action_flat), dim=-1))
        uncertainty = torch.zeros(next_latent.shape[0], device=next_latent.device, dtype=next_latent.dtype)
        return {"next_latent": next_latent, "uncertainty": uncertainty}


class LatentStateDetector(nn.Module, EmpiricalPercentileMixin):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 256,
        time_embed_dim: int = 32,
        min_sigma: float = 0.01,
        max_sigma: float = 1.0,
        score_samples: int = 4,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.min_sigma = float(min_sigma)
        self.max_sigma = float(max_sigma)
        self.score_samples = int(score_samples)
        self.denoiser = DenoisingMLP(
            input_dim=self.latent_dim,
            cond_dim=0,
            hidden_dim=hidden_dim,
            time_embed_dim=time_embed_dim,
        )
        self._init_reference_errors()

    def _sample_sigma(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        sigma = torch.rand(batch_size, device=device, dtype=dtype)
        return sigma * (self.max_sigma - self.min_sigma) + self.min_sigma

    def denoising_loss(self, latent: torch.Tensor) -> torch.Tensor:
        x = _flatten_batch(latent)
        sigma = self._sample_sigma(x.shape[0], x.device, x.dtype)
        noise = torch.randn_like(x)
        noisy_x = x + sigma[:, None] * noise
        cond = x.new_zeros((x.shape[0], 0))
        pred_noise = self.denoiser(noisy_x, cond, sigma)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def reconstruction_error(self, latent: torch.Tensor, num_samples: Optional[int] = None) -> torch.Tensor:
        x = _flatten_batch(latent)
        n = self.score_samples if num_samples is None else int(num_samples)
        cond = x.new_zeros((x.shape[0], 0))
        errors = []
        for _ in range(max(n, 1)):
            sigma = self._sample_sigma(x.shape[0], x.device, x.dtype)
            noise = torch.randn_like(x)
            noisy_x = x + sigma[:, None] * noise
            pred_noise = self.denoiser(noisy_x, cond, sigma)
            denoised_x = noisy_x - sigma[:, None] * pred_noise
            errors.append(F.mse_loss(denoised_x, x, reduction="none").mean(dim=-1))
        return torch.stack(errors, dim=0).mean(dim=0)

    @torch.no_grad()
    def score(self, latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        error = self.reconstruction_error(latent)
        out = {"error": error}
        if self.has_reference_errors():
            out["percentile"] = self.percentile(error)
        return out
