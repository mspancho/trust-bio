"""Wrappers for the evaluated foundation models.

Each wrapper reproduces the exact load + extraction recipe from the original
SignalMC-MED `extract-features_<model>.py` scripts (verified against that repo):

  MOMENT       MOMENTPipeline(x_enc=segs).embeddings;  segs [B,1,2500] @250Hz
  Chronos-Bolt embed(segs) -> hidden[:, :-1, :].mean(1); @250Hz, dim 512
  D-BETA       get_ecg_feats(model, 12-lead with signal in lead idx 1) @500Hz
               (local repo); HF AutoModel.pooler_output fallback
  ECGFounder   ft_1lead_ECGFounder(...).return_features; (_, feats)=model(segs)
               segs [B,1,5000] @500Hz, dim 1024
  xECG         xECG.from_pretrained(...); signal in [B,T,12] lead idx 1;
               feats,_=model(signals) @100Hz, dim 1024
  PaPaGei      ResNet1DMoE; model(segs)[0]; segs [B,1,1250] @125Hz, dim 512
  CSFM         CSFM(...); model(segs, channel=[1]/[12]/[1,12]) @250Hz, dim 768

The external model repos (D-BETA, ECGFounder, xecg, papagei-foundation-model,
Cardiac-Sensing-FM) are vendored under a directory pointed to by
SIGNALMCMED_MODEL_REPOS (default: <repo>/model_repos). Each wrapper lazily
imports its dependency in load(); a load failure becomes a clean skip upstream
(pipeline.extract_features_for_model) rather than a crash. Pass
allow_fallback=True to substitute a deterministic stand-in for plumbing.
"""
from __future__ import annotations

import contextlib
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

from ..config import FMSpec
from .base import FeatureExtractor

# Where the external model source repos are vendored.
MODEL_REPOS_DIR = Path(
    os.environ.get(
        "SIGNALMCMED_MODEL_REPOS",
        Path(__file__).resolve().parent.parent.parent / "model_repos",
    )
)
DEFAULT_BATCH_SIZE = 32


@contextlib.contextmanager
def _torch_load_weights_only_false():
    """Temporarily restore torch.load's pre-2.6 `weights_only=False` default.

    PyTorch 2.6 changed the default to True, which raises
    `UnpicklingError: Weights only load failed` on checkpoints that pickle
    anything beyond plain tensors. Some vendored model repos call torch.load
    internally, and we mirror those repos rather than patching them, so scope
    the old behaviour to just the wrapped load.

    Only use this around checkpoints WE vendored (model_weights/), never around
    arbitrary downloaded input -- weights_only=True exists to stop a malicious
    checkpoint executing code at load time.
    """
    import torch

    original = torch.load

    def _patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = _patched
    try:
        yield
    finally:
        torch.load = original


#: Generic top-level package names that MORE THAN ONE vendored repo defines.
#: Whichever repo loads first wins and poisons the others, because a cached
#: sys.modules entry beats any later sys.path change.
_SHADOWED_TOP_LEVEL = ("models", "utils", "modules", "src", "data", "config")


def _add_repo_to_path(name: str) -> Path:
    """Prepend a vendored model repo to sys.path; raise if absent.

    Also evicts generically-named top-level modules that a PREVIOUSLY loaded
    repo may have already imported. Several vendored repos ship a package
    literally called `models`, so in a process that loads more than one of them
    the first import wins and every later repo silently gets the wrong package:

        _add_repo_to_path("D-BETA")            # imports D-BETA's `models`
        _add_repo_to_path("papagei-...")       # sys.path updated, but...
        from models.resnet import ResNet1DMoE  # -> ModuleNotFoundError

    sys.path cannot fix that: `models` is already in sys.modules, so Python
    never re-searches. This is exactly how papagei failed in a full run while
    passing in isolation -- and because _FMBase turns a failed load into a
    SILENT random-projection fallback, it would have shown up as plausible
    numbers rather than an error.
    """
    repo = MODEL_REPOS_DIR / name
    if not repo.exists():
        raise RuntimeError(
            f"model repo {name!r} not found under {MODEL_REPOS_DIR}. Clone it "
            f"there (or set SIGNALMCMED_MODEL_REPOS) — see README 'Vendoring'."
        )
    if str(repo) in sys.path:
        sys.path.remove(str(repo))
    sys.path.insert(0, str(repo))

    # Drop any cached module that resolved out of a DIFFERENT vendored repo, so
    # the import machinery re-resolves it against the repo we just prepended.
    for mod_name in list(sys.modules):
        top = mod_name.split(".", 1)[0]
        if top not in _SHADOWED_TOP_LEVEL:
            continue
        origin = getattr(sys.modules[mod_name], "__file__", None) or ""
        paths = list(getattr(sys.modules[mod_name], "__path__", []) or [])
        locations = [origin, *paths]
        if any(str(MODEL_REPOS_DIR) in loc and str(repo) not in loc
               for loc in locations if loc):
            del sys.modules[mod_name]
    return repo


class _RandomProjectionFallback:
    """Deterministic fixed random projection used only for plumbing/tests.

    Projects per-segment summary statistics through a frozen, name-seeded matrix
    to `feature_dim`. NOT a substitute for the real FM.
    """

    def __init__(self, spec: FMSpec):
        seed = int(hashlib.sha256(spec.name.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        self._n_stats = 16
        self._proj = rng.standard_normal((self._n_stats, spec.feature_dim)).astype(np.float32)

    def __call__(self, segments: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(segments).astype(np.float32)
        feats = np.stack(
            [
                x.mean(1), x.std(1), x.min(1), x.max(1),
                np.median(x, 1), np.percentile(x, 25, 1), np.percentile(x, 75, 1),
                np.abs(np.fft.rfft(x, axis=1)).mean(1),
                np.abs(np.fft.rfft(x, axis=1)).std(1),
                np.diff(x, axis=1).std(1),
                (x > 0).mean(1), np.square(x).mean(1),
                np.abs(x).mean(1), np.percentile(x, 5, 1),
                np.percentile(x, 95, 1),
                (np.diff(np.sign(np.diff(x, 1)), 1) != 0).mean(1),
            ],
            axis=1,
        ).astype(np.float32)
        return feats @ self._proj


class _FMBase(FeatureExtractor):
    """Shared loading + batched encoding logic with optional fallback."""

    def __init__(self, spec: FMSpec, device: str = "cpu", allow_fallback: bool = False,
                 checkpoint: str | None = None, batch_size: int = DEFAULT_BATCH_SIZE,
                 force_fallback: bool = False):
        super().__init__(spec, device)
        self.allow_fallback = allow_fallback or force_fallback
        self.force_fallback = force_fallback
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self._model = None
        self._fallback = None

    def load(self):  # noqa: D401 — subclasses override; this handles force.
        if self.force_fallback:
            return self._use_fallback("force_fallback")
        return self._load_real()

    def _load_real(self):
        raise NotImplementedError

    def _use_fallback(self, reason: str):
        if not self.allow_fallback:
            raise RuntimeError(
                f"{self.spec.name}: {reason}. Provide weights / vendor the model "
                f"repo, or pass allow_fallback=True for a deterministic stand-in."
            )
        self._fallback = _RandomProjectionFallback(self.spec)
        return self

    def encode_segments(self, segments: np.ndarray, modality: str) -> np.ndarray:
        if self._fallback is not None:
            return self._fallback(segments)
        return self._encode_real(segments, modality)

    def _encode_real(self, segments: np.ndarray, modality: str) -> np.ndarray:
        raise NotImplementedError

    # Helper: iterate segments in batches as torch tensors on device.
    def _batched(self, segments: np.ndarray):
        import torch
        n = len(segments)
        for i in range(0, n, self.batch_size):
            chunk = segments[i: i + self.batch_size]
            yield torch.as_tensor(chunk, dtype=torch.float32, device=self.device)


class MomentExtractor(_FMBase):
    """MOMENT (general TS FM) @250 Hz. MOMENTPipeline default embedding."""

    _HF = {
        "moment-small": "AutonLab/MOMENT-1-small",
        "moment-base": "AutonLab/MOMENT-1-base",
        "moment-large": "AutonLab/MOMENT-1-large",
    }

    def _load_real(self):
        try:
            from momentfm import MOMENTPipeline
        except Exception as e:
            return self._use_fallback(f"momentfm not importable ({e})")
        self._model = MOMENTPipeline.from_pretrained(
            self._HF[self.spec.name], model_kwargs={"task_name": "embedding"}
        )
        self._model.init()
        self._model.to(self.device).eval()
        return self

    def _encode_real(self, segments: np.ndarray, modality: str) -> np.ndarray:
        import torch
        out = []
        with torch.no_grad():
            for x in self._batched(segments):       # x: (B, seg_len)
                x = x.unsqueeze(1)                    # (B, 1, seg_len)
                emb = self._model(x_enc=x).embeddings.cpu().numpy()
                out.append(emb)
        return np.concatenate(out, axis=0).astype(np.float32)


class ChronosBoltExtractor(_FMBase):
    """Chronos-Bolt (general TS FM) @250 Hz. embed() then mean over patch
    tokens, dropping the final (EOS) token (original: hidden[:, :-1, :].mean(1))."""

    _HF = {
        "chronos-bolt-tiny": "amazon/chronos-bolt-tiny",
        "chronos-bolt-mini": "amazon/chronos-bolt-mini",
        "chronos-bolt-small": "amazon/chronos-bolt-small",
        "chronos-bolt-base": "amazon/chronos-bolt-base",
    }

    def _load_real(self):
        try:
            from chronos import ChronosBoltPipeline
        except Exception as e:
            return self._use_fallback(f"chronos not importable ({e})")
        self._model = ChronosBoltPipeline.from_pretrained(
            self._HF[self.spec.name], device_map=self.device
        )
        return self

    def _encode_real(self, segments: np.ndarray, modality: str) -> np.ndarray:
        import torch
        out = []
        for x in self._batched(segments):            # (B, seg_len)
            hidden, _ = self._model.embed(x)         # (B, n_patch+1, d)
            feats = hidden[:, :-1, :].mean(axis=1)   # drop EOS, mean patches
            out.append(feats.cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)


class DBetaExtractor(_FMBase):
    """D-BETA (ECG FM) @500 Hz. Signal placed in lead index 1 of a 12-lead zero
    tensor. Prefers the vendored D-BETA repo (get_ecg_feats); falls back to the
    HuggingFace AutoModel whose `pooler_output` is the 768-d feature."""

    HF_REPO = "Manhph2211/D-BETA"

    def _load_real(self):
        try:
            import torch  # noqa: F401
        except Exception as e:
            return self._use_fallback(f"torch not importable ({e})")
        # Prefer the local repo (matches the paper exactly).
        try:
            _add_repo_to_path("D-BETA")
            from models.processor import get_model, get_ecg_feats
            cfg = os.environ.get("SIGNALMCMED_DBETA_CONFIG")
            self._get_ecg_feats = get_ecg_feats
            self._model = get_model(config_path=cfg, checkpoint_path=self.checkpoint)
            self._model.eval(); self._model.to(self.device)
            self._mode = "repo"
            return self
        except Exception as repo_err:
            self._repo_err = repo_err
        # Fall back to HuggingFace AutoModel.
        try:
            from transformers import AutoModel
            self._model = AutoModel.from_pretrained(self.HF_REPO, trust_remote_code=True)
            self._model.to(self.device).eval()
            self._mode = "hf"
            return self
        except Exception as hf_err:
            return self._use_fallback(
                f"D-BETA repo unavailable ({self._repo_err}); HF load failed ({hf_err})"
            )

    def _encode_real(self, segments: np.ndarray, modality: str) -> np.ndarray:
        import torch
        out = []
        with torch.no_grad():
            for x in self._batched(segments):            # (B, seg_len)
                signals = torch.zeros((x.shape[0], 12, x.shape[1]),
                                      dtype=torch.float32, device=self.device)
                signals[:, 1, :] = x
                if self._mode == "repo":
                    feats = self._get_ecg_feats(self._model, signals)
                else:
                    output = self._model(signals)
                    feats = getattr(output, "pooler_output", output)
                out.append(np.asarray(feats.cpu()))
        return np.concatenate(out, axis=0).astype(np.float32)


class ECGFounderExtractor(_FMBase):
    """ECGFounder (ECG FM) @500 Hz. Single-lead model, return_features=True,
    (_, feats) = model(segs); segs [B,1,5000], dim 1024."""

    def _load_real(self):
        try:
            _add_repo_to_path("ECGFounder")
            from finetune_model import ft_1lead_ECGFounder
        except Exception as e:
            return self._use_fallback(f"ECGFounder repo not importable ({e})")
        # PyTorch 2.6 flipped torch.load's `weights_only` default to True, which
        # breaks this checkpoint with "UnpicklingError: Weights only load
        # failed". The torch.load call lives inside the vendored third-party
        # repo (finetune_model.py), which we mirror rather than patch, so restore
        # the old default around just this call. Safe here: the checkpoint is one
        # we vendored ourselves (model_weights/1_lead_ECGFounder.pth), not
        # arbitrary remote input.
        with _torch_load_weights_only_false():
            self._model = ft_1lead_ECGFounder(
                self.device, self.checkpoint, 1, linear_prob=False
            )
        self._model.return_features = True
        self._model.eval()
        return self

    def _encode_real(self, segments: np.ndarray, modality: str) -> np.ndarray:
        import torch
        out = []
        with torch.no_grad():
            for x in self._batched(segments):            # (B, seg_len)
                x = x.unsqueeze(1)                        # (B, 1, seg_len)
                _, feats = self._model(x)
                out.append(feats.cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)


class XECGExtractor(_FMBase):
    """xECG (ECG FM, xLSTM) @100 Hz. Signal in [B, T, 12] at lead index 1;
    feats, _ = model(signals); dim 1024. The long-input variant feeds the
    entire window as a single sequence."""

    HF_REPO = "riccardolunelli/xECG_base_model_v1"

    def _load_real(self):
        try:
            _add_repo_to_path("xecg")
            from xecg.xECG import xECG
        except Exception as e:
            return self._use_fallback(f"xecg repo not importable ({e})")
        # from_pretrained accepts an HF repo id or a local DIRECTORY -- never a
        # single weights file. config.resolve_checkpoint() hands us the file
        # (model_weights/xECG_base_model_v1.safetensors), which made
        # from_pretrained treat the absolute path as a repo id and raise
        # HFValidationError. Point it at the containing directory when the
        # resolved checkpoint is a file; otherwise fall back to the HF repo.
        src = self.checkpoint or self.HF_REPO
        ckpt = Path(src) if src else None
        if ckpt is not None and ckpt.is_file():
            src = str(ckpt.parent)
        try:
            self._model = xECG.from_pretrained(src)
        except Exception as e:
            if src != self.HF_REPO:
                # A local dir without the config.json from_pretrained expects
                # still fails; the gated-free HF repo is the reliable source.
                self._model = xECG.from_pretrained(self.HF_REPO)
            else:
                raise
        self._model.to(self.device).eval()
        return self

    def _encode_real(self, segments: np.ndarray, modality: str) -> np.ndarray:
        import torch
        out = []
        with torch.no_grad():
            for x in self._batched(segments):            # (B, seg_len)
                signals = torch.zeros((x.shape[0], x.shape[1], 12),
                                      dtype=torch.float32, device=self.device)
                signals[:, :, 1] = x
                feats, _ = self._model(signals)
                out.append(feats.cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    def encode_long(self, signal: np.ndarray, modality: str) -> np.ndarray:
        if self._fallback is not None:
            return self._fallback(signal[None, :])
        return self._encode_real(signal[None, :], modality)


class PaPaGeiExtractor(_FMBase):
    """PaPaGei-S (PPG FM) @125 Hz. ResNet1DMoE, model(segs)[0]; dim 512."""

    _CONFIG = dict(base_filters=32, kernel_size=3, stride=2, groups=1,
                   n_block=18, n_classes=512, n_experts=3)

    def _load_real(self):
        try:
            _add_repo_to_path("papagei-foundation-model")
            from linearprobing.utils import load_model_without_module_prefix
            from models.resnet import ResNet1DMoE
        except Exception as e:
            return self._use_fallback(f"papagei repo not importable ({e})")
        model = ResNet1DMoE(
            in_channels=1,
            base_filters=self._CONFIG["base_filters"],
            kernel_size=self._CONFIG["kernel_size"],
            stride=self._CONFIG["stride"],
            groups=self._CONFIG["groups"],
            n_block=self._CONFIG["n_block"],
            n_classes=self._CONFIG["n_classes"],
            n_experts=self._CONFIG["n_experts"],
        )
        self._model = load_model_without_module_prefix(model, self.checkpoint)
        self._model.to(self.device).eval()
        return self

    def _encode_real(self, segments: np.ndarray, modality: str) -> np.ndarray:
        import torch
        out = []
        with torch.no_grad():
            for x in self._batched(segments):            # (B, seg_len)
                x = x.unsqueeze(1)                        # (B, 1, seg_len)
                outputs = self._model(x)
                feats = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                out.append(feats.cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)


class CSFMExtractor(_FMBase):
    """CSFM (multimodal ECG+PPG FM) @250 Hz. model(segs, channel) with
    channel=[1] (ECG), [12] (PPG), [1,12] (joint). Primary results use late
    fusion of the unimodal vectors; encode_joint exposes the internal fusion."""

    _CHANNEL = {"ecg": [1], "ppg": [12]}
    _SIGNAL_SIZE = 2500   # 10 s @ 250 Hz

    def _load_real(self):
        try:
            import torch
            _add_repo_to_path("Cardiac-Sensing-FM")
            from network.model import CSFM
        except Exception as e:
            return self._use_fallback(f"CSFM (Cardiac-Sensing-FM) not importable ({e})")
        if not self.checkpoint:
            return self._use_fallback("no CSFM checkpoint (restricted access)")
        dim = self.spec.feature_dim
        self._model = CSFM(
            signal_size=self._SIGNAL_SIZE, patch_size=25, num_classes=1,
            channels=13, dim=dim, depth=12, heads=12, mlp_dim=4 * dim,
            dropout=0.1, emb_dropout=0.1, text_len=64, pool="cls",
        )
        ckpt = torch.load(self.checkpoint, map_location=self.device)
        enc = {k.replace("encoder.", ""): v for k, v in ckpt.items()
               if k.startswith("encoder.") and "mlp_head" not in k}
        self._model.load_state_dict(enc, strict=False)
        self._model.to(self.device).eval()
        return self

    def _encode_real(self, segments: np.ndarray, modality: str) -> np.ndarray:
        import torch
        channel = np.asarray(self._CHANNEL[modality])
        out = []
        with torch.no_grad():
            for x in self._batched(segments):            # (B, seg_len)
                x = x.unsqueeze(1)                        # (B, 1, seg_len)
                feats = self._model(x, channel)
                out.append(feats.cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)
