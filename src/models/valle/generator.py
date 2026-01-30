"""VALL-E generator wrapper for off-the-shelf cloning."""
from __future__ import annotations

import importlib
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.models.model import BaseModel


@dataclass
class VallEGeneratorConfig:
    code_path: Path
    checkpoint_path: Path
    text_extractor: str = "espeak"
    top_k: int = -100
    temperature: float = 1.0
    output_sample_rate: int = 24000


class VallEGenerator(BaseModel):
    """Minimal VALL-E inference bridge used by the off-the-shelf adversary."""

    def __init__(
        self,
        config: VallEGeneratorConfig,
        device: torch.device,
        logger,
    ) -> None:
        super().__init__(
            model_name_or_path=str(config.checkpoint_path),
            device=device,
            logger=logger,
        )
        self.config = config

        self._imports_loaded = False
        self._AttributeDict = None
        self._get_model = None
        self._TextTokenizer = None
        self._AudioTokenizer = None
        self._tokenize_text = None
        self._tokenize_audio = None
        self._get_text_token_collater = None

        self._model = None
        self._audio_tokenizer = None
        self._text_tokenizer = None
        self._text_collater = None
        self._sample_rate = int(self.config.output_sample_rate)

        self._validate_paths()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        text: str,
        prompt_audio: Path,
        prompt_text: Optional[str],
        sample_index: int = 0,
    ) -> Tuple[np.ndarray, int]:
        """Generate audio conditioned on a prompt clip and transcript."""
        del sample_index  # Reserved for future seeding logic.
        self.ensure_model()
        assert self._model is not None
        assert self._audio_tokenizer is not None
        assert self._text_tokenizer is not None
        assert self._text_collater is not None

        prompt_path = Path(prompt_audio)
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt audio not found: {prompt_audio}")

        spoken_text = (text or "").strip()
        if not spoken_text:
            spoken_text = (prompt_text or "").strip()
        if not spoken_text:
            raise ValueError("Text prompt for VALL-E inference cannot be empty.")

        # Upstream inference concatenates the prompt transcript with the desired text.
        combined_text = spoken_text
        if prompt_text:
            combined_text = f"{prompt_text.strip()} {spoken_text}".strip()

        text_tokens, text_lens = self._text_collater(
            [self._tokenize_text(self._text_tokenizer, text=combined_text)]
        )

        enroll_lens = None
        if prompt_text:
            _, enroll_lens = self._text_collater(
                [self._tokenize_text(self._text_tokenizer, text=prompt_text.strip())]
            )
        if self.logger:
            prompt_len = int(enroll_lens.max().item()) if enroll_lens is not None else 0
            self.logger.debug(
                "[VALL-E] Tokenized text length=%s; prompt length=%s",
                int(text_lens.max().item()),
                prompt_len,
            )

        encoded_frames = self._tokenize_audio(self._audio_tokenizer, str(prompt_path))
        try:
            prompt_codes = encoded_frames[0][0].transpose(2, 1)
        except (IndexError, AttributeError) as exc:  # pragma: no cover - defensive guard
            raise RuntimeError("Unexpected audio token shape returned by VALL-E tokenizer.") from exc
        prompt_codes = prompt_codes.to(self.device)

        text_tokens = text_tokens.to(self.device)
        text_lens = text_lens.to(self.device)
        if enroll_lens is not None:
            enroll_lens = enroll_lens.to(self.device)

        with torch.no_grad():
            generated_codes = self._model.inference(
                text_tokens,
                text_lens,
                prompt_codes,
                enroll_x_lens=enroll_lens,
                top_k=int(self.config.top_k),
                temperature=float(self.config.temperature),
            )

        # Encodec decoder expects a minimum temporal length (kernel size 7); pad short outputs.
        if generated_codes.dim() == 2:
            generated_codes = generated_codes.unsqueeze(0)
        if generated_codes.dim() == 3 and generated_codes.size(-1) < 7:
            pad = 7 - generated_codes.size(-1)
            generated_codes = F.pad(generated_codes, (0, pad), mode="replicate")

        frames = generated_codes.transpose(2, 1)
        if frames.size(-1) == 0:
            frames = torch.zeros(
                frames.size(0),
                frames.size(1),
                7,
                device=frames.device,
                dtype=frames.dtype,
            )
        elif frames.size(-1) < 7:
            frames = F.pad(frames, (0, 7 - frames.size(-1)), mode="replicate")

        samples = self._audio_tokenizer.decode([(frames, None)])

        if self.logger:
            self.logger.debug(
                "[VALL-E] Generated %d frames per codebook (duration≈%.2fs)",
                frames.size(-1),
                frames.size(-1) * 0.04,  # EnCodec 24 kHz uses ~40 ms frame stride
            )

        if isinstance(samples, torch.Tensor):
            sample_tensor = samples[0]
        elif isinstance(samples, (list, tuple)):
            sample_tensor = samples[0]
        else:  # pragma: no cover - defensive contingency
            raise TypeError(f"Unexpected decode return type: {type(samples)}")
        sample_tensor = sample_tensor.detach().cpu()
        if sample_tensor.dim() == 2 and sample_tensor.size(0) == 1:
            sample_tensor = sample_tensor.squeeze(0)
        wav = sample_tensor.numpy().astype(np.float32)
        return wav, int(self._sample_rate)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate_paths(self) -> None:
        if not self.config.code_path.exists():
            raise FileNotFoundError(f"VALL-E code path not found: {self.config.code_path}")
        if not self.config.checkpoint_path.exists():
            raise FileNotFoundError(f"VALL-E checkpoint not found: {self.config.checkpoint_path}")

    def load_model(self) -> None:
        if self._model is not None:
            return

        self._ensure_imports()

        # Some upstream checkpoints were pickled on Windows; allow them to load on POSIX.
        try:
            import pathlib
            pathlib.WindowsPath = pathlib.PosixPath  # type: ignore[attr-defined]
        except Exception:
            pass

        checkpoint = torch.load(self.config.checkpoint_path, map_location=self.device)
        if "model" not in checkpoint:
            raise ValueError("VALL-E checkpoint is missing the 'model' state dict.")

        args = self._AttributeDict(checkpoint)
        text_tokens_path = self._resolve_text_tokens_path(args.text_tokens)
        model = self._get_model(args)
        target_state = model.state_dict()
        loaded_state = checkpoint["model"]

        # Filter out weights whose shapes do not line up with the current config.
        aligned_state = {}
        skipped = {}
        for key, value in loaded_state.items():
            target_value = target_state.get(key)
            if target_value is None:
                skipped[key] = f"missing in target ({value.shape})"
                continue
            if value.shape != target_value.shape:
                skipped[key] = f"{value.shape} -> {target_value.shape}"
                continue
            aligned_state[key] = value

        missing, unexpected = model.load_state_dict(aligned_state, strict=False)
        if self.logger and (missing or unexpected or skipped):
            self.logger.warning(
                "Loaded VALL-E checkpoint with partial state. Missing: %s; unexpected: %s; skipped: %s",
                missing,
                unexpected,
                skipped,
            )
        model.to(self.device)
        model.eval()

        self._model = model
        self._audio_tokenizer = self._AudioTokenizer()
        if hasattr(self._audio_tokenizer, "sample_rate"):
            self._sample_rate = int(self._audio_tokenizer.sample_rate)
        self._text_tokenizer = self._TextTokenizer(backend=self.config.text_extractor)
        self._text_collater = self._get_text_token_collater(str(text_tokens_path))
        if self.logger:
            try:
                vocab_size = len(getattr(self._text_collater, "token2idx", {}))
            except Exception:
                vocab_size = None
            self.logger.debug(
                "[VALL-E] Using text token file %s (vocab size=%s)",
                text_tokens_path,
                vocab_size,
            )

    def _resolve_text_tokens_path(self, raw_path: str) -> Path:
        """Resolve the text_tokens.k2symbols file across common roots."""
        candidate = Path(str(raw_path))
        search_paths = []
        if candidate.is_absolute():
            search_paths.append(candidate)
        else:
            try:
                from hydra.utils import to_absolute_path

                search_paths.append(Path(to_absolute_path(str(candidate))))
            except Exception:
                pass
            search_paths.extend(
                [
                    Path.cwd() / candidate,
                    self.config.code_path / candidate,
                    Path(__file__).resolve().parent.parent.parent / candidate,
                    Path(__file__).resolve().parent.parent.parent.parent / candidate,
                ]
            )

        for path in search_paths:
            if path.exists():
                return path.resolve()

        # Fall back to the original (possibly missing) path so the caller sees a clear error.
        return candidate

    def _ensure_imports(self) -> None:
        if self._imports_loaded:
            return

        os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

        code_path = Path(self.config.code_path)
        code_path_str = str(code_path)
        if code_path_str not in sys.path:
            sys.path.insert(0, code_path_str)

        # The upstream repo vendors Icefall under checkpoints/vall-e/icefall/icefall/.
        # Add the inner package root explicitly so `import icefall.*` succeeds.
        icefall_path = code_path / "icefall"
        if icefall_path.exists():
            icefall_pkg = str(icefall_path)
            if icefall_pkg not in sys.path:
                sys.path.insert(0, icefall_pkg)

        lhotse_ok = True
        try:
            import lhotse  # noqa: F401
        except Exception as exc:
            lhotse_ok = False
            if self.logger:
                self.logger.warning("lhotse import failed (%s); installing lightweight stub for VALL-E inference.", exc)
            sys.modules.pop("lhotse", None)
        if not lhotse_ok:
            self._install_lhotse_stub()

        try:
            from icefall.utils import AttributeDict
        except ImportError as exc:  # pragma: no cover - import guard
            # Fall back to a lightweight shim to avoid the heavy icefall/k2 deps when only
            # the utility helpers are needed for inference.
            if self.logger:
                self.logger.warning(
                    "Failed to import icefall (%s); using lightweight shim for VALL-E inference.",
                    exc,
                )
            AttributeDict = self._install_icefall_stub()

        valle_root = code_path / "valle"

        def _register_stub(name: str, attrs=None, is_package: bool = False):
            module = types.ModuleType(name)
            if is_package:
                module.__path__ = []
                module.__package__ = name.rsplit(".", 1)[0]
                module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
            if attrs:
                for key, value in attrs.items():
                    setattr(module, key, value)
            sys.modules[name] = module
            return module

        # Bypass valle/__init__.py (which pulls in training-only deps) by loading
        # the modules directly from their file paths.
        valle_pkg = sys.modules.get("valle")
        if valle_pkg is None or getattr(valle_pkg, "__file__", None):
            valle_pkg = types.ModuleType("valle")
            valle_pkg.__path__ = [str(valle_root)]
            sys.modules["valle"] = valle_pkg

        # Pre-populate valle.data stubs to prevent training-time imports from firing.
        data_pkg = sys.modules.get("valle.data")
        if data_pkg is None or getattr(data_pkg, "__file__", None):
            data_pkg = _register_stub("valle.data", is_package=True)

        input_stub = _register_stub(
            "valle.data.input_strategies",
            attrs={
                "PromptedFeatures": type(
                    "PromptedFeatures",
                    (),
                    {
                        "__init__": lambda self, prompts=None, features=None: None,
                        "to": lambda self, device=None: self,
                        "sum": lambda self: 0,
                        "ndim": property(lambda self: 0),
                        "data": property(lambda self: (None, None)),
                    },
                ),
                "PromptedPrecomputedFeatures": type("PromptedPrecomputedFeatures", (), {}),
            },
        )
        dataset_stub = _register_stub(
            "valle.data.dataset", attrs={"SpeechSynthesisDataset": type("SpeechSynthesisDataset", (), {})}
        )
        datamodule_stub = _register_stub(
            "valle.data.datamodule", attrs={"TtsDataModule": type("TtsDataModule", (), {})}
        )
        data_pkg.input_strategies = input_stub
        data_pkg.dataset = dataset_stub
        data_pkg.datamodule = datamodule_stub

        models_spec = importlib.util.spec_from_file_location(
            "valle.models",
            valle_root / "models" / "__init__.py",
            submodule_search_locations=[str(valle_root / "models")],
        )
        if models_spec is None or models_spec.loader is None:  # pragma: no cover - defensive guard
            raise ImportError("Unable to locate valle.models module for VALL-E.")
        models_module = importlib.util.module_from_spec(models_spec)
        sys.modules["valle.models"] = models_module
        models_spec.loader.exec_module(models_module)
        get_model = getattr(models_module, "get_model", None)
        if get_model is None:  # pragma: no cover - defensive guard
            raise ImportError("valle.models does not expose get_model; check code_path.")

        data_pkg = sys.modules.get("valle.data")
        if data_pkg is None or getattr(data_pkg, "__file__", None):
            data_pkg = types.ModuleType("valle.data")
            data_pkg.__path__ = [str(valle_root / "data")]
            sys.modules["valle.data"] = data_pkg

        tok_spec = importlib.util.spec_from_file_location(
            "valle.data.tokenizer",
            valle_root / "data" / "tokenizer.py",
            submodule_search_locations=[str(valle_root / "data")],
        )
        if tok_spec is None or tok_spec.loader is None:  # pragma: no cover
            raise ImportError("Unable to locate valle.data.tokenizer for VALL-E.")
        tok_module = importlib.util.module_from_spec(tok_spec)
        sys.modules["valle.data.tokenizer"] = tok_module
        tok_spec.loader.exec_module(tok_module)

        coll_spec = importlib.util.spec_from_file_location(
            "valle.data.collation",
            valle_root / "data" / "collation.py",
            submodule_search_locations=[str(valle_root / "data")],
        )
        if coll_spec is None or coll_spec.loader is None:  # pragma: no cover
            raise ImportError("Unable to locate valle.data.collation for VALL-E.")
        coll_module = importlib.util.module_from_spec(coll_spec)
        sys.modules["valle.data.collation"] = coll_module
        coll_spec.loader.exec_module(coll_module)

        AudioTokenizer = tok_module.AudioTokenizer
        TextTokenizer = tok_module.TextTokenizer
        tokenize_audio = tok_module.tokenize_audio
        tokenize_text = tok_module.tokenize_text
        get_text_token_collater = coll_module.get_text_token_collater

        self._AttributeDict = AttributeDict
        self._get_model = get_model
        self._TextTokenizer = TextTokenizer
        self._AudioTokenizer = AudioTokenizer
        self._tokenize_text = tokenize_text
        self._tokenize_audio = tokenize_audio
        self._get_text_token_collater = get_text_token_collater

        self._imports_loaded = True

    def _install_icefall_stub(self):
        """Inject a minimal icefall.utils shim to bypass k2 during inference."""
        import types

        if "icefall.utils" in sys.modules:
            utils_mod = sys.modules["icefall.utils"]
        else:
            utils_mod = types.ModuleType("icefall.utils")

        class AttributeDict(dict):
            """dict with attribute-style access used by upstream checkpoints."""

            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError as exc:
                    raise AttributeError(name) from exc

            def __setattr__(self, name, value):
                self[name] = value

        def str2bool(value):
            if isinstance(value, bool):
                return value
            if value is None:
                return False
            return str(value).lower() in {"true", "1", "yes", "y", "t"}

        def make_pad_mask(lengths, num_frames: Optional[int] = None):
            lengths_tensor = torch.as_tensor(lengths)
            max_frames = int(num_frames) if num_frames is not None else int(lengths_tensor.max().item())
            rng = torch.arange(max_frames, device=lengths_tensor.device)
            return rng.unsqueeze(0).expand(lengths_tensor.size(0), -1) >= lengths_tensor.unsqueeze(1)

        utils_mod.AttributeDict = AttributeDict
        utils_mod.str2bool = str2bool
        utils_mod.make_pad_mask = make_pad_mask

        icefall_mod = sys.modules.get("icefall")
        if icefall_mod is None:
            icefall_mod = types.ModuleType("icefall")
            icefall_mod.__path__ = []
            sys.modules["icefall"] = icefall_mod
        icefall_mod.utils = utils_mod
        sys.modules["icefall.utils"] = utils_mod

        return AttributeDict

    def _install_lhotse_stub(self):
        """Provide the minimal lhotse API required for inference-only imports."""
        import types

        if "lhotse" in sys.modules:
            lhotse_mod = sys.modules["lhotse"]
        else:
            lhotse_mod = types.ModuleType("lhotse")
            lhotse_mod.__path__ = []
            sys.modules["lhotse"] = lhotse_mod

        utils_mod = types.ModuleType("lhotse.utils")
        Seconds = float

        def compute_num_frames(duration, frame_shift, sampling_rate=None):
            try:
                dur = float(duration)
                shift = float(frame_shift)
                return max(0, int(round(dur / shift)))
            except Exception:
                return 0

        utils_mod.Seconds = Seconds
        utils_mod.compute_num_frames = compute_num_frames
        utils_mod.fix_random_seed = lambda seed=0: None
        sys.modules["lhotse.utils"] = utils_mod
        lhotse_mod.utils = utils_mod

        features_mod = types.ModuleType("lhotse.features")

        class FeatureExtractor:
            name = "stub"
            config_type = object

            def __init__(self, config=None):
                self.config = config or self.config_type

            def to_dict(self):
                cfg = getattr(self, "config", None)
                return cfg.to_dict() if hasattr(cfg, "to_dict") else {}

        features_mod.FeatureExtractor = FeatureExtractor
        sys.modules["lhotse.features"] = features_mod
        lhotse_mod.features = features_mod

        dataset_mod = types.ModuleType("lhotse.dataset")
        dataset_mod.__path__ = []
        dataset_mod.__package__ = "lhotse"

        class _DatasetStub:
            def __init__(self, *args, **kwargs):
                pass

            def __call__(self, *args, **kwargs):
                return self

        dataset_mod.CutConcatenate = _DatasetStub
        dataset_mod.DynamicBucketingSampler = _DatasetStub
        dataset_mod.PrecomputedFeatures = _DatasetStub
        dataset_mod.SimpleCutSampler = _DatasetStub
        dataset_mod.SpecAugment = _DatasetStub
        sys.modules["lhotse.dataset"] = dataset_mod
        lhotse_mod.dataset = dataset_mod

        input_mod = types.ModuleType("lhotse.dataset.input_strategies")
        input_mod.__package__ = "lhotse.dataset"

        class OnTheFlyFeatures(_DatasetStub):
            pass

        class PrecomputedFeatures(_DatasetStub):
            pass

        input_mod.ExecutorType = object
        input_mod.PrecomputedFeatures = PrecomputedFeatures
        input_mod._get_executor = lambda *args, **kwargs: None
        input_mod.OnTheFlyFeatures = OnTheFlyFeatures
        sys.modules["lhotse.dataset.input_strategies"] = input_mod
        dataset_mod.input_strategies = input_mod
        dataset_mod.PrecomputedFeatures = PrecomputedFeatures

        collation_mod = types.ModuleType("lhotse.dataset.collation")
        collation_mod.collate_features = lambda *args, **kwargs: (None, None)
        sys.modules["lhotse.dataset.collation"] = collation_mod
        dataset_mod.collation = collation_mod

        sampling_mod = types.ModuleType("lhotse.dataset.sampling")
        sys.modules["lhotse.dataset.sampling"] = sampling_mod
        dataset_mod.sampling = sampling_mod

        def _cutset_stub(*args, **kwargs):
            return []

        lhotse_mod.CutSet = _DatasetStub
        lhotse_mod.load_manifest_lazy = _cutset_stub
        lhotse_mod.validate = lambda *args, **kwargs: None
        lhotse_mod.fix_random_seed = lambda seed=0: None

        return FeatureExtractor
