#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import importlib
import logging
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import hydra
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


LOGGER = logging.getLogger("report_model_sizes")

VC_MODEL_CONFIGS = {
    "bertvits2": "ots_vc/clean/vctk/bert_ots",
    "cosyvoice": "ots_vc/clean/vctk/cosyvoice_ots",
    "fishspeech": "ots_vc/clean/vctk/fishspeech_ots",
    "glm_tts": "ots_vc/clean/vctk/glmtts_ots",
    "glowtts": "ots_vc/clean/vctk/glowtts_ots",
    "higgs_audio": "ots_vc/clean/vctk/higgs_audio_ots",
    "kimi_audio": "ots_vc/clean/vctk/kimi_audio_ots",
    "mgm_omni": "ots_vc/clean/vctk/mgm_omni_ots",
    "moss_ttsd": "ots_vc/clean/vctk/moss_ttsd_ots",
    "ozspeech": "ots_vc/clean/vctk/ozspeech_ots",
    "playdiffusion": "ots_vc/clean/vctk/playdiffusion_ots",
    "qwen3_omni": "ots_vc/clean/vctk/qwen3_omni_ots",
    "sparktts": "ots_vc/clean/vctk/sparktts_ots",
    "styletts2": "ots_vc/clean/vctk/styletts2_ots",
    "vall_e": "ots_vc/clean/vctk/vall_e_ots",
    "vibevoice": "ots_vc/clean/vctk/vibevoice_ots",
}

PROTECTION_CONFIGS = {
    "EMProtector": "em_on_libritts",
    "EnkiduProtector": "enkidu_on_libritts",
    "GRNoiseProtector": "grnoise_on_libritts",
    "SafeSpeechProtector": "safespeech_on_libritts",
}

SHARED_MODEL_CONFIGS = {
    "BertVits2": "model/BertVits2",
    "OZSpeech": "model/OZSpeech",
    "SpeakerRecognition": "model/SpeakerRecognition",
}


@dataclass
class ModelResult:
    category: str
    name: str
    config_name: str
    status: str
    parameter_count: Optional[int] = None
    module_count: int = 0
    module_paths: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    error: Optional[str] = None
    reason: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load benchmark models and report parameter counts in Markdown."
    )
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument(
        "--output",
        default="docs/model_sizes.md",
        help="Markdown output path. Defaults to docs/model_sizes.md.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device to use while loading models. Defaults to cpu.",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=["vc", "protection", "shared_model"],
        choices=["vc", "protection", "shared_model"],
        help="Categories to include.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional subset of model names to run.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full tracebacks in the Markdown report for failures.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def compose_config(repo_root: Path, config_name: str) -> DictConfig:
    config_dir = str((repo_root / "configs").resolve())
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        conf = compose(config_name=config_name)
    return conf


def import_symbol(path: str):
    module_path, symbol_name = path.split(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, symbol_name)


def build_category_plan() -> Dict[str, Dict[str, str]]:
    return {
        "vc": VC_MODEL_CONFIGS,
        "protection": PROTECTION_CONFIGS,
        "shared_model": SHARED_MODEL_CONFIGS,
    }


def resolve_selected_entries(
    categories: Sequence[str],
    selected_models: Optional[Sequence[str]],
) -> List[Tuple[str, str, str]]:
    selected = set(selected_models or [])
    entries: List[Tuple[str, str, str]] = []
    for category in categories:
        model_map = build_category_plan()[category]
        for name, config_name in model_map.items():
            if selected and name not in selected:
                continue
            entries.append((category, name, config_name))
    return entries


def maybe_release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def count_parameters_from_modules(modules: Sequence[Tuple[str, torch.nn.Module]]) -> Tuple[int, List[str]]:
    seen_parameters: set[int] = set()
    total = 0
    module_paths: List[str] = []
    for path, module in modules:
        module_paths.append(path)
        for param in module.parameters(recurse=True):
            ident = id(param)
            if ident in seen_parameters:
                continue
            seen_parameters.add(ident)
            total += int(param.numel())
    return total, module_paths


def collect_modules(root: Any, max_nodes: int = 50000) -> List[Tuple[str, torch.nn.Module]]:
    modules: Dict[int, Tuple[str, torch.nn.Module]] = {}
    seen_objects: set[int] = set()
    stack: List[Tuple[str, Any]] = [("root", root)]

    while stack and len(seen_objects) < max_nodes:
        path, current = stack.pop()
        current_id = id(current)
        if current_id in seen_objects:
            continue
        seen_objects.add(current_id)

        if isinstance(current, torch.nn.Module):
            modules.setdefault(current_id, (path, current))

        if isinstance(current, dict):
            for key, value in current.items():
                stack.append((f"{path}[{key!r}]", value))
            continue

        if isinstance(current, (list, tuple, set)):
            for index, value in enumerate(current):
                stack.append((f"{path}[{index}]", value))
            continue

        if isinstance(current, (str, bytes, int, float, bool, Path, type(None))):
            continue

        current_dict = getattr(current, "__dict__", None)
        if isinstance(current_dict, dict):
            for key, value in current_dict.items():
                stack.append((f"{path}.{key}", value))

    return sorted(modules.values(), key=lambda item: item[0])


def materialize_adversary_for_counting(adversary: Any) -> Any:
    if hasattr(adversary, "_ensure_model"):
        adversary._ensure_model()
    if hasattr(adversary, "_ensure_imports"):
        adversary._ensure_imports()
    if hasattr(adversary, "_ensure_synthesizer"):
        adversary._ensure_synthesizer()
        synthesizer = getattr(adversary, "_synthesizer", None)
        if synthesizer is not None and hasattr(synthesizer, "ensure_model"):
            synthesizer.ensure_model()
        return synthesizer or adversary
    if hasattr(adversary, "_ensure_generator"):
        adversary._ensure_generator()
        generator = getattr(adversary, "_generator", None)
        if generator is not None and hasattr(generator, "ensure_model"):
            generator.ensure_model()
        return generator or adversary
    if hasattr(adversary, "ensure_model"):
        adversary.ensure_model()
    return adversary


def load_vc_model(repo_root: Path, model_name: str, config_name: str, device: torch.device, logger) -> Any:
    conf = compose_config(repo_root, config_name)
    from rvcbench.workflows.vc import _ADVERSARY_REGISTRY

    target = _ADVERSARY_REGISTRY["ots"][str(conf.vc.model).lower()]
    cls = import_symbol(target)
    adversary = cls(conf, conf.dataset, device, logger)
    return materialize_adversary_for_counting(adversary)


def count_module_from_path(module: torch.nn.Module) -> Tuple[int, List[str]]:
    total = sum(int(param.numel()) for param in module.parameters())
    return total, ["root"]


def load_shared_model(repo_root: Path, model_name: str, config_name: str, device: torch.device, logger) -> Any:
    conf = compose_config(repo_root, config_name)
    model_conf = conf.model
    if model_name == "BertVits2":
        from rvcbench.models.bertvits2_model import BertVITS2Model, BertVITS2Config
        from rvcbench.models.bertvits2_model import (
            DurationDiscriminatorConfig,
            MultiPeriodDiscriminatorConfig,
            SynthesizerTrnConfig,
            WavLMDiscriminatorConfig,
        )

        net_g_conf = OmegaConf.to_container(model_conf.net_g, resolve=True)
        net_d_conf = OmegaConf.to_container(model_conf.net_d, resolve=True)
        net_dur_disc_conf = OmegaConf.to_container(model_conf.net_dur_disc, resolve=True)
        net_wd_conf = OmegaConf.to_container(model_conf.net_wd, resolve=True)

        net_g = SynthesizerTrnConfig(
            n_vocab=int(net_g_conf["n_vocab"] or 178),
            spec_channels=int(net_g_conf["spec_channels"] or 513),
            segment_size=int(net_g_conf["segment_size"] or 32),
            inter_channels=int(net_g_conf["inter_channels"]),
            hidden_channels=int(net_g_conf["hidden_channels"]),
            filter_channels=int(net_g_conf["filter_channels"]),
            n_heads=int(net_g_conf["n_heads"]),
            n_layers=int(net_g_conf["n_layers"]),
            kernel_size=int(net_g_conf["kernel_size"]),
            p_dropout=float(net_g_conf["p_dropout"]),
            resblock=str(net_g_conf["resblock"]),
            resblock_kernel_sizes=list(net_g_conf["resblock_kernel_sizes"]),
            resblock_dilation_sizes=list(net_g_conf["resblock_dilation_sizes"]),
            upsample_rates=list(net_g_conf["upsample_rates"]),
            upsample_initial_channel=int(net_g_conf["upsample_initial_channel"]),
            upsample_kernel_sizes=list(net_g_conf["upsample_kernel_sizes"]),
            n_speakers=int(net_g_conf["n_speakers"] or max(int(model_conf.num_speakers), 1)),
            gin_channels=int(net_g_conf["gin_channels"]),
            use_sdp=bool(net_g_conf["use_sdp"]),
            n_flow_layer=int(net_g_conf["n_flow_layer"]),
            n_layers_trans_flow=int(net_g_conf["n_layers_trans_flow"]),
            flow_share_parameter=bool(net_g_conf["flow_share_parameter"]),
            use_transformer_flow=bool(net_g_conf["use_transformer_flow"]),
            use_spk_conditioned_encoder=bool(net_g_conf["use_spk_conditioned_encoder"]),
            use_noise_scaled_mas=bool(net_g_conf["use_noise_scaled_mas"]),
            mas_noise_scale_initial=float(net_g_conf["mas_noise_scale_initial"]),
            noise_scale_delta=float(net_g_conf["noise_scale_delta"]),
            current_mas_noise_scale=float(net_g_conf["current_mas_noise_scale"]),
        )
        config = BertVITS2Config(
            net_g_config=net_g,
            net_d_config=MultiPeriodDiscriminatorConfig(
                use_spectral_norm=bool(net_d_conf["use_spectral_norm"])
            ),
            net_dur_disc_config=DurationDiscriminatorConfig(
                in_channels=int(net_dur_disc_conf["in_channels"]),
                filter_channels=int(net_dur_disc_conf["filter_channels"]),
                kernel_size=int(net_dur_disc_conf["kernel_size"]),
                p_dropout=float(net_dur_disc_conf["p_dropout"]),
                gin_channels=int(net_dur_disc_conf["gin_channels"]),
            ),
            net_wd_config=WavLMDiscriminatorConfig(
                model_name_or_path=str(net_wd_conf["model_name_or_path"]),
                model_sr=24000,
                slm_sr=int(net_wd_conf["slm_sr"]),
                hidden=int(net_wd_conf["hidden"]),
                nlayers=int(net_wd_conf["nlayers"]),
                initial_channel=int(net_wd_conf["initial_channel"]),
                use_spectral_norm=bool(net_wd_conf["use_spectral_norm"]),
            ),
        )
        model = BertVITS2Model(config)
        model.to(device)
        return model

    if model_name == "OZSpeech":
        from rvcbench.models.ozspeech import OzSpeechSynthesizer

        synth = OzSpeechSynthesizer(
            cfg_path=Path(str(model_conf.config_path)),
            checkpoint_path=Path(str(model_conf.checkpoint_path)),
            device=device,
            temperature=float(getattr(model_conf, "temperature", 0.01)),
            logger=logger,
            code_path=Path(str(model_conf.code_path)),
        )
        synth.ensure_model()
        return synth

    if model_name == "SpeakerRecognition":
        from speechbrain.inference.speaker import EncoderClassifier

        classifier = EncoderClassifier.from_hparams(
            source=str(model_conf.source),
            savedir=str(model_conf.checkpoint_path),
            run_opts={"device": str(device)},
        )
        return classifier

    raise ValueError(f"Unsupported shared model '{model_name}'")


def load_protection_model(repo_root: Path, model_name: str, config_name: str, device: torch.device, logger) -> Any:
    conf = compose_config(repo_root, config_name)

    if model_name == "SafeSpeechProtector":
        from rvcbench.models.bertvits2_model import BertVits2Wrapper

        wrapper = BertVits2Wrapper(conf.protection, conf.model, conf.dataset, logger)
        wrapper.ensure_model()
        return wrapper

    if model_name == "EnkiduProtector":
        from speechbrain.inference.speaker import EncoderClassifier

        classifier = EncoderClassifier.from_hparams(
            source=str(conf.model.source),
            savedir=str(conf.model.checkpoint_path),
            run_opts={"device": str(device)},
        )
        return classifier

    if model_name in {"AttackVCProtector", "EMProtector", "GRNoiseProtector"}:
        return None

    raise ValueError(f"Unsupported protection model '{model_name}'")


def render_error(exc: BaseException, verbose: bool) -> str:
    if verbose:
        return "".join(traceback.format_exception(exc)).strip()
    return f"{type(exc).__name__}: {exc}"


def classify_exception(exc: BaseException) -> Tuple[str, str]:
    message = str(exc)
    if isinstance(exc, FileNotFoundError):
        return "skipped", "missing local assets"
    if type(exc).__name__ == "MissingConfigException":
        return "skipped", "missing Hydra config fragment"
    if isinstance(exc, ModuleNotFoundError):
        return "skipped", "missing module or incompatible environment"
    if isinstance(exc, ImportError):
        return "skipped", "missing import or incompatible environment"
    if isinstance(exc, OSError) and "path_or_model_id" in message:
        return "skipped", "missing model path or hub id not available locally"
    return "error", "runtime error"


def run_one(
    repo_root: Path,
    category: str,
    model_name: str,
    config_name: str,
    device: torch.device,
    verbose: bool,
) -> ModelResult:
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)

    try:
        loaded = None
        if category == "vc":
            loaded = load_vc_model(repo_root, model_name, config_name, device, logger)
        elif category == "shared_model":
            loaded = load_shared_model(repo_root, model_name, config_name, device, logger)
        elif category == "protection":
            loaded = load_protection_model(repo_root, model_name, config_name, device, logger)
        else:
            raise ValueError(f"Unsupported category '{category}'")

        if loaded is None:
            return ModelResult(
                category=category,
                name=model_name,
                config_name=config_name,
                status="skipped",
                notes=["No loader has been implemented for this model category yet."],
            )

        modules = collect_modules(loaded)
        parameter_count, module_paths = count_parameters_from_modules(modules)
        return ModelResult(
            category=category,
            name=model_name,
            config_name=config_name,
            status="ok",
            parameter_count=parameter_count,
            module_count=len(modules),
            module_paths=module_paths,
        )
    except Exception as exc:
        status, reason = classify_exception(exc)
        return ModelResult(
            category=category,
            name=model_name,
            config_name=config_name,
            status=status,
            error=render_error(exc, verbose),
            reason=reason,
        )
    finally:
        maybe_release_memory()


def format_int(value: Optional[int]) -> str:
    if value is None:
        return "N/A"
    return f"{value:,}"


def format_short_params(value: Optional[int]) -> str:
    if value is None:
        return "N/A"
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return str(value)


def render_markdown(repo_root: Path, results: Sequence[ModelResult], device: torch.device) -> str:
    grouped: Dict[str, List[ModelResult]] = defaultdict(list)
    for result in results:
        grouped[result.category].append(result)

    lines = [
        "# Model Parameter Report",
        "",
        f"Generated from `{repo_root}` using device `{device}`.",
        "",
        "Parameter counts are computed by loading one representative config per model family and recursively summing unique `torch.nn.Parameter` objects found in the loaded runtime object graph.",
        "",
    ]

    for category in ("vc", "protection", "shared_model"):
        bucket = grouped.get(category)
        if not bucket:
            continue
        lines.append(f"## {category.replace('_', ' ').title()}")
        lines.append("")
        lines.append("| Model | Config | Status | Parameters | Modules |")
        lines.append("| --- | --- | --- | ---: | ---: |")
        for result in sorted(bucket, key=lambda item: item.name.lower()):
            lines.append(
                f"| `{result.name}` | `{result.config_name}` | {result.status} | {format_short_params(result.parameter_count)} | {result.module_count} |"
            )
        lines.append("")

        for result in sorted(bucket, key=lambda item: item.name.lower()):
            lines.append(f"### `{result.name}`")
            lines.append("")
            lines.append(f"- Config: `{result.config_name}`")
            lines.append(f"- Status: `{result.status}`")
            if result.reason:
                lines.append(f"- Reason: {result.reason}")
            if result.parameter_count is not None:
                lines.append(f"- Parameters: `{format_int(result.parameter_count)}`")
                lines.append(f"- Modules counted: `{result.module_count}`")
            if result.notes:
                for note in result.notes:
                    lines.append(f"- Note: {note}")
            if result.error:
                lines.append("- Error:")
                lines.append("```text")
                lines.append(result.error)
                lines.append("```")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    setup_logging()

    repo_root = Path(args.repo_root).expanduser().resolve()
    output_path = (repo_root / args.output).resolve()
    device = torch.device(args.device)

    results: List[ModelResult] = []
    for category, model_name, config_name in resolve_selected_entries(args.categories, args.models):
        LOGGER.info("Counting parameters for %s/%s using %s", category, model_name, config_name)
        results.append(run_one(repo_root, category, model_name, config_name, device, args.verbose))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown(repo_root, results, device), encoding="utf-8")
    print(f"Wrote {output_path.relative_to(repo_root)} with {len(results)} entries.")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    main()
