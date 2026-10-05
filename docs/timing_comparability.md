---
title: "Compare voice cloning runtime measurements"
description: "Review timing scopes, request boundaries and provenance checks needed to compare voice cloning latency in RVCBench without mixing incompatible measurements."
---

# Timing evidence audit

Audited base: `e5cfbf5e43b523e7e79aac3b79e42f47bc33cf2c`.

| Stage | Evidence and limitation | Guardrail |
| --- | --- | --- |
| GenerationResult | Legacy measures the whole adapter call, potentially including lazy initialization. Adapter timing records can replace that value with a narrower, unaudited measurement. | Keep selected elapsed seconds and original scope. Also retain the enclosing legacy adapter call as `adapter_call_time_sec`. Generic adapter scope remains generic. |
| Qwen3 | `generate_sample` encodes reference prompt features before starting `perf_counter`. Timer surrounds generator.generate, ends before array validation and WAV write. | Keep `qwen3_generate_excluding_prompt_encoding_and_io_v1`. |
| F5 | Timer starts after transcript resolution, surrounds generator.generate, ends before array validation and WAV write. Generator uses its own reference preprocessing and sequential inference. | Keep `f5_sequential_inference_excluding_transcription_and_output_write_v1`. This is not the Qwen boundary. |
| Runner / resume | Manifest already retains selected seconds and scope. Timing CSV previously lost scope. | CSV now retains scope and enclosing adapter call. Resume copies original measurement rather than re-timing copied audio. |
| Modern aggregation | Ratio of summed selected seconds to summed full generated WAV durations per scope. Previous single-scope test did not prove comparability. | Preserve per-scope totals and raw single-scope RTF. Explicit `timing_comparability.ranking_allowed=false`. Missing, invalid, zero-duration or mixed records cannot produce a pooled RTF. |
| Legacy evaluation | CSV reader previously discarded scope. Run-level elapsed / audio duration could include a different boundary. | Retain exact CSV scope, aggregate using the same guardrail, retain old run ratio as `raw_run_rtf`, without authorizing comparison. Old CSVs stay unspecified. |
| Historical comparison | Exact pairing supports MCD/WER/SIM regression, not timing equivalence. Historical replay deliberately has no measured timing. | Add a separate timing verdict with null RTF delta. Never infer historical scope from the current backend or transfer quality authorization to timing. |
| README / site | Static historical values, without audited scope or execution evidence. README rank is SIM, but RTF had a best-value highlight. Website allowed RTF sorting. | Keep numerical records, label raw/incomparable, remove RTF best highlight and website sorting control/data. Regenerate published HTML and llms.txt. |

## Decision rule

All adapter-reported scopes fail closed for horizontal speed ranking, including identical generic scope strings and identical Qwen/F5 scope strings. `synthesis_time_sec` has no approved common measurement protocol. A scope name is provenance, not proof of equal boundaries.

The separate `request_wall_time_sec` measurement is checked pairwise by `rvcbench compare-timing` (see the [run guide](run_protocol.md#compare-request-timing)). It compares host request latency under matched recorded settings and does not lift the refusal above for adapter-reported times or published RTF values.

To authorize a future group, audit actual timed operations (reference preprocessing, model load, warm-up/cache state, inference/decoding, device synchronization, output conversion and I/O), then record and match hardware, device/runtime and execution settings, exact paired workload, timing coverage and denominator. Implement and test that measured protocol before permitting rankings. Equal labels alone must never open the gate.

The two existing plan statuses remain pending. These changes establish preservation and refusal rules, not new comparable measurements. No model was loaded and no new runtime benchmark was measured.

## Verification

`python -m pytest -q tests/test_timing.py` exercises weighted aggregation, missing and invalid records, mixed and unknown scopes, equal legacy labels, historical missing scope, actual CSV export, and published ranking controls without model loading. Site generation is checked for deterministic output.
