# RVCBench 中文入门

[English](quickstart.md) · [文档目录](README.md)

RVCBench 提供两种用法：**给自己的音频自动打分**，或者**用我们提供的数据评测模型**。
两种方式都通过 pip 安装即可使用，不需要把你的模型加入仓库，也不需要在评分环境里安装模型。
本指南对应 **2.1.0**。

## 1. 安装

推荐 Linux、Python 3.10–3.12。下面创建 CPU 评分环境，不要求 GPU：

```bash
# Ubuntu / Debian；已安装 FFmpeg 可跳过
sudo apt-get install ffmpeg
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install "rvcbench[eval]==2.1.0"
rvcbench doctor --eval --imports
rvcbench setup-scorers
```

`[eval]` 安装评分依赖；`setup-scorers` 下载并校验评分模型，首次需要联网和数 GB 缓存空间，
之后会复用。GPU 安装、离线使用和报错处理见[安装指南](installation.md)。
以下命令都在你的工作目录执行，运行评分时保持这个环境激活。

## 2. 用法一：给自己的音频打分

准备模型生成的 `generated.wav`、目标说话人的 `speaker_reference.wav`，以及你要求生成的文本。
参考音频可以说不同的内容。把以下代码存成 `score_audio.py`，替换为你的文件名和文本：

```python
import json
from rvcbench import metrics

with metrics.Evaluator(["sim", "wer", "speechmos"], device="cpu") as evaluator:
    scores = evaluator.score(
        "generated.wav",
        reference="speaker_reference.wav",
        text="你好，欢迎使用语音评测。",
        language="zh",
    )
print(json.dumps(scores, indent=2))
```

```bash
python score_audio.py
```

返回一个包含三个分数的字典：`sim` 越高表示说话人越相似；`wer` 越低表示识别出的内容与预期越一致；
`speechmos` 越高表示预测的自然度越好。WER 是比例，`0.1` 表示 10%，也可能超过 1。
评分模型得到的是自动指标，不等同于人工听测。

需要全部七项指标时，在同样的 import 后使用：

```python
with metrics.Evaluator("all", device="cpu") as evaluator:
    scores = evaluator.score(
        "generated.wav",
        reference="speaker_reference.wav",
        target="same_text_recording.wav",
        text="你好，欢迎使用语音评测。",
        language="zh",
    )
print(scores)
```

| 指标 | 衡量什么 | 额外需要什么 |
| --- | --- | --- |
| `sim`、`sva` | 说话人相似度、身份验证是否通过 | 目标说话人的参考录音 |
| `wer` | 内容准确度 | 预期文本，可选语言提示 |
| `speechmos` | 预测的语音自然度 | 无 |
| `mcd`、`stoi` | 声学失真、可懂度 | 与生成音频说**相同文字**的目标录音 |
| `emotion` | 预测的情感类别是否一致 | 具有目标情感的参考录音 |

`reference` 和 `target` 可以是不同录音。没有同文本录音时，就选用其余适合的指标。
批量评分时，在一个 `Evaluator` 里循环调用 `score()`，避免重复加载模型。
完整示例见[指标 API](metrics.md)。

## 3. 用法二：使用我们的数据评测模型

### 导出评测输入

先用 52 条音频的入门套件跑通流程：

```bash
rvcbench prompts --suite onboarding-v1 --output prompts/
```

命令会下载所需数据，并在 `prompts/` 中提供参考录音以及三种输入格式：

- `prompts.jsonl`：供自己的 Python 推理脚本读取。
- `prompts.tsv`：供 ZipVoice 风格的批量脚本使用。
- `prompts.lst`：供 Seed-TTS 风格的批量脚本使用。

### 用自己的模型生成音频

在模型自己的环境中读取这些输入，按 `text` 合成语音，使用 `reference_audio` 克隆说话人。
每条输出必须保存为 `outputs/my-model/<id>.wav`，其中 `id` 使用导出文件里的原值。
以下是对接模板，需要把模型加载和 `synthesize` 换成你的实际 API：

```python
import json
from pathlib import Path
import soundfile as sf

# 在这里加载你的模型，命名为 model。
prompts = Path("prompts")
outputs = Path("outputs/my-model")
outputs.mkdir(parents=True, exist_ok=True)
for line in (prompts / "prompts.jsonl").read_text().splitlines():
    item = json.loads(line)
    waveform, sample_rate = model.synthesize(
        text=item["text"],
        reference_audio=str(prompts / item["reference_audio"]),
        reference_text=item["reference_text"],
        language=item["language"],
    )
    sf.write(outputs / f"{item['id']}.wav", waveform, sample_rate)
```

保存单声道 WAV，并使用实际采样率；不要把文件重命名为简单的数字序号。
模型生成这一步由你完成；下载数据、准备输入、评分和报告由 RVCBench 完成。
评分用的目标录音不会出现在导出的 prompts 中。

### 自动评分

切回 RVCBench 环境，保持与导出时相同的 suite：

```bash
rvcbench score --suite onboarding-v1 \
  --generated outputs/my-model --output results/my-model --device cpu
```

打开 `results/my-model/submission.json`，或者运行以下 Python 代码查看摘要：

```python
import json
from pathlib import Path

report = json.loads(Path("results/my-model/submission.json").read_text())
print(report["model"], report["status"])
for task, result in report["tasks"].items():
    print(task, result["status"], result.get("means", {}))
    if result["status"] != "complete":
        print(result["coverage"], result["failures"])
```

`complete` 表示要求的样本和指标全部完成；`partial` 表示有缺失或失败。
`tasks` 中包含每项任务的状态、覆盖情况和完成任务的 `means`。
更详细的逐样本错误和评分记录在各任务目录的 `run_manifest.json` 中。
不同指标分别反映不同能力，不能用一个总分替代所有维度。

修好缺失或损坏的音频后，用原命令加 `--resume` 继续：

```bash
rvcbench score --suite onboarding-v1 \
  --generated outputs/my-model --output results/my-model --device cpu --resume
```

保持 suite、模型名、输入与输出目录一致。匹配的成功评分会复用，失败或改变的样本会重新检查。

### 比较模型，扩大评测范围

用两个模型生成相同的 prompts，然后一次评分，使用新的结果目录：

```bash
rvcbench score --suite onboarding-v1 \
  --generated outputs/model-a outputs/model-b --output results/comparison --device cpu
```

查看 `results/comparison/comparison.md`，也可以读取同目录的 CSV/JSON。
每个模型的完整报告在 `results/comparison/<模型名>/submission.json`。
比较已有结果的方法见[模型评测指南](adding_a_model.md)。

跑通后可以选用 `core-v1`（480 条）或 `full-v1`（12,724 条）。需要重新导出对应 suite 的
prompts，并在评分时使用同一个 suite 名称。`core-v1` 包含保护音频任务；`full-v1` 目前不包含
这些任务。详细覆盖范围和成本见[套件说明](core_suite.md)。套件目前是预览协议，论文复现请使用
[冻结的 v1 代码](versions.md)。

从 2.0.0 升级后，生成的音频可以复用，但请用新的结果目录重新评分。
所有待比较模型应在同一评分环境运行。详见[升级说明](installation.md#upgrade-from-200)。
