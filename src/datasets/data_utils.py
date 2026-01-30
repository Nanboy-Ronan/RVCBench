import json
import os
import re
import random
from dataclasses import fields, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

import pandas as pd
import torch
import torch.utils.data
import torchaudio
from hydra.utils import to_absolute_path
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.utils.commons as commons
from src.datasets.mel_preprocessing import spectrogram_torch, mel_spectrogram_torch
from src.models.text import cleaned_text_to_sequence
from src.utils.commons import load_wav_to_torch


class TextAudioSpeakerDataset(torch.utils.data.Dataset):
    """
    1) loads audio, speaker_id, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """

    def __init__(self, data_conf, sid, sidx, sampling_rate, logger=None):
        self.logger = logger
        if self.logger:
            self.logger.debug(f"TextAudioSpeakerDataset __init__: data_conf type: {type(data_conf)}")
            self.logger.debug(f"TextAudioSpeakerDataset __init__: data_conf content: {data_conf}")
        self.sid = sid
        self.sidx = sidx
        if sampling_rate is not None:
            self.sampling_rate = int(sampling_rate)
        self.filter_length = data_conf.filter_length
        self.win_length = data_conf.win_length
        self.hop_length = int(getattr(data_conf, "hop_length", 512))
        self.data_conf = data_conf
        self.logger = logger
        
        self.max_wav_value = float(data_conf.max_wav_value)

        json_file_path = os.path.join(
            data_conf.root_path,
            "filelists",
            f"{sid}.json"
        )
        self.data_root = Path(data_conf.root_path)

        try:
            self.audiopaths_sid_text = pd.read_json(json_file_path)
            if not isinstance(self.audiopaths_sid_text, pd.DataFrame):
                raise TypeError(f"Expected pandas DataFrame after reading JSON, but got {type(self.audiopaths_sid_text)}")
            if self.logger:
                self.logger.debug("Successfully loaded JSON into DataFrame.")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading JSON file {json_file_path}: {e}", exc_info=True)
            raise

        self._spk_mem = {}
        # spectrogram type
        self.use_mel_spec_posterior = getattr(data_conf, "use_mel_posterior_encoder", False)
        if self.use_mel_spec_posterior:
            self.n_mel_channels = getattr(data_conf, "n_mel_channels", 80)

        self.cleaned_text = getattr(data_conf, "cleaned_text", False)
        self.add_blank = data_conf.add_blank
        self.min_text_len = getattr(data_conf, "min_text_len", 1)
        self.max_text_len = getattr(data_conf, "max_text_len", 640)
        # self.audiopaths_sid_text = self.audiopaths_sid_text.sample(frac=1).reset_index(drop=True)
        self.skipped=0
        self.logger.info("Init dataset for Speaker {}...".format(self.sid))
        self.logger.info("Found {} audios from the index file...".format(len(self.audiopaths_sid_text)))

        if self.logger:
            self.logger.debug(f"Before _filter: audiopaths_sid_text type: {type(self.audiopaths_sid_text)}")
            self.logger.debug(f"Before _filter: audiopaths_sid_text head:\n{getattr(self.audiopaths_sid_text, 'head', lambda: 'N/A')()}")

        self._filter()

    # -------- Speaker parsing (no file persistence) --------
    def _canon_spk(self, s: str) -> str:
        """Normalize speaker token: strip, drop path/ext, keep part before '_' or '-', lowercase."""
        s = str(s).strip()
        s = s.split("/")[-1]
        s = s.split(".")[0]
        s = re.split(r"[_\-]", s)[0]
        return s.lower()

    def _sid_tensor(self, sid):
        """
        Convert various sid forms to tensor[int].
        Prefer trailing digits (e.g., 'p225'->225, 'spk001'->1, '5339'->5339).
        If no digits, assign a temporary in-memory id (consistent within the process).
        """
        tok = self._canon_spk(sid)
        m = re.search(r"(\d+)$", tok)
        if m:
            return torch.LongTensor([int(m.group(1))])
        if tok not in self._spk_mem:
            self._spk_mem[tok] = len(self._spk_mem)
            if self.logger:
                self.logger.warning(
                    f"Speaker '{tok}' has no trailing digits; assign temp id {self._spk_mem[tok]}."
                )
        return torch.LongTensor([self._spk_mem[tok]])

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        audiopaths_sid_text_new = []
        lengths = []
        self.logger.info("Init dataset for Speaker {}...".format(self.sid))
        self.logger.info("Found {} audios from the index file...".format(len(self.audiopaths_sid_text)))

        for _, item in tqdm(
                self.audiopaths_sid_text.iterrows()
        ):
            item.ori_pth = self.data_root / item.ori_pth
            item.gt_pth = self.data_root / item.gt_pth

            # Ensure phonemes are parsed as a list of strings
            ori_phonemes_str = str(item.ori_phonemes)
            ori_phonemes_list = ori_phonemes_str.split(" ")
            ori_phonemes_len = len(ori_phonemes_list)

            if not (self.min_text_len <= ori_phonemes_len and ori_phonemes_len <= self.max_text_len):
                self.skipped += 1
                continue

            item.ori_phonemes = ori_phonemes_list
            item.ori_tone = [int(i) for i in str(item.ori_tone).split(" ") if i.strip().isdigit()]
            item.ori_word2ph = [int(i) for i in str(item.ori_word2ph).split(" ") if i.strip().isdigit()]

            # Ensure phonemes are parsed as a list of strings
            gt_phonemes_str = str(item.gt_phonemes)
            gt_phonemes_list = gt_phonemes_str.split(" ")
            gt_phonemes_len = len(gt_phonemes_list)

            if not (self.min_text_len <= gt_phonemes_len and gt_phonemes_len <= self.max_text_len):
                self.skipped += 1
                continue

            item.gt_phonemes = gt_phonemes_list
            item.gt_tone = [int(i) for i in str(item.gt_tone).split(" ") if i.strip().isdigit()]
            item.gt_word2ph = [int(i) for i in str(item.gt_word2ph).split(" ") if i.strip().isdigit()]

            audiopaths_sid_text_new.append(item)

            if not item.ori_pth.exists():
                self.skipped += 1
                if self.logger:
                    self.logger.warning(f"[Dataset] Original audio path not found: {item.ori_pth}; skipping item.")
                continue

            lengths.append(os.path.getsize(item.ori_pth) // (2 * self.hop_length))

        print("Skipped: ", self.skipped/ len(self))
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths

    def get_audio_text_speaker_pair(self, item):
        # separate filename, speaker_id and text
        ori_text_raw = item.ori_text
        ori_bert, ori_ja_bert, ori_en_bert, ori_phones, ori_tone, ori_language = self.get_text(
            item.ori_text, item.ori_word2ph, item.ori_phonemes, item.ori_tone, item.ori_lang, item.ori_pth
        )
        ori_spec, ori_wav, ori_audio_raw = self.get_audio(item.ori_pth)
        ori_sid = self._sid_tensor(item.ori_spk)

        gt_text_raw = item.gt_text
        gt_bert, gt_ja_bert, gt_en_bert, gt_phones, gt_tone, gt_language = self.get_text(
            item.gt_text, item.gt_word2ph, item.gt_phonemes, item.gt_tone, item.gt_lang, item.gt_pth
        )
        gt_spec, gt_wav, gt_audio_raw = self.get_audio(item.gt_pth)
        gt_sid = self._sid_tensor(item.gt_spk)

        data_sample = VCDataset(
            ori_sidx=self.sidx,
            ori_phones=ori_phones, ori_spec=ori_spec, ori_wav=ori_wav, ori_sid=ori_sid, ori_tone=ori_tone,
            ori_language=ori_language,
            ori_bert=ori_bert, ori_ja_bert=ori_ja_bert, ori_en_bert=ori_en_bert, ori_text_raw=ori_text_raw,
            ori_audio_raw=ori_audio_raw, ori_audiopath=item.ori_pth,
            gt_phones=gt_phones, gt_spec=gt_spec, gt_wav=gt_wav, gt_sid=gt_sid, gt_tone=gt_tone,
            gt_language=gt_language,
            gt_bert=gt_bert, gt_ja_bert=gt_ja_bert, gt_en_bert=gt_en_bert, gt_text_raw=gt_text_raw,
            gt_audio_raw=gt_audio_raw,
            gt_audiopath=item.gt_pth
        )
        return data_sample

    def get_audio(self, filename):
        audio, audio_sampling_rate = load_wav_to_torch(filename)
        audio_raw = audio.unsqueeze(0)  # note w/o any modification
        # print("a_sr", audio_sampling_rate, 'sr', self.sampling_rate)
        # breakpoint()
        if self.sampling_rate is not None and audio_sampling_rate != self.sampling_rate:
            # note: the audio_sampling_rate is different from the model sample rate -> resample audio to meet the requirement of model
            audio_transform = torchaudio.transforms.Resample(orig_freq=audio_sampling_rate, new_freq=self.sampling_rate)
            audio = audio_transform(audio)

        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        if self.use_mel_spec_posterior:
            spec = mel_spectrogram_torch(
                audio_norm,
                self.filter_length,
                self.n_mel_channels,
                self.sampling_rate,
                self.hop_length,
                self.win_length,
                self.data_conf.mel_fmin,
                self.data_conf.mel_fmax,
                center=False,
            )
        else:
            spec = spectrogram_torch(
                audio_norm,
                self.filter_length,
                self.sampling_rate,
                self.hop_length,
                self.win_length,
                center=False,
            )
        spec = torch.squeeze(spec, 0)

        return spec, audio_norm, audio_raw

    def get_text(self, text, word2ph, phone, tone, language_str, wav_path):
        phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)
        if self.add_blank:
            phone = commons.intersperse(phone, 0)
            tone = commons.intersperse(tone, 0)
            language = commons.intersperse(language, 0)
            for i in range(len(word2ph)):
                word2ph[i] = word2ph[i] * 2
            word2ph[0] += 1

        # 把扩展名从 .wav 改成 .bert.pt（只生成路径，不操作文件）
        bert_path = wav_path.with_name(wav_path.stem + ".bert.pt")
        try:
            bert_ori = torch.load(bert_path)
            assert bert_ori.shape[-1] == len(phone)
        except Exception as e:
            if self.logger:
                self.logger.warning("Bert load Failed")
                self.logger.warning(e)
            # fallback to random to avoid crash
            bert_ori = torch.randn(1024, len(phone))

        if language_str == "ZH":
            bert = bert_ori
            ja_bert = torch.randn(1024, len(phone))
            en_bert = torch.randn(1024, len(phone))
        elif language_str == "JP":
            bert = torch.randn(1024, len(phone))
            ja_bert = bert_ori
            en_bert = torch.randn(1024, len(phone))
        elif language_str == "EN":
            bert = torch.randn(1024, len(phone))
            ja_bert = torch.randn(1024, len(phone))
            en_bert = bert_ori
        else:
            bert = torch.randn(1024, len(phone))
            ja_bert = torch.randn(1024, len(phone))
            en_bert = torch.randn(1024, len(phone))

        phone = torch.LongTensor(phone)
        tone = torch.LongTensor(tone)
        language = torch.LongTensor(language)
        return bert, ja_bert, en_bert, phone, tone, language

    def get_sid(self, sid):
        # keep API but use unified logic
        return self._sid_tensor(sid)

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)


@dataclass
class VCDataset:
    ori_sidx: torch.LongTensor
    ori_phones: torch.LongTensor
    ori_spec: torch.LongTensor
    ori_wav: torch.LongTensor
    ori_sid: torch.LongTensor
    ori_tone: torch.LongTensor
    ori_language: torch.LongTensor
    ori_bert: torch.FloatTensor
    ori_ja_bert: torch.FloatTensor
    ori_en_bert: torch.FloatTensor
    ori_text_raw: str
    ori_audio_raw: torch.FloatTensor
    ori_audiopath: str
    gt_phones: torch.LongTensor
    gt_spec: torch.LongTensor
    gt_wav: torch.LongTensor
    gt_sid: torch.LongTensor
    gt_tone: torch.LongTensor
    gt_language: torch.LongTensor
    gt_bert: torch.FloatTensor
    gt_ja_bert: torch.FloatTensor
    gt_en_bert: torch.FloatTensor
    gt_text_raw: str
    gt_audio_raw: torch.FloatTensor
    gt_audiopath: str


@dataclass
class ModelOutput:
    """用于封装模型输出的数据类"""
    text: torch.Tensor
    text_len: torch.Tensor
    spec: torch.Tensor
    spec_len: torch.Tensor
    wav: torch.Tensor
    wav_len: torch.Tensor
    sid: torch.Tensor
    tone: torch.Tensor
    language: torch.Tensor
    bert: torch.Tensor
    ja_bert: torch.Tensor
    en_bert: torch.Tensor
    text_raw: str
    audio_raw: torch.Tensor
    audio_raw_len: torch.Tensor
    path_out: Optional[List[str]] = None

    def to(self, device):
        """
        Moves all tensor attributes of this dataclass to the specified device.
        """
        # Iterate over all the fields of the dataclass
        for field in fields(self):
            # Get the value of the current field
            value = getattr(self, field.name)

            # If the value is a torch.Tensor, move it to the device
            if isinstance(value, torch.Tensor):
                setattr(self, field.name, value.to(device))

        # Return self to allow for method chaining
        return self


@dataclass(frozen=True)
class ZeroShotSample:
    """Light-weight view over a zero-shot synthesis example."""

    speaker_id: str
    index: int
    prompt_path: Optional[Path] # original audio's path (voice clone need to turn the text using this audio)
    prompt_text: str # original audio's text
    prompt_language: Optional[str] # original audio's language
    target_path: Optional[Path] # target audio's path (will used in evaluation)
    target_text: str # target audio's text
    target_language: Optional[str] # target audio's language
    extra: Dict[str, object]

    @property
    def target_stub(self) -> str:
        if self.target_path is None:
            return f"sample_{self.index}"
        return self.target_path.stem


class AllSpeakerData:
    def __init__(self, config, dataset_config, logger):
        self.config = config
        self.dataset_config = dataset_config
        if logger:
            logger.debug("AllSpeakerData __init__: config and dataset_config set")
        self.logger = logger

        root_path = Path(to_absolute_path(str(self.dataset_config.root_path))).resolve()
        self._dataset_root = root_path
        self._filelists_dir = self._dataset_root / "filelists"
        self._data_root = self._dataset_root.parent
        if self.dataset_config.speaker_id is not None:
            self.speakers_ids = [self.dataset_config.speaker_id]
        else:
            audios_root = self._dataset_root / "audios"
            if audios_root.exists():
                self.speakers_ids = sorted(d for d in os.listdir(audios_root) if not d.startswith("."))
            else:
                # Fallback to manual listing to remain compatible with legacy layouts.
                self.speakers_ids = sorted(
                    d for d in os.listdir(os.path.join(self.dataset_config.root_path, "audios"))
                    if not d.startswith(".")
                )
        self.speaker_id_indices = {}
        for idx, sid in enumerate(self.speakers_ids):
            self.speaker_id_indices[sid] = idx

        collate_fn = TextAudioSpeakerCollate()

        self.speaker_datasets = {}
        self.speaker_dataloaders = {}
        for speaker_id in self.speakers_ids:
            speaker_idx = self.speaker_id_indices[speaker_id]
            self.speaker_datasets[speaker_id] = TextAudioSpeakerDataset(
                dataset_config,
                speaker_id,
                speaker_idx,
                sampling_rate=getattr(config, "sampling_rate", None),
                logger=logger,
            )
            num_workers = int(os.environ.get("DATA_LOADER_WORKERS", 4))
            try:
                self.speaker_dataloaders[speaker_id] = DataLoader(
                    self.speaker_datasets[speaker_id],
                    num_workers=num_workers,
                    shuffle=False,
                    collate_fn=collate_fn,
                    batch_size=config.batch_size,
                    pin_memory=True,
                    drop_last=False,
                )
            except PermissionError as exc:
                if num_workers <= 0:
                    raise
                if self.logger:
                    self.logger.warning(
                        "DataLoader with num_workers=%s failed (%s); falling back to num_workers=0.",
                        num_workers,
                        exc,
                    )
                self.speaker_dataloaders[speaker_id] = DataLoader(
                    self.speaker_datasets[speaker_id],
                    num_workers=0,
                    shuffle=False,
                    collate_fn=collate_fn,
                    batch_size=config.batch_size,
                    pin_memory=True,
                    drop_last=False,
                )

        self._zero_shot_samples: List[ZeroShotSample] = []
        self._zero_shot_samples_loaded = False
        self._zero_shot_eval_cache: Optional[Dict[str, Dict[str, object]]] = None
        self._train_files_map: Optional[Dict[str, str]] = None

    def get_speaker_dataset(self, speaker_id):
        return self.speaker_datasets.get(speaker_id, None)

    # ------------------------------------------------------------------
    # Zero-shot dataset helpers
    # ------------------------------------------------------------------
    def _resolve_audio_path(self, value: Optional[str]) -> Optional[Path]:
        if not value:
            return None
        candidate = Path(str(value))
        if candidate.is_absolute():
            return candidate.resolve()

        search_roots: Sequence[Path] = (self._data_root, self._dataset_root)
        for base in search_roots:
            resolved = (base / candidate).resolve()
            if resolved.exists():
                return resolved
        return (self._data_root / candidate).resolve()

    def _load_zero_shot_samples(self) -> None:
        if self._zero_shot_samples_loaded:
            return

        samples: List[ZeroShotSample] = []
        for speaker_id in self.speakers_ids:
            manifest_path = self._filelists_dir / f"{speaker_id}.json"
            if not manifest_path.exists():
                if self.logger:
                    self.logger.warning("Missing filelist for speaker %s at %s", speaker_id, manifest_path)
                continue
            try:
                with open(manifest_path, "r", encoding="utf-8") as handle:
                    records = json.load(handle)
            except json.JSONDecodeError as exc:
                if self.logger:
                    self.logger.error("Failed to parse %s: %s", manifest_path, exc)
                continue

            for idx, record in enumerate(records):
                prompt_path = self._resolve_audio_path(record.get("ori_pth"))
                target_path = self._resolve_audio_path(record.get("gt_pth")) or prompt_path

                prompt_text = record.get("ori_text", "") or ""
                target_text = record.get("gt_text", "") or prompt_text

                prompt_lang = record.get("ori_lang")
                target_lang = record.get("gt_lang")

                sample = ZeroShotSample(
                    speaker_id=str(speaker_id),
                    index=idx,
                    prompt_path=prompt_path,
                    prompt_text=prompt_text,
                    prompt_language=prompt_lang,
                    target_path=target_path,
                    target_text=target_text,
                    target_language=target_lang,
                    extra=dict(record),
                )
                samples.append(sample)

        self._zero_shot_samples = samples
        self._zero_shot_samples_loaded = True
        self._zero_shot_eval_cache = None

    def get_zero_shot_samples(
            self,
            *,
            speaker_id: Optional[str] = None,
            max_samples: Optional[int] = None,
    ) -> List[ZeroShotSample]:
        self._load_zero_shot_samples()

        filtered: Iterable[ZeroShotSample] = self._zero_shot_samples
        if speaker_id is not None:
            speaker_id = str(speaker_id)
            filtered = (entry for entry in filtered if entry.speaker_id == speaker_id)

        if max_samples is not None:
            result: List[ZeroShotSample] = []
            for entry in filtered:
                result.append(entry)
                if len(result) >= max_samples:
                    break
            return result

        return list(filtered)

    def iter_zero_shot_samples(
            self,
            *,
            speaker_id: Optional[str] = None,
            max_samples: Optional[int] = None,
    ) -> Iterator[ZeroShotSample]:
        for entry in self.get_zero_shot_samples(speaker_id=speaker_id, max_samples=max_samples):
            yield entry

    def get_zero_shot_evaluation_map(
            self,
            *,
            speaker_id: Optional[str] = None,
            max_samples: Optional[int] = None,
    ) -> Dict[str, Dict[str, object]]:
        if speaker_id is None and max_samples is None and self._zero_shot_eval_cache is not None:
            return dict(self._zero_shot_eval_cache)

        samples = self.get_zero_shot_samples(speaker_id=speaker_id, max_samples=max_samples)
        evaluation_map = {}
        for sample in samples:
            if not sample.target_path:
                continue
            evaluation_map[str(sample.target_path)] = {
                "speaker_id": sample.speaker_id,
                "text": sample.target_text,
                # Propagate language to evaluation so transcription can be multilingual-aware
                "language": sample.target_language or sample.prompt_language,
            }

        if speaker_id is None and max_samples is None:
            self._zero_shot_eval_cache = dict(evaluation_map)

        return evaluation_map

    def get_train_files_map(self) -> Dict[str, str]:
        if self._train_files_map is not None:
            return dict(self._train_files_map)

        transcripts: Dict[str, str] = {}
        for speaker_id, dataset in self.speaker_datasets.items():
            samples = getattr(dataset, "audiopaths_sid_text", [])
            if not samples:
                continue
            for record in samples:
                original_path = getattr(record, "ori_pth", None)
                if original_path is None:
                    continue
                original_path = Path(original_path).resolve()
                text = getattr(record, "ori_text", None) or getattr(record, "gt_text", "") or ""
                transcripts[str(original_path)] = text

        self._train_files_map = transcripts
        return dict(transcripts)

class TextAudioSpeakerCollate:
    """Zero-pads model inputs and targets"""

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """
        Collate's training batch from normalized text, audio and speaker identities
        batch: [text, spec, wav, sid, tone, language, bert, ja_bert, en_bert] (+ optional path)
        """
        # sort by spec len (descending)
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x.ori_spec.size(1) for x in batch]), dim=0, descending=True
        )
        paths_out = []

        # x: tuple (phones, spec, wav, sid, tone, language, bert, ja_bert, en_bert, text_raw, audio_raw)
        max_text_len = max([len(x.ori_phones) for x in batch])
        max_spec_len = max([x.ori_spec.size(1) for x in batch])
        max_wav_len = max([x.ori_wav.size(1) for x in batch])
        max_audio_raw_len = max([x.ori_audio_raw.size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        audio_raw_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        tone_padded = torch.LongTensor(len(batch), max_text_len)
        language_padded = torch.LongTensor(len(batch), max_text_len)
        bert_padded = torch.FloatTensor(len(batch), 1024, max_text_len)
        ja_bert_padded = torch.FloatTensor(len(batch), 1024, max_text_len)
        en_bert_padded = torch.FloatTensor(len(batch), 1024, max_text_len)

        spec_padded = torch.FloatTensor(len(batch), batch[0].ori_spec.size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        audio_raw_padded = torch.FloatTensor(len(batch), 1, max_audio_raw_len)

        text_padded.zero_()
        tone_padded.zero_()
        language_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        bert_padded.zero_()
        ja_bert_padded.zero_()
        en_bert_padded.zero_()
        text_raw = [''] * len(batch)
        audio_raw_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            src_item = batch[ids_sorted_decreasing[i]]
            paths_out.append(src_item.ori_audiopath)
            text = src_item.ori_phones  # Careful: it is the phones, e.g., (phones, spec, wav, sid, tone, language, bert, ja_bert, en_bert, text_raw)
            text_padded[i, : text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = src_item.ori_spec
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = src_item.ori_wav
            wav_padded[i, :, : wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            sid[i] = src_item.ori_sidx

            tone = src_item.ori_tone
            tone_padded[i, : tone.size(0)] = tone

            language = src_item.ori_language
            language_padded[i, : language.size(0)] = language

            bert = src_item.ori_bert
            bert_padded[i, :, : bert.size(1)] = bert

            ja_bert = src_item.ori_ja_bert
            ja_bert_padded[i, :, : ja_bert.size(1)] = ja_bert

            en_bert = src_item.ori_en_bert
            en_bert_padded[i, :, : en_bert.size(1)] = en_bert

            text_raw[i] = src_item.ori_text_raw
            audio_raw = src_item.ori_audio_raw
            audio_raw_padded[i, :, :audio_raw.size(1)] = audio_raw
            audio_raw_lengths[i] = audio_raw.size(1)

        outputs = ModelOutput(
            text=text_padded,
            text_len=text_lengths,
            spec=spec_padded,
            spec_len=spec_lengths,
            wav=wav_padded,
            wav_len=wav_lengths,
            sid=sid,
            tone=tone_padded,
            language=language_padded,
            bert=bert_padded,
            ja_bert=ja_bert_padded,
            en_bert=en_bert_padded,
            text_raw=text_raw,
            audio_raw=audio_raw_padded,
            audio_raw_len=audio_raw_lengths,
            path_out=paths_out
        )
        # NOTE:
        '''
            OZSPEECH:
            target_name <--> path_out,  target_transcript <--> text, prompt_filepath <--> wav
        '''
        return outputs


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <= b2}
    or {x | b2 < length(x) <= b3}.
    It removes samples which are not included in the boundaries.
    """

    def __init__(
            self,
            dataset,
            batch_size,
            boundaries,
            num_replicas=None,
            rank=None,
            shuffle=True,
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        try:
            for i in range(len(buckets) - 1, 0, -1):
                if len(buckets[i]) == 0:
                    buckets.pop(i)
                    self.boundaries.pop(i + 1)
            assert all(len(bucket) > 0 for bucket in buckets)
        except Exception as e:
            print("Bucket warning ", e)
            for i in range(len(buckets) - 1, -1, -1):
                if len(buckets[i]) == 0:
                    buckets.pop(i)
                    self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (
                          total_batch_size - (len_bucket % total_batch_size)
                  ) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            if len_bucket == 0:
                continue
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = (
                    ids_bucket
                    + ids_bucket * (rem // len_bucket)
                    + ids_bucket[: (rem % len_bucket)]
            )

            # subsample
            ids_bucket = ids_bucket[self.rank:: self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [
                    bucket[idx]
                    for idx in ids_bucket[
                        j * self.batch_size: (j + 1) * self.batch_size
                    ]
                ]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size
