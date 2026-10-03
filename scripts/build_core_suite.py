#!/usr/bin/env python3
"""Freeze an RVCBench suite: paired tasks for the robustness evaluations of the paper.

By default it builds Core: small, stratified subsets. With ``--full`` every task takes all
pairs of its dataset instead, with the same tasks, pair identifiers and anchors, so Core is a
subset of the full suite. Selection uses only dataset metadata and SHA-256 ranking with a
fixed seed; it never looks at model outputs. Run it from the repository root:

    python scripts/build_core_suite.py --mirror results/.core/mirror --output src/rvcbench/suites/core_v1
    python scripts/build_core_suite.py --full --compress --suite full-v1 ... --output src/rvcbench/suites/full_v1

``--mirror`` must contain the Hub dataset folders (``Libritts/``, ``VCTK/``, ...) at the pinned
revision plus the protected references (``Protected_LibriTTS/<method>/...``).
"""
import argparse
import gzip
import json
import logging
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pandas as pd  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from rvcbench.benchmark.artifacts import atomic_json, digest, file_hash, input_fingerprint, input_records  # noqa: E402
from rvcbench.benchmark.submission import CONFIGS_DIR  # noqa: E402
from rvcbench.datasets.zero_shot import ZeroShotDataset  # noqa: E402

SEED = 20261002
STANDARD = ['sim', 'speechmos', 'wer', 'mcd']  # S/M/W/C of the paper's Table 11
NO_GROUND_TRUTH = ['sim', 'speechmos', 'wer', 'sva', 'emotion']  # Table 25: target text has no recording
COMPRESSION = ['stoi', 'mcd', 'sim', 'wer']  # Tables 43-44: processed clone vs clone
PROTECTED_METHODS = {'gaussian': 'Gaussian', 'spec': 'SPEC', 'safespeech': 'SafeSpeech', 'pop': 'POP', 'enkidu': 'Enkidu'}

# VCTK demographics of the 40 speakers in RVCBench (paper Appendix E, Tables 15-24).
ACCENTS = {
    'American': ['p294', 'p297', 'p299', 'p311', 'p334', 'p345'], 'Australian': ['p326'], 'British': ['s5'],
    'Canadian': ['p303', 'p317', 'p302', 'p316', 'p363'], 'English': ['p228', 'p229', 'p231', 'p226', 'p227', 'p232'],
    'Indian': ['p248', 'p251'], 'Irish': ['p266', 'p283', 'p288', 'p245', 'p298'], 'New Zealand': ['p335'],
    'Northern Irish': ['p238', 'p292', 'p304'], 'Scottish': ['p234', 'p249', 'p262', 'p237', 'p241'],
    'South African': ['p314', 'p323', 'p336', 'p347'], 'Welsh': ['p253'],
}
AGES = {18: ['p334', 'p336'], 19: ['p298', 'p323'], 20: ['p297', 'p302', 'p316'], 21: ['p311', 'p241'],
        22: ['p345', 's5', 'p228', 'p226', 'p363', 'p266', 'p288', 'p238', 'p304', 'p234', 'p249', 'p237', 'p253'],
        23: ['p317', 'p229', 'p231', 'p232', 'p248', 'p292', 'p262'], 24: ['p303', 'p283'],
        25: ['p299', 'p245', 'p335'], 26: ['p326', 'p251', 'p314', 'p347'], 33: ['p294'], 38: ['p227']}
FEMALE = {'p294', 'p297', 'p299', 's5', 'p303', 'p317', 'p228', 'p229', 'p231', 'p248', 'p266', 'p283', 'p288',
          'p335', 'p238', 'p234', 'p249', 'p262', 'p314', 'p323', 'p336', 'p253'}
ACCENT_OF = {s: a for a, speakers in ACCENTS.items() for s in speakers}
AGE_OF = {s: age for age, speakers in AGES.items() for s in speakers}
TEXTSHIFT_SPEAKERS = ['p283', 'p288', 'p316', 'p363']
DURATION_BINS = [(0, 3, 'under_3s'), (3, 6, '3_to_6s'), (6, 10, '6_to_10s'), (10, 1e9, 'over_10s')]
FULL = False  # set by --full: every selection keeps all candidates, in rank order


def first(items, count):
    return list(items) if FULL else list(items)[:count]


def rank(*parts):
    return digest([SEED, *[str(p) for p in parts]])


def age_group(speaker):
    age = AGE_OF.get(speaker)
    return 'unknown' if age is None else 'under_20' if age < 20 else '20_to_29' if age < 30 else '30_plus'


def row(source, *, pair_id, prompt, target, prefix='', dataset_name=None, **groups):
    """A manifest row in the frozen-subset format; ``prefix`` makes paths repository-relative."""
    return {'dataset_name': dataset_name or str(source['dataset_name']), 'split': 'default', 'pair_id': pair_id,
            'speaker_id': str(source['speaker_id']), 'manifest_variant': 'speaker',
            'source_manifest': str(source['source_manifest']), 'source_row': int(source['source_row']),
            'prompt_file_name': prefix + prompt, 'target_file_name': prefix + target,
            'prompt_text': str(source['prompt_text']), 'target_text': str(source['target_text']),
            'prompt_language': str(source['prompt_language']), 'target_language': str(source['target_language']),
            'source_index': int(source['_index']), **groups}


def load(mirror, config):
    frame = pd.read_parquet(Path(mirror) / config / 'metadata.parquet')
    frame['_index'] = range(len(frame))
    return frame


def pick(frame, count, *keys):
    return first(sorted(frame.to_dict('records'), key=lambda r: rank(*keys, r['pair_id'])), count)


def per_speaker(frame, speakers, per, key):
    rows = []
    for speaker in speakers:
        rows += pick(frame[frame.speaker_id == speaker], per, key, speaker)
    return rows


def ranked_speakers(frame, count, key):
    return first(sorted(frame.speaker_id.unique(), key=lambda s: rank(key, s)), count)


# --------------------------------------------------------------------------- tasks

def audioshift(m):
    vctk = load(m, 'VCTK')
    rows = []
    if FULL:
        rows = vctk[vctk.source_manifest == vctk.speaker_id + '.json'].to_dict('records')
    for accent, speakers in ({} if FULL else ACCENTS).items():
        chosen = sorted(speakers, key=lambda s: rank('audioshift', s))
        # Two pairs per accent, from two speakers when the accent has more than one.
        for i in range(2):
            speaker = chosen[i % len(chosen)]
            candidates = pick(vctk[vctk.speaker_id == speaker], 2, 'audioshift', speaker)
            rows.append(candidates[i // len(chosen)] if len(chosen) == 1 else candidates[0])
    return [row(r, pair_id=r['pair_id'], prompt=r['prompt_file_name'], target=r['target_file_name'],
                accent=ACCENT_OF[r['speaker_id']], gender='F' if r['speaker_id'] in FEMALE else 'M',
                age_group=age_group(r['speaker_id'])) for r in rows]


def textshift_standard(m):
    rows = per_speaker(load(m, 'VCTK'), TEXTSHIFT_SPEAKERS, 6, 'textshift-standard')
    return [row(r, pair_id=r['pair_id'], prompt=r['prompt_file_name'], target=r['target_file_name']) for r in rows]


def textshift_hallucination(m):
    rows = per_speaker(load(m, 'vctk_text_robust'), TEXTSHIFT_SPEAKERS, 6, 'hallucination')
    return [row(r, pair_id=r['pair_id'], prompt=r['prompt_file_name'], target=r['target_file_name']) for r in rows]


def robocall(m, group):
    frame = load(m, 'robotcall')
    frame = frame[frame.speaker_id.str.endswith(group)]
    rows = []
    for speaker in sorted(frame.speaker_id.unique()):
        own = frame[frame.speaker_id == speaker]
        if FULL:
            rows += pick(own, len(own), 'scam', speaker)
        elif group == 'robocall':
            # Two different scam categories per speaker, rotating through the categories.
            types = sorted(own.spam_type.unique(), key=lambda t: rank('scam', speaker, t))[:2]
            rows += [pick(own[own.spam_type == t], 1, 'scam', speaker, t)[0] for t in types]
        else:
            rows += pick(own, 2, 'scam-standard', speaker)
    return [row(r, pair_id=r['pair_id'], prompt=r['prompt_file_name'], target=r['target_file_name'],
                spam_type=str(r['spam_type'])) for r in rows]


def english(m):
    lib = load(m, 'Libritts')
    lib = lib[lib.source_manifest == lib.speaker_id + '.json']
    gender = pd.read_csv(Path(m) / 'Libritts' / 'audios_speaker_gender.csv', dtype=str)
    speakers = []
    for sex in ('F', 'M'):
        pool = lib[lib.speaker_id.isin(gender[gender.gender == sex].speaker_id)]
        speakers += ranked_speakers(pool, 6, f'english-{sex}')
    if FULL:  # every speaker, including those without a gender label
        speakers += ranked_speakers(lib[~lib.speaker_id.isin(gender.speaker_id)], 0, 'english-unknown')
    rows = per_speaker(lib, speakers, 2, 'english')
    sex_of = dict(zip(gender.speaker_id, gender.gender))
    return [row(r, pair_id=r['pair_id'], prompt=r['prompt_file_name'], target=r['target_file_name'],
                gender=sex_of.get(str(r['speaker_id']), 'unknown')) for r in rows]


def simple(config, key, speakers=12, per=2):
    def build(m):
        frame = load(m, config)
        rows = per_speaker(frame, ranked_speakers(frame, speakers, key), per, key)
        return [row(r, pair_id=r['pair_id'], prompt=r['prompt_file_name'], target=r['target_file_name']) for r in rows]
    return build


def crosslingual(m):
    frame = load(m, 'Bilingual_uedin')
    rows = []
    for source, target in (('EN', 'ZH'), ('ZH', 'EN')):
        direction = frame[(frame.prompt_language == source) & (frame.target_language == target)]
        for speaker in ranked_speakers(direction, 12, f'cross-{source}'):
            rows += pick(direction[direction.speaker_id == speaker], 1, 'cross', source, speaker)
    return [row(r, pair_id=r['pair_id'], prompt=r['prompt_file_name'], target=r['target_file_name'],
                direction=f"{r['prompt_language']}_to_{r['target_language']}") for r in rows]


def longtext(m):
    return [row(r, pair_id=r['pair_id'], prompt=r['prompt_file_name'], target=r['target_file_name'])
            for r in load(m, 'Long_context').to_dict('records')]


def longaudio(m, durations):
    """Same target per speaker; only the reference length changes across the four bins."""
    lib = load(m, 'Libritts')
    lib = lib[lib.source_manifest == lib.speaker_id + '.json']
    prompts = (lib[['speaker_id', 'prompt_file_name', 'prompt_text', 'prompt_language']].drop_duplicates('prompt_file_name'))
    prompts = prompts.assign(seconds=prompts.prompt_file_name.map(durations))

    def bin_of(seconds):
        return next(label for low, high, label in DURATION_BINS if low <= seconds < high)
    prompts = prompts.assign(bin=prompts.seconds.map(bin_of))
    eligible = [s for s in lib.speaker_id.unique()
                if all((prompts[(prompts.speaker_id == s)].bin == label).any() for *_, label in DURATION_BINS)]
    rows = []
    for speaker in first(sorted(eligible, key=lambda s: rank('longaudio', s)), 6):
        target = pick(lib[lib.speaker_id == speaker], 1, 'longaudio-target', speaker)[0]
        for *_, label in DURATION_BINS:
            own = prompts[(prompts.speaker_id == speaker) & (prompts.bin == label) &
                          (prompts.prompt_file_name != target['target_file_name'])]
            reference = sorted(own.to_dict('records'), key=lambda p: rank('longaudio', speaker, label, p['prompt_file_name']))[0]
            source = {**target, 'prompt_text': reference['prompt_text'], 'prompt_language': reference['prompt_language']}
            rows.append(row(source, pair_id=f"LongAudio-{speaker}-{label}", prompt=reference['prompt_file_name'],
                            target=target['target_file_name'], reference_duration_bin=label))
    return rows


def background(m, noisy):
    items = []
    for f in sorted((Path(m) / 'Background_noise' / 'filelists_noise').glob('p*.json')):
        items += json.loads(f.read_text())
    frame = load(m, 'Background_noise')
    base = frame.iloc[0].to_dict()
    rows = []
    for noise in sorted({re.search(r'_10dB_([a-z]+)\.wav$', i['ori_pth']).group(1) for i in items}):
        of_type = [i for i in items if i['ori_pth'].endswith(f'_10dB_{noise}.wav')]
        for n, item in enumerate(first(sorted(of_type, key=lambda i: rank('background', noise, i['ori_pth'])), 2)):
            prompt = item['ori_pth'] if noisy else re.sub(r'_10dB_[a-z]+\.wav$', '_clean.wav', item['ori_pth'])
            source = {**base, 'speaker_id': item['ori_spk'], 'source_manifest': f"{item['ori_spk']}.json",
                      'source_row': n, 'prompt_text': item['ori_text'], 'target_text': item['gt_text'],
                      'prompt_language': item['ori_lang'], 'target_language': item['gt_lang'],
                      '_index': len(rows)}
            rows.append(row(source, pair_id=f"Background-{item['ori_spk']}-{noise}-{n}", prompt=prompt,
                            target=item['gt_pth'], noise=noise))
    return rows


def multispeaker(m, mixed):
    frame = load(m, 'Multispeaker_libri')
    mix = frame[frame.prompt_file_name.str.endswith('_mixture.wav')].copy()
    mix['snr'] = mix.prompt_file_name.str.extract(r'_([+-]\d+dB)_')[0]
    mix['interferer'] = mix.prompt_file_name.str.extract(r'interferer(\d+)')[0]
    rows = []
    for snr in ('-5dB', '+0dB', '+5dB', '+10dB'):
        for interferer in ('121', '672'):
            for r in pick(mix[(mix.snr == snr) & (mix.interferer == interferer)], 3, 'multispeaker', snr, interferer):
                prompt = r['prompt_file_name'] if mixed else r['prompt_file_name'].replace('_mixture.wav', '_target.wav')
                rows.append(row(r, pair_id=r['pair_id'], prompt=prompt, target=r['target_file_name'],
                                snr=snr, interferer=interferer))
    return rows


def protected_pairs(m):
    lib = load(m, 'Libritts')
    lib = lib[lib.source_manifest == lib.speaker_id + '.json']
    return per_speaker(lib, ranked_speakers(lib, 10, 'adv'), 2, 'adv')


def adversarial(m, method):
    rows = []
    for r in protected_pairs(m):
        name = r['prompt_file_name'].split('audios/', 1)[1]
        prompt = f'Libritts/{r["prompt_file_name"]}' if method == 'clean' else f'Protected_LibriTTS/{method}/{name}'
        rows.append(row(r, pair_id=r['pair_id'], prompt=prompt, target=f"Libritts/{r['target_file_name']}",
                        protection=method))
    return rows


def compression_tasks():
    conditions = [('mp3-64k', {'kind': 'codec', 'codec': 'mp3', 'bitrate': '64k'}),
                  ('aac-64k', {'kind': 'codec', 'codec': 'aac', 'bitrate': '64k'}),
                  ('opus-24k', {'kind': 'codec', 'codec': 'opus', 'bitrate': '24k'}),
                  ('mp3-32k', {'kind': 'codec', 'codec': 'mp3', 'bitrate': '32k'}),
                  ('aac-32k', {'kind': 'codec', 'codec': 'aac', 'bitrate': '32k'}),
                  ('opus-16k', {'kind': 'codec', 'codec': 'opus', 'bitrate': '16k'}),
                  ('narrowband', {'kind': 'narrowband'})]
    return [{'task': f'compression-{name}', 'dimension': 'output',
             'evaluation': 'RVC-Compression/' + ('NarrowBand' if name == 'narrowband' else 'CodecCompression'),
             'derived_from': 'audioshift', 'transform': {**transform, 'sample_rate': 24000},
             'required_metrics': COMPRESSION} for name, transform in conditions]


def generated_tasks(m, durations):
    """(task, builder, dataset config, Hub folder or None for repository paths, spec fields)."""
    tasks = [
        ('audioshift', audioshift, 'vctk', 'VCTK',
         {'dimension': 'input', 'evaluation': 'RVC-AudioShift/Demography', 'group_by': ['accent', 'gender', 'age_group']}),
        ('textshift-standard', textshift_standard, 'vctk', 'VCTK',
         {'dimension': 'input', 'evaluation': 'RVC-TextShift/Standard prompts (reference for Hallucination)'}),
        ('textshift-hallucination', textshift_hallucination, 'vctk_text_robust', 'vctk_text_robust',
         {'dimension': 'input', 'evaluation': 'RVC-TextShift/Hallucination', 'anchor': 'textshift-standard'}),
        ('textshift-scam', lambda m: robocall(m, 'robocall'), 'robotcall', 'robotcall',
         {'dimension': 'input', 'evaluation': 'RVC-TextShift/Scam; RVC-Expression/Persuasion',
          'anchor': 'textshift-scam-standard', 'required_metrics': NO_GROUND_TRUTH, 'group_by': ['spam_type']}),
        ('textshift-scam-standard', lambda m: robocall(m, 'vctk'), 'robotcall', 'robotcall',
         {'dimension': 'input', 'evaluation': 'RVC-Expression/Normal VCTK context (reference for Scam)',
          'required_metrics': NO_GROUND_TRUTH}),
        ('english-libritts', english, 'libritts', 'Libritts',
         {'dimension': 'generation', 'evaluation': 'RVC-Multilingual/English-VC', 'group_by': ['gender']}),
        ('chinese', simple('AISHELL1_dev', 'chinese'), 'aishell', 'AISHELL1_dev',
         {'dimension': 'generation', 'evaluation': 'RVC-Multilingual/Chinese-VC'}),
        ('crosslingual', crosslingual, 'bilingual_uedin', 'Bilingual_uedin',
         {'dimension': 'generation', 'evaluation': 'RVC-Multilingual/CrossLingual', 'group_by': ['direction']}),
        ('french', simple('CommonVoiceFR_dev', 'french'), 'french', 'CommonVoiceFR_dev',
         {'dimension': 'generation', 'evaluation': 'RVC-Multilingual/French (Appendix Table 38)'}),
        ('longtext', longtext, 'long_librispeech', 'Long_context',
         {'dimension': 'generation', 'evaluation': 'RVC-LongContext/LongText'}),
        ('longaudio', lambda m: longaudio(m, durations), 'libritts', 'Libritts',
         {'dimension': 'generation', 'evaluation': 'RVC-LongContext/LongAudio', 'group_by': ['reference_duration_bin']}),
        ('background-clean', lambda m: background(m, False), 'background_noise', 'Background_noise',
         {'dimension': 'perturbation', 'evaluation': 'RVC-PassiveNoise/Background (clean references)'}),
        ('background', lambda m: background(m, True), 'background_noise', 'Background_noise',
         {'dimension': 'perturbation', 'evaluation': 'RVC-PassiveNoise/Background', 'anchor': 'background-clean',
          'group_by': ['noise']}),
        ('multispeaker-clean', lambda m: multispeaker(m, False), 'multispeaker_libri', 'Multispeaker_libri',
         {'dimension': 'perturbation', 'evaluation': 'RVC-PassiveNoise/MultiSpeaker (clean references)'}),
        ('multispeaker', lambda m: multispeaker(m, True), 'multispeaker_libri', 'Multispeaker_libri',
         {'dimension': 'perturbation', 'evaluation': 'RVC-PassiveNoise/MultiSpeaker', 'anchor': 'multispeaker-clean',
          'group_by': ['snr', 'interferer']}),
        ('adv-clean', lambda m: adversarial(m, 'clean'), 'libritts', None,
         {'dimension': 'perturbation', 'evaluation': 'RVC-AdvNoise (clean references)'}),
    ]
    for method, label in PROTECTED_METHODS.items():
        evaluation = 'RVC-AdvNoise/Gaussian' if method == 'gaussian' else f'RVC-AdvNoise/Adversary ({label})'
        tasks.append((f'adv-{method}', lambda m, method=method: adversarial(m, method), 'libritts', None,
                      {'dimension': 'perturbation', 'evaluation': evaluation, 'anchor': 'adv-clean'}))
    tasks.append(('antiprotect-spec', lambda m: adversarial(m, 'demucs_spec'), 'libritts', None,
                  {'dimension': 'perturbation', 'evaluation': 'RVC-AntiProtect/AntiProtection (DEMUCS on SPEC)',
                   'anchor': 'adv-clean'}))
    return tasks


def write_json(path, value, compress):
    """Plain JSON, or gzip with a fixed timestamp so rebuilds are byte-identical."""
    if not compress:
        atomic_json(path, value)
        return path
    path = path.with_name(path.name + '.gz')
    with path.open('wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0, compresslevel=9) as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False).encode('utf-8'))
    return path


def freeze_task(mirror, output, name, rows, dataset_config, folder, compress=False):
    pair_ids = [r['pair_id'] for r in rows]
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError(f'{name}: duplicate pair ids')
    manifest = write_json(output / f'{name}.metadata.json', rows, compress)
    config = OmegaConf.load(CONFIGS_DIR / 'dataset' / f'{dataset_config}.yaml')
    root = Path(mirror) / (folder or '')
    config.update({'root_path': str(root), 'use_hf_dataset': False, 'manifest_filename': str(manifest.resolve()),
                   'speaker_id': None})
    samples = ZeroShotDataset(OmegaConf.create({}), config, logging.getLogger('core')).get_zero_shot_samples()
    records = input_records(samples)
    missing = [r['pair_id'] for r in records if not r['prompt_sha256'] or not r['target_sha256']]
    if missing:
        raise FileNotFoundError(f'{name}: audio missing under {root} for {missing[:5]}')
    selection = write_json(output / f'{name}.selection.json', {
        'schema_version': 1, 'selection': 'all_pairs_v1' if FULL else 'sha256_rank_stratified_core_v1',
        'selection_seed': SEED,
        'requested': len(records), 'input_fingerprint': input_fingerprint(records),
        'metadata_sha256': file_hash(manifest),
        'samples': [{k: v for k, v in r.items() if k not in ('prompt_path', 'target_path', 'status', 'error')}
                    for r in records]}, compress)
    return len(records), manifest.name, selection.name


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mirror', type=Path, required=True)
    parser.add_argument('--durations', type=Path, required=True, help='JSON map of Libritts prompt file to seconds')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--suite', default='core-v1')
    parser.add_argument('--leaderboard', action='store_true')
    parser.add_argument('--full', action='store_true', help='Take every pair of each task instead of the Core subset')
    parser.add_argument('--compress', action='store_true', help='Write the per-task files as .json.gz')
    parser.add_argument('--skip-protected', action='store_true',
                        help='Leave out the protected-reference tasks (when their audio is not in the mirror)')
    args = parser.parse_args()
    global FULL
    FULL = args.full
    args.output.mkdir(parents=True, exist_ok=False)
    durations = json.loads(args.durations.read_text())
    tasks, counts = [], {}
    for name, build, dataset_config, folder, fields in generated_tasks(args.mirror, durations):
        if args.skip_protected and (name.startswith('adv-') or name.startswith('antiprotect-')):
            continue
        rows = build(args.mirror)
        counts[name], manifest, selection = freeze_task(args.mirror, args.output, name, rows, dataset_config, folder,
                                                        args.compress)
        task = {'task': name, **fields, 'dataset_config': dataset_config, 'manifest': manifest, 'selection': selection}
        task.update({'hf_config_name': folder} if folder else {'paths': 'repo'})
        tasks.append(task)
    tasks += compression_tasks()
    name = 'RVCBench-Full' if FULL else 'RVCBench-Core'
    scope = 'every pair of the datasets of' if FULL else 'small, paired subsets of'
    not_included = {
        'RVC-Detectability/GroundTruth, Deepfake': 'deepfake detectors are not yet part of this package (planned for core-v1.1)',
        'RVC-Expression/Persuasion EmTXT': 'the audio-LLM emotion-alignment judge is planned for core-v1.1; EMC is reported'}
    if args.skip_protected:
        not_included['RVC-AdvNoise, RVC-AntiProtect'] = ('the protected references of every LibriTTS prompt are not yet '
                                                         'in the Hub dataset; core-v1 covers these evaluations')
    atomic_json(args.output / 'suite.json', {
        'suite': args.suite, 'version': 1, 'leaderboard': args.leaderboard,
        'label': (f'{name}: {scope} the 18 robustness evaluations of the RVCBench paper.'
                  if args.leaderboard else
                  f'{name} preview: data and protocol under review. Scores are not leaderboard results.'),
        'paper': 'https://arxiv.org/abs/2602.00443',
        'hf_dataset_id': 'Nanboy/RVCBench', 'hf_revision': args.revision,
        'evaluation': {'required_metrics': STANDARD, 'wer_normalization': 'ascii_punctuation_removed_v2',
                       'generated_audio_max_seconds': None, 'seed': 42,
                       'bootstrap': {'enabled': True, 'num_samples': 1000, 'confidence_level': 0.95, 'seed': None}},
        'not_included': not_included,
        'tasks': tasks})
    print(json.dumps({'tasks': counts, 'generations': sum(counts.values())}, indent=2))


if __name__ == '__main__':
    main()
