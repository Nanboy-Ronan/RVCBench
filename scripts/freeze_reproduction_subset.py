#!/usr/bin/env python3
"""Freeze a speaker-balanced subset before inspecting model outcomes."""
import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from omegaconf import OmegaConf
from rvcbench.benchmark.artifacts import atomic_json, digest, file_hash, input_records, input_fingerprint, sample_id
from rvcbench.datasets.zero_shot import ZeroShotDataset


from rvcbench.benchmark.subsets import freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-config', type=Path, default=Path('src/rvcbench/configs/dataset/libritts.yaml'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--speakers', type=int, default=8)
    parser.add_argument('--pairs-per-speaker', type=int, default=2)
    args = parser.parse_args()
    dataset_conf = OmegaConf.load(args.dataset_config)
    dataset_conf.use_hf_dataset = False
    conf = OmegaConf.create({'dataset': OmegaConf.to_container(dataset_conf)})
    dataset = ZeroShotDataset(conf, dataset_conf, logging.getLogger(__name__))
    print(freeze(dataset, args.output, args.speakers, args.pairs_per_speaker))


if __name__ == '__main__':
    main()
