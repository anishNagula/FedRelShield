# ULTRA Baseline Reproduction

## Upstream

Repository: DeepGraphLearning/ULTRA

Commit:

427966ad8ed60420eef034063d44f3153addff90

## Environment

Platform: Apple Silicon macOS

Python: 3.9.6

NumPy: 1.26.4

PyTorch: 2.1.0

PyTorch Geometric: 2.4.0

torch-scatter: 2.1.2

Device: CPU

## Experiment

Dataset: CoDExSmall

Checkpoint: ultra_4g.pth

Seed: 1024

Training epochs: 0

Evaluation protocol: filtered link prediction

## Dataset Statistics

Train triples: 32888

Validation triples: 1827

Test triples: 1828

## Validation Results

MR: 38.5285

MRR: 0.477777

Hits@1: 0.373290

Hits@3: 0.529557

Hits@10: 0.676793

## Test Results

MR: 42.3884

MRR: 0.463778

Hits@1: 0.360503

Hits@3: 0.514497

Hits@10: 0.665208

## Command

python script/run.py \
  -c config/fedrelshield/reproduction.yaml \
  --dataset CoDExSmall \
  --epochs 0 \
  --bpe null \
  --gpus null \
  --ckpt "$(pwd)/ckpts/ultra_4g.pth"

48a046e708adf5632d87c30eacae01f5f51466b2301effdc2cb42358d22854e0  ckpts/ultra_4g.pth
