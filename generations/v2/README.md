# V2

V2 contains the staged-training retrieval and selector stack. Its Python package is `DanKS`, isolated from V1.

## Import

```bash
python -m pip install -r generations/v2/source/requirements.txt
PYTHONPATH=generations/v2/source python -c "import DanKS; print(DanKS.__file__)"
```

## Read the code

```text
source/
├── DanKS/
│   ├── retrieval/       # action generation, partitioning, ranking, scoring
│   └── training/        # features, schema, NumPy and ONNX selectors
└── requirements.txt
```

Recommended first files:

1. `source/DanKS/retrieval/action_generator.py`
2. `source/DanKS/retrieval/ranker.py`
3. `source/DanKS/training/featurizer.py`
4. `source/DanKS/training/onnx_phase14_selector.py`

The shared game environment is available as `guandan.engine` at the repository root. `manifest.json` is the authoritative inventory. Model weights and original internal documentation are not included. Do not combine this package with V1 or V3 in one Python process.
