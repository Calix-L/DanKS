# V1

V1 contains the retrieval and NumPy selector foundations of DanKS. Its Python package is `DanKS`.

## Import

```bash
python -m pip install -r generations/v1/source/requirements.txt
PYTHONPATH=generations/v1/source python -c "import DanKS; print(DanKS.__file__)"
```

## Read the code

```text
source/
├── DanKS/
│   ├── retrieval/       # candidate generation, partitioning, ranking, scoring
│   ├── training/        # features, schema, NumPy selector
│   ├── plm_adapter/     # request payload adapter
│   └── plm_eval/        # evaluation-side source code
├── KSplatform/          # platform protocol source
├── scripts/             # operational wrappers retained with the snapshot
└── requirements.txt
```

Recommended first files:

1. `source/DanKS/retrieval/ranker.py`
2. `source/DanKS/retrieval/partitioner.py`
3. `source/DanKS/training/featurizer.py`
4. `source/DanKS/training/numpy_selector.py`

`manifest.json` is the authoritative inventory. Model weights and original internal documentation are not included. Do not combine this package with V2 or V3 in one Python process.
