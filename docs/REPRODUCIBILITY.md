# Reproducibility

DanKS supports reproducibility of the **public code bundles**, not reproduction of private trained models.

## Snapshot identity

Each generation manifest contains a stable generation name and a sorted list of file paths, byte sizes, and SHA256 values. It contains no source-machine path or import timestamp.

The importer uses explicit top-level allowlists, omits generated Cython sources, and deterministically normalizes approved private workspace prefixes. These transformations are part of the public snapshot definition.

## Archive identity

Archive construction normalizes member order, owner/group identifiers, names, modes, and modification times. The gzip header is normalized as well. Two builds from unchanged repository content therefore produce the same SHA256.

## Verification

```bash
make test
make audit
make package
sha256sum -c dist/SHA256SUMS
```

Model training and evaluation reproducibility are outside the public repository because their weights, datasets, records, and internal procedures are private.
