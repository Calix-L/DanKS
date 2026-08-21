# Contributing

Thank you for helping improve DanKS.

## Before opening a change

- Keep generation-specific changes inside one generation unless repository tooling truly needs an update.
- Never add model weights, datasets, game records, internal documents, evaluation output, logs, credentials, or private endpoints.
- Do not rename or merge generation package namespaces.
- Add or update tests for changes to repository tooling.

## Development workflow

```bash
python -m pip install -e '.[dev]'
make test
make audit
```

Pull requests should explain the affected generation, compatibility impact, and verification performed. Generated `dist/` archives are not committed.

## Source provenance

Only contribute code you have the right to submit under Apache-2.0. Preserve applicable third-party copyright and license notices. Do not copy private project documents into issues or pull requests.
