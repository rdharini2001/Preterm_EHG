# Execution environment

- Python: `3.13.5`
- Platform: `Linux-6.18.44-x86_64-with-glibc2.41`

| Library | Version used for included computations |
|---|---|
| `numpy` | `2.3.5` |
| `pandas` | `2.2.3` |
| `sklearn` | `1.8.0` |
| `scipy` | `1.17.0` |
| `imblearn` | `0.14.1` |
| `xgboost` | `3.1.3` |
| `torch` | `2.10.0+cpu` |
| `matplotlib` | `3.10.8` |

For a new environment, install from `pyproject.toml` with `pip install -e '.[test,deep]'`. The full workflow uses seed 2026 and saves its train/test manifests.
