# Security policy

## Supported versions

Fixes are made on `main`. The `v1` branch is frozen at the paper release and does not receive fixes.

## Reporting a vulnerability

Do not open a public issue. Report it privately through GitHub
([Security → Report a vulnerability](https://github.com/Nanboy-Ronan/RVCBench/security/advisories/new)) or by
email to **ruinanjin@alumni.ubc.ca**, with the affected command or file and the steps to reproduce it.
You can expect a first reply within a week.

RVCBench runs third-party model code and loads checkpoints from paths and Hugging Face repositories named
in its configs. Only run configs and checkpoints you trust.
