# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-02-22

### Added

- Initial release of NorthernTribe Medical Models (NTMM).
- Pipeline: prepare data → train teacher → distill NTMM student → evaluate.
- Multi-dataset support: merge several Hugging Face medical datasets into one train/val/test split (see `mcp.json` → `datasets`).
- Scripts: `prepare_data.py`, `train_teacher.py`, `distil_student.py`, `evaluate_student.py`, `run_pipeline.py`.
- Shell script `run_all_steps.sh` with optional quick mode (`./run_all_steps.sh quick`).
- Config-driven run via `mcp.json` (paths, models, training and distillation params).
- Tests: config validation and optional import checks in `tests/`.
- Branded output: NTMM student models owned by NorthernTribe Research.
