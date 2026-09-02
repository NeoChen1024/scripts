# Neo_Chen's Scripts — Agent Guide

Most scripts are standalone CLI tools — run directly via shebang.

## Python scripts (`python-src/neoscripts/`)
- `#!/usr/bin/env python` shebang — run directly: `./script.py` or `python script.py`
- Mostly `click` CLI; some `argparse`
- Two modules share code within the package:
  - `llm_request.py` is a library (`LLM_Config`, `LLM_History`, `llm_query()`) used by other scripts
  - `pic_dir_reduce.py` imports from `pic_dir_dedupe.py`
  - These need `PYTHONPATH=python-src` or `pip install -e python-src` (no `__init__.py`)
- Code style: Black (line-length 132, py311) + isort (profile=black) via pre-commit
  - Manual: `black --line-length 132 --target-version py311 . && isort --profile black .`
- No tests; deps in `requirements.txt`

## Shell scripts (`shellscript/`)
- `#!/usr/bin/env sh` or `#!/usr/bin/env bash` — transcoding helpers in `shellscript/transcoding/`
