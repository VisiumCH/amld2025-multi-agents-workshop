Creating env

Conda:
- conda create -n financial-advisor python=3.12
- conda activate financial-advisor
- pip install -e .

Venv:
- python -m venv .venv
- source .venv/bin/activate (Linux)
- .venv\Scripts\activate (Windows)
- pip install -e .

UV:
- uv pip sync -e .