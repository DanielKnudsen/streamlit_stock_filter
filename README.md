# Streamlit Stock Filter
A Streamlit app for technical analysis of Stockholm Stock Exchange stocks using indicators like SMA, RSI, and MACD.

## Setup
1. Clone the repository:
	```sh
	git clone https://github.com/DanielKnudsen/streamlit_stock_filter
	cd streamlit_stock_filter
	```
2. Install dependencies using [uv](https://github.com/astral-sh/uv):
	```sh
	uv pip install -r requirements.txt
	```
	Or, if you use a `pyproject.toml`/`uv.lock` workflow (recommended):
	```sh
	uv pip install --all
	```
	> **Note:** `requirements.txt` is not required if you use `pyproject.toml` and `uv.lock`.
3. Run the app:
	```sh
	uv run streamlit run app.py
	```

## Dependency Management

This project uses [uv](https://github.com/astral-sh/uv) for fast Python dependency management. Prefer `uv` commands over `pip` or `requirements.txt` for installing and updating dependencies. All dependencies are tracked in `pyproject.toml` and `uv.lock`.

## Deployment
Deployed on Streamlit Community Cloud.


## Virtual Environment

If you use a virtual environment, activate it with:
```sh
source .venv/bin/activate
```

