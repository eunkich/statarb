# statarb

Exploratory Study in Statistical Arbitrage

# Quick Start

To run training code,

```{bash}
pip install -r requirements.txt
python -u main.py | tee out.txt
```

To fetch data,

```{bash}
cp .env.example .env
```

Store your API key for [polygon.io](https://polygon.io) in `.env` file or export
it as a system environment variable.

```
export POLYGON_API_KEY="[YOUR_API_KEY]"
```
