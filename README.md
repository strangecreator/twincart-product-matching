![logo](pictures/logo.png)

# TwinCart — Multimodal product matching for E-commerce

## Installation Guide

### Python & Environment

#### 1. Package Manager

We use modern [uv](https://github.com/astral-sh/uv) for managing python
packages.

To install it run:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
unset VIRTUAL_ENV
```

(it will save `uv` binary into the
<span style="color:#6ee7b7; background:rgba(110, 231, 183, 0.05); padding: 2px 5px; border-radius: 6px;">~/.local/bin</span>)

Check the installation by running:

```
uv --version
```

(you should see something like
<span style="color:#f2c94c; background:rgba(242, 201, 76, 0.05); padding: 2px 5px; border-radius: 6px;">uv
0.9.9</span>)

#### 2. Python

We use `Python 3.11.9`. You can try other Python versions, but we recommend
installing `Python 3.11.9`, as the project was tested with it. You can do it
using:

```
uv python install 3.11.9
uv python list
```

(this installs `CPython 3.11.9` into `uv`’s own directory under
<span style="color:#6ee7b7; background:rgba(110, 231, 183, 0.05); padding: 2px 5px; border-radius: 6px;">~/.local/share/uv/python/...</span>)

#### 3. Environment

It is time to create and activate a virtual environment.

Move into the repo root:

```
cd <path-to-repository>/twincart-product-matching
```

Run:

```
uv sync
```

(`uv` creates `.venv` directory)

#### 4. DVC

To initialize and connect to dvc run:

```
uv run dvc init
uv run dvc remote add -d gdrive "gdrive://<FOLDER_ID>"
uv run dvc remote modify gdrive gdrive_use_service_account false
```

(`FOLDER_DIR` can be retrieved from the url)

Verification (optional):

```
cat .dvc/config
uv run dvc remote list
```

```
uv run dvc remote modify --local gdrive gdrive_client_id 'CLIENT_ID'
uv run dvc remote modify --local gdrive gdrive_client_secret 'CLIENT_SECRET'
```

(check with `uv run dvc status -c`)

```
git add .gitignore dvc.yaml .dvc/config
```

#### 5. Precommit (optional)

If you ever want to collaborate, run:

```
uv run pre-commit install
```

## Downloading data

```
uv tool install kaggle
```

(check with `kaggle --version`)

Download `kaggle.json` legacy API from kaggle and place it in
<span style="color:#6ee7b7; background:rgba(110, 231, 183, 0.05); padding: 2px 5px; border-radius: 6px;">~/.kaggle/kaggle.json</span>.

```
chmod 600 ~/.kaggle/kaggle.json
```

```
kaggle competitions download -c shopee-product-matching -p data/raw
uv run python -m zipfile -e data/raw/shopee-product-matching.zip data/raw
rm -f data/raw/shopee-product-matching.zip
```

## Training

```
uv sync --extra train --extra dev --extra logging
uv run pre-commit install
uv run pre-commit run -a
```

```
uv run python -m twincart.commands prepare_folds
```

```
uv run python -m twincart.commands train --mode image --fold 0 --overrides model@image_model=image_efficientnet_b0_gem
```
