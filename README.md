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

(`uv` creates
<span style="color:#6ee7b7; background:rgba(110, 231, 183, 0.05); padding: 2px 5px; border-radius: 6px;">.venv</span>
directory)

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
uv run dvc add models lightning_logs mlruns plots data
```

If you already have files stored in your Google Drive, then you can pull them
(optional):

```
uv run dvc pull
```

#### 5. Hugging Face

```
uv run hf auth login
```

#### 6. MLflow

```
mkdir -p mlruns
uv run mlflow server --host 0.0.0.0 --port 8080 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

#### 7. Downloading data

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

#### 8. Precommit (optional)

If you ever want to collaborate, run:

```
uv run pre-commit install
```

## Training

```
uv sync --extra train --extra logging
```

Folds preparation:

```
uv run python -m twincart.commands prepare_folds
```

Image embedding model training:

```
uv run python -m twincart.commands train --mode image --fold 0 \
  --overrides model@image_model=image_efficientnet_b0_gem
```

Text embedding model training:

```
uv run python -m twincart.commands train --mode text --fold 0
```

Push trained checkpoints to Google Drive (optional):

```
uv run dvc add models mlruns plots
uv run dvc push
```

## Exporting to ONNX

```
uv sync --extra export
```

To export models to `ONNX` run:

```
uv run python -m twincart.export.cli \
  --image-ckpt models/checkpoints/image/fold_0/best.ckpt \
  --text-ckpt models/checkpoints/text/fold_0/best.ckpt \
  --out-dir models/onnx \
  --image-h 420 --image-w 420 \
  --text-max-len 64 \
  --opset 17
```

## Serving with MLflow

```
uv sync --extra serve
```

To build `MLFlow` model, run:

```
uv run python -m twincart.serving.build_mlflow \
  --image-onnx models/onnx/image/fold_0/model.onnx \
  --text-onnx models/onnx/text/fold_0/model.onnx \
  --tokenizer cahya/distilbert-base-indonesian \
  --out-dir models/mlflow/twincart_onnx_matcher \
  --image-h 420 --image-w 420 \
  --text-max-len 64 \
  --force
```

This will start a new `MLflow` server:

```
uv run mlflow models serve -m models/mlflow/twincart_onnx_matcher --no-conda -h 127.0.0.1 -p 5001
```

Sample server test:

```
curl -sS http://127.0.0.1:5001/invocations \
  -H 'Content-Type: application/json' \
  -d '{
    "dataframe_records": [
      {
        "image_path": "/workspace/userspace/ml-course/mlops/twincart-product-matching/data/raw/train_images/0a1f72b12dee7317f586fa7f155dd681.jpg",
        "title": "some product title"
      }
    ]
  }' | python -m json.tool
```

## Useful Utilities

To commit changes:

```
uv sync --extra dev
uv run pre-commit install
```

To stash changes (if hooks changed something):

```
uv run pre-commit run --all-files
git add -A
```

To print the project tree, use:

```
tree --prune -I 'train_images|test_images'
```

## Licence

MIT
