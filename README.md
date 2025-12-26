![logo](pictures/logo.png)

# TwinCart — Multimodal product matching for E-commerce

TwinCart is an implementation of a
[7th place solution](https://www.kaggle.com/competitions/shopee-product-matching/writeups/t-nakamura-7th-place-solution)
from the
[Kaggle Shopee – Price Match Guarantee competition](https://www.kaggle.com/competitions/shopee-product-matching/overview).
It matches products using both images and titles by training separate embedding
models for vision and text, then retrieving nearest neighbors in the joint
search space to propose “same product” candidates. The project is built to be
easy to run end to end with uv for dependencies, dvc for datasets and artifacts,
and optional MLflow serving. It also supports exporting models to ONNX and
TensorRT for faster inference on GPU.

Tested setup: Ubuntu 22.04 on [Vast.ai](https://vast.ai/) with an NVIDIA A100
40GB GPU.

## Installation Guide

### Python & Environment

#### 1. Package Manager

We use modern [uv](https://github.com/astral-sh/uv) for managing python
packages.

To install it run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

```bash
unset VIRTUAL_ENV
```

(it will save `uv` binary into the
<span style="color:#6ee7b7; background:rgba(110, 231, 183, 0.05); padding: 2px 5px; border-radius: 6px;">~/.local/bin</span>)

Check the installation by running:

```bash
uv --version
```

(you should see something like
<span style="color:#f2c94c; background:rgba(242, 201, 76, 0.05); padding: 2px 5px; border-radius: 6px;">uv
0.9.9</span>)

#### 2. Python

We use `Python 3.11.9`. You can try other Python versions, but we recommend
installing `Python 3.11.9`, as the project was tested with it. You can do it
using:

```bash
uv python install 3.11.9
uv python list
```

(this installs `CPython 3.11.9` into `uv`’s own directory under
<span style="color:#6ee7b7; background:rgba(110, 231, 183, 0.05); padding: 2px 5px; border-radius: 6px;">~/.local/share/uv/python/...</span>)

#### 3. Environment

It is time to create and activate a virtual environment.

Move into the repo root:

```bash
cd <path-to-repository>/twincart-product-matching
```

Run:

```bash
uv sync
```

(`uv` creates
<span style="color:#6ee7b7; background:rgba(110, 231, 183, 0.05); padding: 2px 5px; border-radius: 6px;">.venv</span>
directory)

#### 4. DVC

To initialize and connect to dvc run:

```bash
uv run dvc init
uv run dvc remote add -d gdrive "gdrive://<FOLDER_ID>"
uv run dvc remote modify gdrive gdrive_use_service_account false
```

(`FOLDER_DIR` can be retrieved from the url)

Verification (optional):

```bash
cat .dvc/config
uv run dvc remote list
```

```bash
uv run dvc remote modify --local gdrive gdrive_client_id 'CLIENT_ID'
uv run dvc remote modify --local gdrive gdrive_client_secret 'CLIENT_SECRET'
```

(check with `uv run dvc status -c`)

```bash
uv run dvc add models lightning_logs mlruns plots data
```

If you already have files stored in your Google Drive, then you can pull them
(optional):

```bash
uv run dvc pull
```

#### 5. Hugging Face

```bash
uv run hf auth login
```

#### 6. MLflow

```bash
uv sync --extra serve
```

```bash
mkdir -p mlruns
uv run mlflow server --host 0.0.0.0 --port 8080 --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
```

#### 7. Downloading data

```bash
uv tool install kaggle
```

(check with `kaggle --version`)

Download `kaggle.json` legacy API from kaggle and place it in
<span style="color:#6ee7b7; background:rgba(110, 231, 183, 0.05); padding: 2px 5px; border-radius: 6px;">~/.kaggle/kaggle.json</span>.

```
chmod 600 ~/.kaggle/kaggle.json
```

```bash
kaggle competitions download -c shopee-product-matching -p data/raw
uv run python -m zipfile -e data/raw/shopee-product-matching.zip data/raw
rm -f data/raw/shopee-product-matching.zip
```

#### 8. Precommit (optional)

If you ever want to collaborate, run:

```bash
uv run pre-commit install
```

## Training

```bash
uv sync --extra train --extra logging
```

Folds preparation:

```bash
uv run python -m twincart.commands prepare-folds
```

Image embedding model training:

```bash
uv run python -m twincart.commands train --mode image --fold 0 \
  --overrides model@image_model=image_efficientnet_b0_gem
```

Text embedding model training:

```bash
uv run python -m twincart.commands train --mode text --fold 0
```

Push trained checkpoints to Google Drive (optional):

```bash
uv run dvc add models mlruns plots
uv run dvc push
```

## Exporting to ONNX

```bash
uv sync --extra export
```

To export models to `ONNX` run:

```bash
uv run python -m twincart.commands export-onnx
```

Outputs:

- <span style="color:#6ee7b7; background:rgba(110, 231, 183, 0.05); padding: 2px 5px; border-radius: 6px;">models/onnx/image/fold_0/model.onnx</span>
- <span style="color:#6ee7b7; background:rgba(110, 231, 183, 0.05); padding: 2px 5px; border-radius: 6px;">models/onnx/text/fold_0/model.onnx</span>

## Exporting to TensorRT

Install `TensorRT` into the current `uv` environment:

```bash
uv sync --extra export --extra tensorrt
```

Verify with (optional):

```bash
uv run python -c "import tensorrt as trt; print('TensorRT:', trt.__version__)"
```

Build TensorRT engines from ONNX model files:

```bash
uv run python -m twincart.commands export-trt
```

Outputs:

- <span style="color:#6ee7b7; background:rgba(110, 231, 183, 0.05); padding: 2px 5px; border-radius: 6px;">models/tensorrt/image/fold_0/model.engine</span>
- <span style="color:#6ee7b7; background:rgba(110, 231, 183, 0.05); padding: 2px 5px; border-radius: 6px;">models/tensorrt/text/fold_0/model.engine</span>

## Serving with MLflow

```bash
uv sync --extra serve
```

To build `MLFlow` model, run:

```bash
uv run python -m twincart.commands build-mlflow
```

This will start a new `MLflow` server:

```bash
uv run mlflow models serve -m models/mlflow/twincart_onnx_matcher \
  --no-conda -h 127.0.0.1 -p 5001
```

Sample server test:

```bash
URL="http://127.0.0.1:5001/invocations"
IMG="data/raw/train_images/0a1f72b12dee7317f586fa7f155dd681.jpg"

curl -sS "$URL" \
  -H "Content-Type: application/json" \
  -d @- <<JSON | python -m json.tool
{
  "dataframe_records": [
    {
      "image_path": "$IMG",
      "title": "some product title"
    }
  ]
}
JSON
```

## Useful Utilities

#### Commit changes:

```bash
uv sync --extra dev
uv run pre-commit install
```

To stash changes (if hooks changed something):

```bash
uv run pre-commit run --all-files
git add -A
```

#### Print the project tree, use:

```bash
tree --prune -I 'train_images|test_images'
```

## Licence

MIT
