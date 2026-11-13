# Environment & Data Setup Guide

This document outlines the verified, step-by-step commands to recreate the local Windows environment, authenticate Kaggle, download the IEEE-CIS Fraud Detection dataset, and initialize the PySpark + PyTorch Geometric stack.

---

## 1. Create & Lock the Conda Environment (Python 3.10)
Locking the environment to Python 3.10 ensures full compatibility with pre-built Windows wheels for PySpark, PyTorch, and PyTorch Geometric.

```powershell
# Create and activate isolated environment with Python 3.10
conda create -n fraud_gnn python=3.10 pip pyyaml -y
conda activate fraud_gnn
```

---

## 2. Install Big Data, Deep Learning & Vector DB Libraries
Install core system dependencies (like OpenJDK for Spark) via `conda-forge`, followed by PyTorch and specialized Python libraries via `pip`.

```powershell
# Install PySpark, Java (OpenJDK), and standard data science tools
conda install -c conda-forge pyspark openjdk numpy pandas matplotlib jupyter -y

# Install PyTorch (CPU baseline)
conda install pytorch torchvision cpuonly -c pytorch -y

# Install PyTorch Geometric, Qdrant Vector DB client, and Kaggle CLI
pip install torch-geometric qdrant-client kaggle
```

---

## 3. Authenticate Kaggle API & Download Dataset (Windows PowerShell)

### Option A: Browser Download & PowerShell Extraction (Recommended)
1. Visit the [IEEE-CIS Fraud Detection Data Page](https://kaggle.com) and click **"Join Competition"** / **"I Understand and Accept"** on the competition rules.
2. Click **"Download All"** to save `ieee-fraud-detection.zip` to your `Downloads` folder.
3. Extract the files directly into your project's `data\raw` directory:

```powershell
mkdir -Force data\raw
Expand-Archive -Path "$HOME\Downloads\ieee-fraud-detection.zip" -DestinationPath "data\raw" -Force
```

### Option B: CLI Authentication via Environment Variables
If using the Kaggle CLI directly from PowerShell, set your credentials in your active session:

```powershell
$env:KAGGLE_USERNAME="your_kaggle_username"
$env:KAGGLE_KEY="your_kaggle_api_token"
kaggle competitions download -c ieee-fraud-detection -p data\raw
```

---

## 4. Protect Git Repository from Large Data Files
Ensure raw CSV files (~600 MB+) are excluded from version control before committing code:

```powershell
Add-Content -Path ".gitignore" -Value "`ndata/"
```

---

## 5. Verify PySpark Ingestion
Launch Jupyter Notebook (`jupyter notebook`) and run the following verification block to confirm Spark initializes with 4GB driver memory and reads the raw CSV schemas:

```python
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder.appName("FraudGraphPipeline")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)

tx_df = spark.read.csv(
    "data/raw/train_transaction.csv", header=True, inferSchema=True
)
id_df = spark.read.csv(
    "data/raw/train_identity.csv", header=True, inferSchema=True
)

print(f"Transaction Rows: {tx_df.count():,}")
print(f"Identity Rows:    {id_df.count():,}")
```
