# Modern Fraud Analytics: Scalable Pipelines with GNNs & Vector DBs

A production-ready pipeline utilizing *PySpark* for large-scale transaction feature engineering, *PyTorch Geometric* for anomalous relational detection via Graph Neural Networks (GNNs), and *Qdrant* for real-time vector similarity searching.

## 🏗️ System Architecture
1. *Data Layer:* PySpark ingestion and feature extraction from the IEEE-CIS Fraud dataset.
2. *Graph Topology Layer:* Mapping user-merchant relationships into heterogeneous network configurations using GraphFrames.
3. *Deep Learning Layer:* Node embedding generation using PyTorch Geometric (GraphSAGE / GCN).
4. *Vector Indexing & Inference:* Storing GNN structural embeddings inside Qdrant for real-time fraud matching queries.
5. *Evaluation:* Statistical evaluation via Precision-Recall trade-off tracking and simulated A/B testing frameworks.

## 🗺️ Project Blueprint & Progress
- [ ] *Phase 1: Spark Infrastructure*
  - [ ] Initialize local Spark cluster and establish data pipeline configurations
  - [ ] Transform raw tabular transactional metrics into scalable vertex and edge frames
- [ ] *Phase 2: Relational Network Modeling*
  - [ ] Map dataset to heterogeneous PyTorch Geometric graph formats
  - [ ] Train a structural GNN architecture to optimize fraud node classification
- [ ] *Phase 3: Production Indexing*
  - [ ] Extract latent vector embeddings from the optimized deep learning layers
  - [ ] Build a local Qdrant collection to manage high-dimensional vector similarity operations
- [ ] *Phase 4: Statistical Validation*
  - [ ] Implement system latency trackers and custom Precision-Recall validation engines
  - [ ] Code a synthetic A/B test simulation framework to verify target metrics

## 🛠️ Environment Configuration
bash
# Initialize isolated conda workspace
conda create -n fraud_gnn python=3.10 -y
conda activate fraud_gnn

# Install Big Data & PyTorch deep learning stacks
conda install -c conda-forge pyspark openjdk numpy pandas matplotlib -y
conda install pytorch torchvision cpuonly -c pytorch -y
pip install torch-geometric qdrant-client kaggle
