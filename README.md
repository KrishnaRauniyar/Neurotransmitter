# Neurotransmitter 

This repository provides a complete machine learning pipeline for predicting neurotransmitter types using multiple algorithms including Deep Neural Networks (balanced & unbalanced), Random Forest, SVM, clustering methods, and a **Roshambo-based 3D shape–color similarity pipeline**.

---

## Installation

Follow the steps below to set up the project on your local machine.

### **Prerequisites**

Install the following before running the project:

- Python 3.8+
- TensorFlow  
- Pandas  
- Scikit-learn  
- Matplotlib  
- Imbalanced-learn  
- Seaborn  
- Scipy  

### **Clone the Repository**

```bash
git clone https://github.com/KrishnaRauniyar/Neurotransmitter.git
cd Neurotransmitter
```

### **Install Dependencies**

```bash
pip install -r requirements.txt
```

## Usage

Below are instructions for running each module.

### **Cluster Module**

#### Distance Matrix Requirement
A distance matrix file is required, which can be generated using:
https://github.com/dbxmcf/hsp70_actin

### **Run Clustermap**

```bash
cd cluster/clustermap
python clustermap.py
```

### **Run K-means**

```bash
cd cluster/kmeans
python kmeans.py
```

### **Deep Neural Network (DNN) — Unbalanced Data**

Create a results/ directory before running.

```bash
mkdir results   # if not already created
cd dnn_unbalanced
python neuro_transmitter_unbalanced.py
```

### **Deep Neural Network (DNN) — Balanced Data**

Create a results/ directory before running.

```bash
mkdir results   # if not already created
cd dnn_balanced
python neuro_transmitter_balanced.py
```

### **Random Forest Classifier**

```bash
cd random_forest
python random_forest.py
```

### **Support Vector Machine (SVM)**

```bash
cd svm
python svm.py
```

### **Roshambo Shape–Color Molecular Similarity**

The **roshambo/** folder contains your Roshambo-based molecular similarity pipeline. 

### Install Roshambo Package

You must install the Roshambo package separately before running the pipeline. More information can be found here: https://github.com/molecularinformatics/roshambo

**1. Clone the repository:**

```bash
git clone https://github.com/rashatwi/roshambo.git
cd roshambo
```

**2. Standard Installation:**

```bash
pip3 install .
```

**3. Editable Installation (Recommended for HPC/Clusters):**

```bash
pip3 install -e .
```

### Example Usage of Roshambo

The following Python code snippet demonstrates how to use the get_similarity_scores API function:

```bash
from roshambo.api import get_similarity_scores

get_similarity_scores(
    ref_file="query.sdf",
    dataset_files_pattern="dataset.sdf",
    ignore_hs=True,
    n_confs=0,
    use_carbon_radii=True,
    color=True,
    sort_by="ComboTanimoto",
    write_to_file=True,
    gpu_id=0,
    working_dir="data/basic_run",
)
```

#### Script Description:

* **Compare** a reference molecule (`query.sdf`) to all molecules in `dataset.sdf`.
* **Compute** 3D shape similarity, color similarity, and `ComboTanimoto` scores.
* **Ignore** hydrogens for molecular alignment and shape calculation.
* **Write results** to the directory specified by `working_dir`.

Your project’s `roshambo/` folder integrates this workflow for specific neurotransmitter datasets.
