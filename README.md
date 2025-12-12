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
