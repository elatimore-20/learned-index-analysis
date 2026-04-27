# Learned Index Analysis

This project implements and evaluates **learned index structures** against traditional indexing methods, focusing on how data distribution impacts performance.

The work is based on the idea introduced in *The Case for Learned Index Structures* (Kraska et al., 2018), where indexing is treated as a prediction problem rather than a fixed data structure.

---

## 📌 Overview

Traditional indexes such as **B-Trees** and **binary search** are designed as general-purpose solutions and do not adapt to the underlying data distribution.

This project explores whether learned models can improve indexing by approximating the mapping:  key → position

We compare:

- Binary Search (baseline)
- B-Tree Index
- Linear Learned Index
- Two-stage Recursive Model Index (RMI)

---

## ⚙️ Features

- Implementation of learned index structures from scratch
- Support for multiple datasets:
  - Uniform (ideal case)
  - Zipf (skewed distribution)
  - Real-world U.S. Baby Names dataset
- Evaluation across multiple metrics:
  - Lookup latency
  - Prediction error
  - Correction cost (last-mile search)
  - Memory usage
- Automated experiment pipeline and result generation
- Visualization of results through graphs

---

## 📊 Key Insights

- Learned indexes perform well when the data distribution is predictable (uniform)
- Performance degrades under skewed distributions (Zipf)
- Prediction error directly impacts correction cost and overall latency
- RMI reduces error compared to a linear model but introduces additional overhead
- Lower memory usage does not necessarily mean faster performance

---

## 🧪 How It Works

The system follows this pipeline:

1. Generate or load dataset
2. Build index structures (Binary, B-Tree, Learned Models)
3. Sample query keys
4. Perform lookups
5. Predict positions (learned models only)
6. Apply local binary search correction
7. Record performance metrics
8. Aggregate results and generate plots

---

## 🚀 How to Run

### Requirements
- Python 3.x
- NumPy
- Pandas
- Matplotlib

### Run the experiment

```bash
python learned_index_experiment.py

Optional parameters

* Dataset type: uniform / zipf / real
* Dataset size: 50K, 100K, 200K
* Number of queries: default 5000

## 📁 Project Structure:
learned-index-analysis/
│── results/                 # Graphs and experiment outputs
│   ├── latency_.png
│   ├── error_.png
│   ├── correction_.png
│   ├── memory_.png
│   └── experiment_results.csv
│── baby_names.csv.zip       # Real-world dataset
│── names.zip                # Additional dataset
│── learned_index_experiment.py  # Main execution script
│── README.md

The repository includes:

* Latency vs dataset size graphs
* Prediction error plots
* Correction cost analysis
* Memory usage comparisons

Additional graphs and detailed outputs are provided beyond those shown in the paper.

This project was developed as part of:

CSC 8712 – Advanced Database Systems
Georgia State University

The full report analyzes performance across multiple datasets and explains when learned indexes succeed or fail.

References

* Kraska et al., The Case for Learned Index Structures, SIGMOD 2018
* Comer, The Ubiquitous B-Tree

Authors

* Emmanuel Latimore
* Victor Lam
* Ninad Bangar

---

# 🔥 Why this is strong

✔ Matches your paper exactly  
✔ Explains pipeline (TA will check this)  
✔ Shows results + graphs exist  
✔ Looks like a real research repo  
✔ Easy to run  

---

# 💯 Optional upgrade (if you want extra polish)

Add a screenshot of one graph at the top:

```markdown
![Latency Comparison](plots/latency_uniform.png)
