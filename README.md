# Dataset Placement

Raw dataset files are not committed to GitHub. They must be placed manually under `data/raw`.

## SKAB

Only the following SKAB folders are used:

```txt
data/raw/skab/valve1/*.csv
data/raw/skab/valve2/*.csv
```

All CSV files in valve1 and valve2 are concatenated into a single dataset.

Additional metadata columns:
```txt
source_group: indicates whether the row comes from valve1 or valve2
source_file: indicates the source CSV file
```
These metadata columns are not used as model input.

Target column:
```txt
anomaly
```
Excluded from model input:
```txt
datetime
changepoint
source_group
source_file
```
BATADAL

Only BATADAL Training Dataset 2 is used.

Expected placement:
```txt
data/raw/batadal/training_dataset_2/*.csv
```
The target/label column must be checked from the actual CSV file and documented in the report.

## How to Run Tests

```bash
python -m pytest
```
## How to Run Smoke Tests

```bash
python -m src.experiments.run_skab_automata_smoke
python -m src.experiments.run_batadal_automata_smoke
```
## How to Run Automata Metrics

```bash
python -m src.experiments.run_skab_automata_metrics
python -m src.experiments.run_batadal_automata_metrics
```


# From Black-Box to Explainability: Probabilistic Automata for Time Series Analysis

## 1. Project Scope
## 2. Dataset Usage
### 2.1 SKAB
### 2.2 BATADAL

## 3. Preprocessing Pipeline
## 4. Automata-Based Model
### 4.1 PAA
### 4.2 SAX
### 4.3 Sliding Window Pattern Extraction
### 4.4 Probabilistic Transition Model
### 4.5 Unseen Pattern Handling with Levenshtein

## 5. Experimental Design
### 5.1 Original Scenario
### 5.2 Gaussian Noise Scenario
### 5.3 Unseen Scenario
### 5.4 Parameter Sweep

## 6. Results
### 6.1 SKAB Results
### 6.2 BATADAL Results
### 6.3 Noise Robustness
### 6.4 Unseen Pattern Behavior
### 6.5 Parameter Sensitivity

## 7. Explainability Example
## 8. Statistical Analysis
## 9. Limitations
## 10. Conclusion
