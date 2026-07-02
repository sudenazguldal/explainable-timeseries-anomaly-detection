# Explainable Anomaly Detection in Industrial Time Series

Deep Learning vs. Probabilistic Automata for anomaly detection in industrial sensor data.

This repository compares two fundamentally different approaches to anomaly detection in industrial time series:

1. **Black-box deep learning models** — LSTM and 1D-CNN.
2. **An explainable probabilistic automata model** — PCA, PAA, SAX symbolic encoding, sliding-window pattern extraction, transition probabilities, and Levenshtein-distance-based handling of unseen patterns.

The goal is not simply to maximize accuracy. The point is to compare the raw predictive power of deep learning against a model that can explain, step by step, why it labeled a given point as anomalous.

## Key Findings

On SKAB, deep learning clearly outperforms the automata model. The best result is an F1-score of `0.850 ± 0.082` from the LSTM model; 1D-CNN is also strong at `0.798 ± 0.120`.

On BATADAL, both deep learning models struggle. LSTM reaches `0.227 ± 0.312` F1, while 1D-CNN collapses to an F1-score of `0.000` despite a high accuracy — it simply never predicts the anomaly class. This is a clean illustration of why accuracy alone is misleading on imbalanced anomaly data.

With its fixed parameters, the automata model produces a low F1-score on both datasets; tuning `window_size` and `alphabet_size` improves it substantially:

| Dataset | Fixed-Parameter Automata F1 | Best Automata F1 | Best Parameters |
|---|---:|---:|---|
| BATADAL | 0.182 | 0.257 | `window_size=6`, `alphabet_size=5` |
| SKAB | 0.083 ± 0.070 | 0.337 ± 0.144 | `window_size=6`, `alphabet_size=6` |

Even where its raw score is lower than deep learning, the automata model can justify its decisions through explicit state transitions, transition probabilities, unseen-pattern mapping, and a confidence score — something the neural models cannot natively provide.

## What This Project Does

This is a **time series anomaly detection** project. A time series is a sequence of measurements recorded in order over time — for example, the temperature, pressure, current, or vibration of a machine sampled every second.

An anomaly is a deviation from a system's normal behavior. In industrial systems this can mean a fault, a cyber attack, sensor degradation, a valve problem, a pressure leak, or unexpected operational behavior.

The project answers two separate questions:

1. How well do deep learning models detect these anomalies?
2. Can the automata model explain its decisions in an understandable way, even if its raw score is lower?

## Datasets

### SKAB

SKAB contains sensor readings from an industrial test rig. Only the `valve1` and `valve2` subsets are used here. Each row carries `source_group` and `source_file` metadata columns so that its origin can be traced; these columns are excluded from the model input and used only for splitting and lineage tracking.

| Property | Value |
|---|---:|
| Rows | 22472 |
| Features | 8 |
| Target column | `anomaly` |
| Normal samples | 14646 |
| Anomalous samples | 7826 |
| Source files used | 20 |

Sensor columns used from SKAB:

| Column | Meaning |
|---|---|
| `Accelerometer1RMS` | RMS vibration from the first accelerometer. |
| `Accelerometer2RMS` | RMS vibration from a second accelerometer at a different point. |
| `Current` | Electrical current draw; can indicate motor or system faults. |
| `Pressure` | Pressure reading; sensitive to valve or flow problems. |
| `Temperature` | Temperature reading. |
| `Thermocouple` | Temperature from a second sensor type. |
| `Voltage` | Supply/operating voltage. |
| `Volume Flow RateRMS` | RMS volumetric flow rate. |

SKAB is split by group rather than randomly. Rows from the same source CSV never appear in both the train and test sets, which prevents leakage across the split.

### BATADAL

BATADAL is a water distribution system dataset used for cyber-attack / anomaly detection. This project uses `Training Dataset 2`. The `DATETIME` column is used only to preserve temporal order and is excluded from the model input. The target column is `ATT_FLAG`.

| Property | Value |
|---|---:|
| Rows | 4177 |
| Features | 43 |
| Target column | `ATT_FLAG` |
| Time column | `DATETIME` |
| Normal samples | 3958 |
| Anomalous samples | 219 |
| Train rows | 2506 |
| Validation rows | 835 |
| Test rows | 836 |

BATADAL is highly imbalanced — far more normal samples than anomalies — so a model that always predicts "normal" can still score a high accuracy. This is exactly the failure mode seen in the BATADAL 1D-CNN result: ~0.901 accuracy with an F1-score of 0.000.

BATADAL is split chronologically:

```text
60% train -> 20% validation -> 20% test
```

A random split is not used, since future data leaking into training would produce unrealistically optimistic results for a time series problem.

### Data provenance and licensing

Both raw datasets are included in `data/raw/` so the project can be cloned and run end to end without external downloads.

- **SKAB** — Skoltech Anomaly Benchmark, [waico/SKAB](https://github.com/waico/SKAB), distributed under **GPL-3.0**. The raw CSV files retain this license.
- **BATADAL** — from the [BATtle of the Attack Detection Algorithms](https://www.batadal.net/data.html), distributed under **CC BY 4.0**. Attribution: Taormina, R. et al., "The Battle of the Attack Detection Algorithms," *Journal of Water Resources Planning and Management*, 2018.

The code in this repository is licensed separately (see [LICENSE](LICENSE)); the two dataset licenses above apply specifically to the files under `data/raw/`.

## Glossary

| Term | Meaning in this project |
|---|---|
| Time series | Sensor measurements ordered by time. |
| Feature | An input variable given to the model, e.g. pressure, temperature, current. |
| Target / Label | The class the model predicts. Here, `0 = normal`, `1 = anomaly`. |
| Anomaly | A deviation from normal system behavior. |
| Train set | The portion of data the model learns from. |
| Validation set | An intermediate split used for parameter/threshold selection. |
| Test set | The held-out split used for final performance measurement. |
| Data leakage | Test information leaking into training, producing overly optimistic results. |
| Accuracy | Fraction of correct predictions; misleading alone on imbalanced data. |
| Precision | Of everything the model calls "anomaly," how much actually is. |
| Recall | Of all real anomalies, how many were caught. |
| F1-score | Harmonic mean of precision and recall; central metric for anomaly detection. |
| LSTM | A recurrent deep learning model that can learn temporal dependencies. |
| 1D-CNN | A convolutional model that captures local patterns within a time window. |
| Sliding window | Splitting a time series into fixed-length consecutive segments. |
| PCA | Dimensionality reduction; the automata pipeline uses the first principal component (PC1). |
| PAA | Piecewise Aggregate Approximation — averages a time series over fixed segments. |
| SAX | Symbolic Aggregate approXimation — converts a numeric series into symbols (e.g. `a`, `b`, `c`). |
| Alphabet size | Number of distinct symbols used by SAX. |
| Window size | Number of symbols that form one state/pattern. |
| State | A symbolic pattern in the automata, e.g. `abca`. |
| Transition | A move from one state to another. |
| Transition probability | How likely a given transition was in the training data. |
| Transition density | Fraction of all possible transitions that were actually observed. |
| Unseen pattern | A pattern that never appeared during training but occurs at test time. |
| Levenshtein distance | Edit distance between two symbolic sequences. |
| Confidence score | The automata model's probability-based confidence in a decision. |

## Methodology

### Deep Learning Pipeline

Two deep learning models are used:

| Model | Why it was chosen |
|---|---|
| LSTM | Can learn how past measurements influence future behavior. |
| 1D-CNN | Captures short, local patterns within a time window. |

Pipeline:

```text
Raw sensor data
-> separate target/metadata columns
-> fit normalization on train only
-> transform validation/test with the fitted scaler
-> build sliding-window sequences
-> train LSTM and 1D-CNN
-> accuracy, precision, recall, F1, confusion matrix, PR/ROC curves
```

Deep learning settings:

| Setting | Value |
|---|---:|
| Sequence length | 32 |
| Batch size | 32 |
| Max epochs | 50 |
| Early stopping patience | 5 |
| Random seeds | 42, 123, 2026, 7, 999 |

SKAB uses 5 group folds and 5 seeds, so each model is summarized over 25 runs. BATADAL uses a single chronological split with 5 seeds, so each model is summarized over 5 runs.

### Automata Pipeline

The automata model does not learn directly from the raw multivariate sensor matrix. The data is first converted into a symbolic time series:

```text
Multivariate sensor data
-> standard scaling
-> PCA to extract PC1
-> PAA segment averaging
-> SAX symbolic encoding
-> sliding-window pattern extraction
-> pattern = state
-> transition probability estimation
-> low-probability transition = anomaly candidate
```

Core assumption:

```text
Frequent transitions in training -> normal behavior
Rare or unseen transitions in training -> anomaly candidate
```

Fixed automata parameters:

| Parameter | Value |
|---|---:|
| `window_size` | 4 |
| `alphabet_size` | 3 |
| `paa_segments` | 256 |
| `fallback_probability` | 0.000001 |
| `anomaly_threshold` | 0.05 |

Parameter sweep range:

```text
window_size:   3, 4, 5, 6
alphabet_size: 3, 4, 5, 6
```

### Unseen Pattern Handling

If a pattern never seen during training appears at test time, the model must not fail. Instead, Levenshtein distance is used to find the closest known state:

```text
Incoming unseen pattern: ccca
Closest known state:     acca
Edit distance:           1
```

The transition probability and decision are then derived from this mapped state, which lets the automata model justify its predictions rather than simply emitting them.

## Experiment Design

| Aspect | Decision |
|---|---|
| Datasets | 2: SKAB and BATADAL |
| DL models | 2: LSTM and 1D-CNN |
| Split strategy | SKAB: group split · BATADAL: chronological 60/20/20 |
| Scenarios | Original, Gaussian noise, unseen-only |
| Automata parameters | Fixed setting + window/alphabet sweep |
| Statistical test | Wilcoxon signed-rank |
| Figures | Confusion matrix, precision-recall curve, parameter heatmap, transition heatmap, transition graph |

A train-on-one-dataset / test-on-the-other cross-dataset transfer experiment was not performed; SKAB and BATADAL were evaluated independently, and this is noted as a limitation below.

## Environment and Runtime

| Pipeline | Environment | Approx. runtime |
|---|---|---:|
| Deep learning full training | Local workstation, Python 3.13.3 | ~7 minutes |
| Automata full experiment suite | Local workstation | ~4 minutes |

Smoke-test outputs are kept separate from final training outputs. Smoke tests only confirm that the code runs; the reported results below come from the full experiment runs.

**Reproducibility note:** the automata pipeline's SKAB split uses `StratifiedGroupKFold`, whose fold assignment can differ across `scikit-learn` versions even with the exact same seed. `requirements.txt` pins exact versions; the SKAB automata numbers in this README were generated and verified against those pinned versions. Deep learning results use the same SKAB grouped-split strategy, so exact reproduction of the DL numbers also depends on `scikit-learn` matching the pinned version, in addition to `torch`/`tensorflow`.

## Deep Learning Results

| Dataset | Model | Runs | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|---:|
| BATADAL | 1D-CNN | 5 | 0.901 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| BATADAL | LSTM | 5 | 0.914 ± 0.022 | 0.611 ± 0.386 | 0.182 ± 0.282 | 0.227 ± 0.312 |
| SKAB | 1D-CNN | 25 | 0.881 ± 0.060 | 0.941 ± 0.071 | 0.711 ± 0.166 | 0.798 ± 0.120 |
| SKAB | LSTM | 25 | 0.908 ± 0.043 | 0.964 ± 0.051 | 0.773 ± 0.130 | 0.850 ± 0.082 |

### Interpretation

On SKAB, LSTM is the strongest model — both precision and recall are high, which drives its F1-score up. 1D-CNN is also strong but less stable across seeds than LSTM.

On BATADAL, accuracy is misleading. 1D-CNN assigns almost every sample to the normal class, so accuracy looks high while precision, recall, and F1 are all zero. This is why F1-score and the confusion matrix matter more than accuracy for BATADAL.

![SKAB LSTM confusion matrix](reports/figures/readme/skab_lstm_confusion_matrix_seed42.png)

![SKAB LSTM precision-recall curve](reports/figures/readme/skab_lstm_precision_recall_seed42.png)

![BATADAL LSTM confusion matrix](reports/figures/readme/batadal_lstm_confusion_matrix_seed42.png)

![BATADAL 1D-CNN confusion matrix](reports/figures/readme/batadal_cnn1d_confusion_matrix_seed42.png)

The BATADAL 1D-CNN confusion matrix shows the model never predicting the anomaly class at all — the clearest illustration of why accuracy alone is insufficient here.

LSTM's BATADAL result (`0.227 ± 0.312` F1) is better than CNN1D's but has a large standard deviation, meaning it behaves inconsistently across seeds: it captures the anomaly signal under some training conditions and misses it under others.

Overall message:

```text
Deep learning performed strongly on SKAB; on BATADAL, class imbalance and data
characteristics made it struggle. Accuracy alone is not sufficient — F1-score
and the confusion matrix must be read together.
```

Deep learning is the stronger predictor, particularly on SKAB, but it cannot explain its decisions the way the automata model can through state transitions and probabilities. In this project, deep learning represents "high predictive performance" and the automata model represents "explainable decision-making."

## Automata Results

### Fixed-Parameter Scenario Results

| Dataset | Scenario | Accuracy | Precision | Recall | F1-score | Unseen ratio |
|---|---|---:|---:|---:|---:|---:|
| BATADAL | original | 0.464 | 0.115 | 0.441 | 0.182 | 0.277 |
| BATADAL | gaussian_noise | 0.474 ± 0.013 | 0.117 ± 0.003 | 0.441 ± 0.000 | 0.185 ± 0.004 | 0.271 ± 0.005 |
| BATADAL | unseen_only | 0.290 | 0.135 | 0.636 | 0.222 | 0.274 |
| SKAB | original | 0.548 ± 0.010 | 0.230 ± 0.129 | 0.054 ± 0.050 | 0.083 ± 0.070 | 0.050 ± 0.019 |
| SKAB | gaussian_noise | 0.549 ± 0.009 | 0.235 ± 0.128 | 0.056 ± 0.046 | 0.086 ± 0.066 | 0.049 ± 0.017 |
| SKAB | unseen_only | 0.366 ± 0.121 | 0.212 ± 0.180 | 0.490 ± 0.224 | 0.276 ± 0.209 | 0.050 ± 0.020 |

`unseen_only` should not be compared directly against the full test set results. It isolates only the subset of patterns that were never seen during training, so it should be read as "how does the unseen-handling mechanism behave?" rather than "is this scenario better or worse?"

![Automata scenario F1 comparison](reports/figures/readme/automata_scenario_f1_comparison.png)

![Automata unseen ratio comparison](reports/figures/readme/automata_scenario_unseen_ratio_comparison.png)

### Noise Sensitivity

Gaussian noise does not cause a large drop in F1-score on either dataset:

| Dataset | Original F1 | Gaussian Noise F1 | Note |
|---|---:|---:|---|
| BATADAL | 0.182 | 0.185 ± 0.004 | Low-level noise did not hurt performance. |
| SKAB | 0.083 ± 0.070 | 0.086 ± 0.066 | Effect of noise stayed limited; run-to-run variance across seeds is itself larger than the noise effect. |

This is because the PAA and SAX transformations partially smooth small numeric changes at the symbolic level — a small sensor perturbation does not always flip a value to a different symbol.

### Parameter Analysis

Automata performance is highly sensitive to `window_size` and `alphabet_size`.

| Dataset | Best Window | Best Alphabet | Accuracy | Precision | Recall | F1-score | State Count | Transition Density |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BATADAL | 6 | 5 | 0.328 | 0.154 | 0.763 | 0.257 | 229.0 | 0.0045 |
| SKAB | 6 | 6 | 0.522 ± 0.047 | 0.452 ± 0.065 | 0.292 ± 0.149 | 0.337 ± 0.144 | 125.8 | 0.0096 |

Increasing `window_size` gives each state a longer symbolic context. For example, `window_size=3` produces a short pattern like `abc`, while `window_size=6` produces longer, more distinctive states. Both datasets achieve their best results at `window_size=6`.

Increasing `alphabet_size` gives SAX a finer symbolic resolution. For example, `alphabet_size=3` only uses `a, b, c`, while `alphabet_size=6` allows finer distinctions. SKAB's best result uses `alphabet_size=6`. BATADAL peaks at `alphabet_size=5`; at `6`, the state space becomes too sparse and F1 drops slightly.

Final parameter choice:

| Dataset | Preferred Window | Preferred Alphabet | Rationale |
|---|---:|---:|---|
| SKAB | 6 | 6 | Longer context and finer symbolic resolution clearly improved F1. |
| BATADAL | 6 | 5 | Longer context helped, but alphabet size 6 produced an overly sparse transition structure, so 5 was more balanced. |

![Fixed vs best automata parameter F1](reports/figures/readme/automata_fixed_vs_best_parameter_f1.png)

![SKAB automata F1 heatmap](reports/figures/readme/skab_automata_parameter_f1_heatmap.png)

![BATADAL automata F1 heatmap](reports/figures/readme/batadal_automata_parameter_f1_heatmap.png)

![Automata multiseed F1 error bars](reports/figures/readme/automata_multiseed_f1_errorbars.png)

## Explainability

The automata model's main advantage is that it produces traceable information for every decision. A deep learning model typically returns only a score or class label; the automata model can show exactly which state transition drove a given decision.

Example unseen-pattern explanation output:

```json
{
  "time_step": 10,
  "previous_state": "bccc",
  "pattern": "ccca",
  "status": "unseen",
  "mapped_to": "acca",
  "probability": 1e-06,
  "prediction": 1,
  "true_label": 0,
  "edit_distance": 1,
  "path_probability_so_far": 2.116402116402116e-27,
  "decision": "anomaly",
  "confidence": 2.116402116402116e-27,
  "decision_reason": "low_probability_path"
}
```

Field reference:

| Field | Meaning |
|---|---|
| `previous_state` | The model's previous symbolic state. |
| `pattern` | The new pattern observed at test time. |
| `status` | Whether the pattern was seen during training. |
| `mapped_to` | The closest known state, found via Levenshtein distance, for an unseen pattern. |
| `probability` | The transition probability used for this step. |
| `edit_distance` | The distance between the incoming pattern and its mapped state. |
| `path_probability_so_far` | The combined probability of the path up to this point. |
| `decision` | The normal/anomaly decision. |
| `confidence` | The probability-based confidence score. |
| `decision_reason` | The primary justification for the decision. |

In the transition heatmaps, rows represent the current state and columns represent the next state. Bright cells indicate high transition probability; large dark regions indicate a sparse transition matrix, which is expected since not every possible symbolic transition appears in the training data.

![SKAB automata transition heatmap](reports/figures/readme/skab_automata_transition_heatmap.png)

![BATADAL automata transition heatmap](reports/figures/readme/batadal_automata_transition_heatmap.png)

Transition graphs present the strongest transition relationships in a more readable form, useful for explaining which state transitions the automata model relies on and with what probability.

![SKAB automata transition graph](reports/figures/readme/skab_automata_transition_graph.png)

![BATADAL automata transition graph](reports/figures/readme/batadal_automata_transition_graph.png)

## Statistical Analysis

Model comparisons use the Wilcoxon signed-rank test.

| Dataset | Comparison | Test | Statistic | p-value | n | Significant at 0.05? |
|---|---|---|---:|---:|---:|---|
| SKAB | automata vs. LSTM F1 | Wilcoxon signed-rank | 0.000 | 0.0625 | 5 | No |
| SKAB | automata vs. 1D-CNN F1 | Wilcoxon signed-rank | 0.000 | 0.0625 | 5 | No |
| SKAB | LSTM vs. 1D-CNN F1 | Wilcoxon signed-rank | 5.000 | 5.960e-07 | 25 | Yes |
| BATADAL | LSTM vs. 1D-CNN F1 | Wilcoxon signed-rank | 0.000 | 0.1250 | 5 | No |

The LSTM vs. 1D-CNN difference on SKAB is statistically significant. The automata-vs-deep-learning comparison has a p-value of `0.0625`, which does not clear the `0.05` threshold, so it is not reported as statistically significant. With only `n=5` matched pairs, the test has limited statistical power, so this result should be read as inconclusive rather than as evidence of no difference.

McNemar's test would be suitable for comparing two models' predictions on identical samples directly. Since automata predictions operate at the PAA/SAX pattern-transition level while deep learning predictions operate at the sequence-window level, the two are not aligned sample-for-sample, so Wilcoxon signed-rank was used instead for the automata-vs-DL comparison.

## Overall Evaluation

### Deep Learning — Strengths

Deep learning models learn multivariate sensor relationships directly and, on SKAB in particular, both LSTM and 1D-CNN reach a high F1-score. If predictive performance is the priority, LSTM is the best choice for SKAB.

### Deep Learning — Limitations

Deep learning models do not naturally explain their decisions. When a model flags "anomaly," it is difficult to say directly which sensor behavior or transition pattern caused it. As seen with BATADAL, accuracy can also be misleading on imbalanced data.

### Automata — Strengths

The automata model can produce state, transition probability, unseen-pattern mapping, and a confidence score for every decision, which makes it strong on explainability:

```text
Deep learning produced a higher score; the automata model could show which
symbolic state transition, and at what probability, produced its decision.
```

### Automata — Limitations

Automata performance depends heavily on the quality of the symbolic representation. Reducing multivariate data to a single PCA component, averaging via PAA, and discretizing via SAX can all discard fine-grained sensor behavior, which is why fixed-parameter performance is low. This is exactly why the parameter sweep matters.

## Requirement Checklist

| Requirement | Status | Note |
|---|---|---|
| At least two datasets | Done | SKAB and BATADAL. |
| Official raw data | Done | Both datasets prepared from their official raw sources. |
| At least two DL models | Done | LSTM and 1D-CNN. |
| Multivariate-to-univariate reduction for automata | Done | Scaling + PCA/PC1. |
| PAA/SAX symbolic representation | Done | Used throughout the automata pipeline. |
| Sliding-window pattern extraction | Done | Patterns used as automata states. |
| Probabilistic transition model | Done | Transition probabilities computed. |
| Unseen-pattern handling | Done | Levenshtein-distance mapping to the nearest known state. |
| Original / noise / unseen scenarios | Done | Reported for the automata pipeline. |
| Window/alphabet parameter sweep | Done | Values 3, 4, 5, 6 tested. |
| Accuracy/precision/recall/F1 | Done | Reported for both DL and automata. |
| Statistical testing | Done | Wilcoxon signed-rank test. |
| Explainability output | Done | State, transition, probability, confidence, and unseen mapping reported. |

## How to Run

### Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

CPU-only PyTorch, if needed:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Reference environment used for the results above: Python 3.13.3.

### Tests

```bash
python -m pytest
```

### Automata Experiments

```bash
python -m src.experiments.run_full_automata_pipeline
```

### Deep Learning Experiments

```bash
python -m src.experiments.run_dl_experiments
python -m src.experiments.summarize_dl_results
```

### Statistical Analysis

```bash
python -m src.experiments.run_statistical_analysis
```

Training time depends on available CPU/GPU resources. Full DL training took roughly 7 minutes in the reference environment above.

## Deep Learning Training Environment and Artifact Index

The reference deep learning training run used Python 3.13.3 and took approximately 7 minutes, covering LSTM and 1D-CNN across SKAB and BATADAL with the 5-seed protocol.

Main commands:

```bash
python -m src.experiments.run_dl_experiments
python -m src.experiments.summarize_dl_results
```

Full training outputs live under `reports/results/deep_learning/`, `reports/tables/deep_learning/`, and `reports/figures/deep_learning/`. Smoke-test outputs are kept separately under `reports/results/smoke/` so they are never mistaken for final results; their purpose is only to confirm the code runs, not to measure model quality.

Figures curated for this README live in `reports/figures/readme/` and are copies of the raw experiment outputs:

| File | What it shows |
|---|---|
| `skab_lstm_confusion_matrix_seed42.png` | SKAB LSTM correct/incorrect classification breakdown. |
| `skab_lstm_precision_recall_seed42.png` | SKAB LSTM precision-recall behavior. |
| `batadal_lstm_confusion_matrix_seed42.png` | BATADAL LSTM anomaly-detection behavior. |
| `batadal_cnn1d_confusion_matrix_seed42.png` | BATADAL 1D-CNN's failure to detect any anomaly, despite high accuracy. |

## Key Output Files

| Type | File |
|---|---|
| DL summary table | `reports/tables/deep_learning/dl_summary.csv` |
| DL metrics | `reports/results/deep_learning/dl_evaluation_metrics.json` |
| DL training summary | `reports/results/deep_learning/dl_training_summary.json` |
| Automata main summary | `reports/results/automata_summary_results.csv` |
| Automata multi-seed summary | `reports/results/automata_multiseed_summary.csv` |
| Automata best parameters | `reports/results/automata_best_parameter_summary.csv` |
| Automata parameter sweep | `reports/results/automata_parameter_sweep_skab.csv`, `reports/results/automata_parameter_sweep_batadal.csv` |
| Statistical analysis | `reports/tables/statistical_analysis_summary.csv` |
| README figures | `reports/figures/readme/` |
| Smoke test outputs | `reports/results/smoke/` |

## Repository Structure

```text
.
├── config.yaml
├── requirements.txt
├── README.md
├── LICENSE
├── data/
│   └── raw/
│       ├── batadal/
│       └── skab/
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── models/
│   │   ├── automata/
│   │   ├── lstm_model.py
│   │   ├── cnn1d_model.py
│   │   └── train_deep_learning.py
│   ├── evaluation/
│   ├── experiments/
│   └── visualization/
├── tests/
└── reports/
    ├── results/
    ├── figures/
    │   └── readme/
    └── tables/
```

## Limitations and Future Work

1. Reducing multivariate sensor data to a single principal component for the automata pipeline can discard useful information.
2. PAA and SAX give an explainable symbolic structure but can smooth over brief, small anomalies.
3. BATADAL's class imbalance is severe, so accuracy alone is misleading there.
4. `unseen_only` results should not be compared directly against full-test-set results.
5. No cross-dataset transfer experiment (train on one dataset, test on the other) was performed; each dataset was analyzed independently for model behavior and generalization tendency.
6. More advanced threshold tuning, class-imbalance strategies, or a higher-dimensional symbolic representation could improve the automata model further.
7. `torch` and `tensorflow` are not pinned against a verified reproduction run in this repository's CI; only the automata-side dependencies (`numpy`, `pandas`, `scikit-learn`, `scipy`) have been confirmed to reproduce the numbers in this README exactly.

## Conclusion

Deep learning delivered stronger predictive performance in this project, particularly on SKAB, where LSTM was the strongest model. On BATADAL, class imbalance made both models struggle, and 1D-CNN's high accuracy masked its complete failure to detect anomalies.

The automata model produced a lower F1-score with fixed parameters but improved substantially under a parameter sweep. Its real contribution is that it can explain its decision process through state transitions, transition probabilities, unseen-pattern mapping, and confidence scores. This project illustrates a genuine trade-off between predictive performance and explainability.
