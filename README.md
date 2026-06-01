# Dataset Placement

Raw dataset files are not committed to GitHub. They must be placed manually under `data/raw`.

## SKAB

Only the following SKAB folders are used:

```txt
data/raw/skab/valve1/*.csv
data/raw/skab/valve2/*.csv
All CSV files in valve1 and valve2 are concatenated into a single dataset.

Additional metadata columns:

source_group: indicates whether the row comes from valve1 or valve2
source_file: indicates the source CSV file

These metadata columns are not used as model input.

Target column:

anomaly

Excluded from model input:

datetime
changepoint
source_group
source_file
BATADAL

Only BATADAL Training Dataset 2 is used.

Expected placement:

data/raw/batadal/training_dataset_2/*.csv

The target/label column must be checked from the actual CSV file and documented in the report.


