# Experiment Report

Generated automatically from ClearML experiments.

## Project: Churn Prediction Experiments

## Top Models by Metrics

### Top 5 Experiments by test_f1_score

| Task Name | test_f1_score | Task ID |
|-----------|---------------|----------|
| rf_n100_d8 | 0.6332 | 2c4bae3e... |
| rf_n50_d5 | 0.6213 | 5455e20a... |
| lr_c1 | 0.6170 | 25b5055a... |
| lr_c10 | 0.6155 | a0fb73f1... |
| rf_enhanced_n300_d15 | 0.6129 | 74b05f3a... |


### Top 5 Experiments by test_accuracy

| Task Name | test_accuracy | Task ID |
|-----------|---------------|----------|
| gb_n150_d5_lr005 | 0.8041 | 7baae013... |
| gb_n50_d3_lr01 | 0.8027 | 0b6e2ce3... |
| gb_n100_d5_lr01 | 0.7949 | 5b29847e... |
| gb_enhanced_n200_d7_lr005 | 0.7906 | 652de620... |
| knn_k10 | 0.7878 | c276b00d... |


## Notes

- Reports are generated automatically from ClearML experiments
- Metrics are extracted from the last scalar metrics or task parameters
- Models are sorted by the specified metric in descending order
- Plots are downloaded from ClearML tasks and embedded in the report

## Reproducibility

To reproduce these results:

1. Ensure ClearML Server is running: `pixi run clearml-server-start`
2. Run experiments: `pixi run churn-experiments`
3. View results in ClearML UI: http://localhost:8080
