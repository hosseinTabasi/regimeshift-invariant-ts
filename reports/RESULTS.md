# RegimeShift — Results

Aggregated from `experiments/results/*_raw.csv`. Each seed's value is first averaged over held-out regimes (leave-one-regime-out); mean ± std is then taken over seeds 0,1,2.

## FD002 — E1 unseen regime / E2 false alarms after switch / E3 regime probe

| method                                           |   auroc_unseen_mean |   auroc_unseen_std |   auroc_seen_mean |   auroc_seen_std |   far_seen_mean |   far_seen_std |   far_unseen_mean |   far_unseen_std |   tpr_unseen_mean |   tpr_unseen_std |   far_after_switch_mean |   far_after_switch_std |   regime_probe_acc_mean |   regime_probe_acc_std |
|:-------------------------------------------------|--------------------:|-------------------:|------------------:|-----------------:|----------------:|---------------:|------------------:|-----------------:|------------------:|-----------------:|------------------------:|-----------------------:|------------------------:|-----------------------:|
| Global norm + PCA-SPE                            |               0.541 |              0.01  |             0.736 |            0.002 |           0.011 |          0.001 |             0.5   |            0     |             0.511 |            0.004 |                   0.5   |                  0     |                   1     |                  0     |
| Global norm + AE                                 |               0.528 |              0.049 |             0.994 |            0.002 |           0.013 |          0.002 |             1     |            0     |             1     |            0     |                   1     |                  0     |                   1     |                  0     |
| Per-regime norm + AE                             |               0.562 |              0.044 |             0.811 |            0.008 |           0.011 |          0.001 |             1     |            0     |             1     |            0     |                   1     |                  0     |                   0.422 |                  0.003 |
| Per-regime norm + AE (transductive target stats) |               0.812 |              0.005 |             0.811 |            0.008 |           0.011 |          0.001 |             0.017 |            0.01  |             0.077 |            0.1   |                   0.017 |                  0.011 |                   0.274 |                  0.007 |
| DANN AE (GRL only)                               |               0.536 |              0.051 |             0.9   |            0.027 |           0.012 |          0.001 |             0.948 |            0.091 |             0.99  |            0.018 |                   0.948 |                  0.091 |                   1     |                  0     |
| FiLM-Inv (ours)                                  |               0.635 |              0.211 |             0.885 |            0.019 |           0.012 |          0.001 |             1     |            0     |             1     |            0     |                   1     |                  0     |                   0.872 |                  0.028 |

## FD002 — E4 misspecified number of regimes (holdouts 0,3; true = 6)

| method               |   k_regimes |   auroc_unseen_mean |   auroc_unseen_std |   far_unseen_mean |   far_unseen_std |
|:---------------------|------------:|--------------------:|-------------------:|------------------:|-----------------:|
| Per-regime norm + AE |           2 |               0.968 |              0.028 |               1   |                0 |
| FiLM-Inv (ours)      |           2 |               0.848 |              0.14  |               1   |                0 |
| Per-regime norm + AE |           3 |               0.819 |              0.114 |               0.5 |                0 |
| FiLM-Inv (ours)      |           3 |               0.886 |              0.109 |               1   |                0 |
| Per-regime norm + AE |           4 |               0.977 |              0.01  |               1   |                0 |
| FiLM-Inv (ours)      |           4 |               0.983 |              0.021 |               1   |                0 |
| Per-regime norm + AE |           6 |               0.6   |              0.112 |               1   |                0 |
| FiLM-Inv (ours)      |           6 |               0.649 |              0.382 |               1   |                0 |
| Per-regime norm + AE |           8 |               0.54  |              0.068 |               1   |                0 |
| FiLM-Inv (ours)      |           8 |               0.918 |              0.13  |               1   |                0 |
| Per-regime norm + AE |          12 |               0.559 |              0.076 |               1   |                0 |
| FiLM-Inv (ours)      |          12 |               0.835 |              0.28  |               1   |                0 |

## FD002 — E5 ablations

| method                      |   auroc_unseen_mean |   auroc_unseen_std |   auroc_seen_mean |   auroc_seen_std |   far_seen_mean |   far_seen_std |   far_unseen_mean |   far_unseen_std |   tpr_unseen_mean |   tpr_unseen_std |   far_after_switch_mean |   far_after_switch_std |   regime_probe_acc_mean |   regime_probe_acc_std |
|:----------------------------|--------------------:|-------------------:|------------------:|-----------------:|----------------:|---------------:|------------------:|-----------------:|------------------:|-----------------:|------------------------:|-----------------------:|------------------------:|-----------------------:|
| FiLM-Inv (ours)             |               0.635 |              0.211 |             0.885 |            0.019 |           0.012 |          0.001 |                 1 |                0 |             1     |            0     |                       1 |                      0 |                   0.872 |                  0.028 |
| FiLM only                   |               0.495 |              0.024 |             0.993 |            0.002 |           0.012 |          0.002 |                 1 |                0 |             1     |            0     |                       1 |                      0 |                   1     |                  0     |
| FiLM + contrastive (no GRL) |               0.573 |              0.056 |             0.968 |            0.001 |           0.012 |          0.001 |                 1 |                0 |             1     |            0     |                       1 |                      0 |                   0.594 |                  0.079 |
| FiLM + GRL (no contrastive) |               0.565 |              0.018 |             0.924 |            0.028 |           0.012 |          0.001 |                 1 |                0 |             1     |            0     |                       1 |                      0 |                   1     |                  0     |
| No FiLM (contrastive + GRL) |               0.471 |              0.164 |             0.766 |            0.019 |           0.012 |          0.002 |                 1 |                0 |             0.993 |            0.013 |                       1 |                      0 |                   0.958 |                  0.009 |

## FD004 — E1 unseen regime / E2 false alarms after switch / E3 regime probe

| method                                           |   auroc_unseen_mean |   auroc_unseen_std |   auroc_seen_mean |   auroc_seen_std |   far_seen_mean |   far_seen_std |   far_unseen_mean |   far_unseen_std |   tpr_unseen_mean |   tpr_unseen_std |   far_after_switch_mean |   far_after_switch_std |   regime_probe_acc_mean |   regime_probe_acc_std |
|:-------------------------------------------------|--------------------:|-------------------:|------------------:|-----------------:|----------------:|---------------:|------------------:|-----------------:|------------------:|-----------------:|------------------------:|-----------------------:|------------------------:|-----------------------:|
| Global norm + PCA-SPE                            |               0.612 |              0.009 |             0.761 |            0.006 |           0.006 |          0.001 |             0.5   |            0     |             0.504 |            0.003 |                   0.5   |                  0     |                   1     |                  0     |
| Global norm + AE                                 |               0.492 |              0.077 |             0.996 |            0.001 |           0.009 |          0.002 |             1     |            0     |             1     |            0     |                   1     |                  0     |                   1     |                  0     |
| Per-regime norm + AE                             |               0.684 |              0.007 |             0.835 |            0.024 |           0.013 |          0.003 |             1     |            0     |             1     |            0     |                   1     |                  0     |                   0.413 |                  0.004 |
| Per-regime norm + AE (transductive target stats) |               0.849 |              0.036 |             0.835 |            0.024 |           0.013 |          0.003 |             0.013 |            0.011 |             0.222 |            0.049 |                   0.013 |                  0.011 |                   0.261 |                  0.004 |
| DANN AE (GRL only)                               |               0.484 |              0.083 |             0.941 |            0.043 |           0.01  |          0.004 |             1     |            0     |             1     |            0     |                   1     |                  0     |                   1     |                  0     |
| FiLM-Inv (ours)                                  |               0.768 |              0.059 |             0.896 |            0.012 |           0.009 |          0.002 |             1     |            0     |             1     |            0     |                   1     |                  0     |                   0.87  |                  0.056 |
