# Hyperparameter tuning (D18)

## Stage 1

Rule: highest val mAP50-95; differences up to 0.02 are a tie, and a tie goes to the smaller model or image size (listed first).

| name | status | best_epoch | precision | recall | mAP50 | mAP50-95 | delta_mAP50-95 | train_minutes | decision |
|---|---|---|---|---|---|---|---|---|---|
| s1_yolo11n_1280 | ok | 141 | 0.913 | 0.862 | 0.939 | 0.757 | 0.000 | 12.900 | winner |
| s1_yolo11n_1600 | ok | 132 | 0.904 | 0.898 | 0.948 | 0.763 | 0.007 | 19.000 |  |
| s1_yolo11s_1280 | ok | 124 | 0.895 | 0.859 | 0.919 | 0.726 | -0.031 | 29.700 |  |

Winner: **s1_yolo11n_1280**.

## Stage 2

Rule: a change is adopted if it beats the baseline's val mAP50-95 by more than 0.02; all adopted changes are combined in s2_final, which must again not be worse than the baseline by more than the tie margin.

| name | status | best_epoch | precision | recall | mAP50 | mAP50-95 | delta_mAP50-95 | train_minutes | decision |
|---|---|---|---|---|---|---|---|---|---|
| s1_yolo11n_1280 | ok | 141 | 0.913 | 0.862 | 0.939 | 0.757 |  | 12.900 | baseline (stage-1 winner) |
| s2_epochs300 | ok | 282 | 0.882 | 0.917 | 0.948 | 0.765 | 0.008 | 24.900 | not adopted |
| s2_lr0.001 | ok | 124 | 0.930 | 0.907 | 0.954 | 0.778 | 0.022 | 12.600 | adopted |
| s2_lr0.005 | ok | 150 | 0.921 | 0.848 | 0.933 | 0.736 | -0.020 | 12.600 | not adopted |
| s2_freeze10 | ok | 144 | 0.925 | 0.905 | 0.965 | 0.791 | 0.035 | 10.000 | adopted |
| s2_flyer_aug | ok | 140 | 0.911 | 0.880 | 0.943 | 0.742 | -0.015 | 10.200 | not adopted |
| s2_final | ok | 90 | 0.961 | 0.900 | 0.979 | 0.809 | 0.052 | 8.200 | final |

Adopted changes: {'lr0': 0.001, 'freeze': 10}.
Final model: **s2_final** (next: tune_threshold.py).
