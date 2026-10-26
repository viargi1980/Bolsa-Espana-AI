import optuna
import pandas as pd
import numpy as np

from xgboost import XGBClassifier

from sklearn.metrics import roc_auc_score

# =====================
# DATASET
# =====================

dataset = pd.read_csv(
    "acciones_espanolas_dataset.csv"
)

features = [

    "dist_ma5",
    "dist_ma10",
    "dist_ma20",
    "dist_ma50",
    "dist_ma200",

    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "ret_20d",

    "vol_5d",
    "vol_20d",

    "volume_ratio",
    "volume_spike",
    "volume_trend",

    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",

    "atr_pct",

    "breakout20",
    "dist_max250",

    "relative_strength",
    "relative_us",

    "ibex_ret_1d",
    "ibex_ret_5d",
    "ibex_ret_20d",
    "ibex_dist_ma50",

    "sp_ret_1d",
    "sp_ret_5d",
    "sp_ret_20d",

    "vix_ret",
    "vix_level",

    "body_pct",
    "upper_shadow",
    "lower_shadow",
    "close_position",

    "gap"
]

X = dataset[features]
y = dataset["target_buy"]

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(
    X.median(numeric_only=True)
)

split = int(len(dataset) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

scale_pos_weight = (
    (y_train == 0).sum()
    /
    (y_train == 1).sum()
)

# =====================
# OPTUNA
# =====================

def objective(trial):

    model = XGBClassifier(scale_pos_weight=scale_pos_weight,

        n_estimators=trial.suggest_int(
            "n_estimators",
            200,
            3000
        ),

        max_depth=trial.suggest_int(
            "max_depth",
            3,
            10
        ),

        learning_rate=trial.suggest_float(
            "learning_rate",
            0.005,
            0.1,
            log=True
        ),

        subsample=trial.suggest_float(
            "subsample",
            0.5,
            1.0
        ),

        colsample_bytree=trial.suggest_float(
            "colsample_bytree",
            0.5,
            1.0
        ),

        min_child_weight=trial.suggest_int(
            "min_child_weight",
            1,
            15
        ),

        gamma=trial.suggest_float(
            "gamma",
            0,
            5
        ),

        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    proba = (
        model.predict_proba(X_test)
        [:, 1]
    )

    auc = roc_auc_score(
        y_test,
        proba
    )

    return auc

# =====================
# BUSCAR
# =====================

study = optuna.create_study(
    direction="maximize"
)

study.optimize(
    objective,
    n_trials=60
)

print()
print("====================")
print("MEJORES PARAMETROS")
print("====================")

print(
    study.best_params
)

print()
print(
    "MEJOR ROC AUC:",
    study.best_value
)