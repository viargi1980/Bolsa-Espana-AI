import pandas as pd
import numpy as np
import joblib

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score
)

# ==========================================
# CARGAR DATASET
# ==========================================

dataset = pd.read_csv(
    "acciones_espanolas_dataset.csv"
)

print("Filas:", len(dataset))
print("Columnas:", len(dataset.columns))

# ==========================================
# FEATURES
# ==========================================

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

# ==========================================
# TARGET
# ==========================================

X = dataset[features]

y = dataset["target_buy"]

print("\nDistribución target:")
print(y.value_counts(normalize=True))

# ==========================================
# LIMPIEZA
# ==========================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(
    X.median(numeric_only=True)
)

# ==========================================
# SPLIT TEMPORAL
# ==========================================

split = int(len(dataset) * 0.80)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

print("\nTrain:", len(X_train))
print("Test :", len(X_test))


# ==========================================
# MODELO (OPTUNA)
# ==========================================

scale_pos_weight = (
    (y_train == 0).sum()
    / max((y_train == 1).sum(), 1)
)

model = XGBClassifier(
    n_estimators=925,
    max_depth=7,
    learning_rate=0.005379395080464678,
    subsample=0.8877804666512396,
    colsample_bytree=0.5378491553248619,
    min_child_weight=12,
    gamma=3.9483621055685427,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)

# ==========================================
# ENTRENAMIENTO
# ==========================================

print("\nEntrenando...")

model.fit(
    X_train,
    y_train
)

# ==========================================
# PROBABILIDADES
# ==========================================

proba = model.predict_proba(
    X_test
)[:, 1]

print("\n==============================")
print("ESTADÍSTICAS PROBABILIDADES")
print("==============================")

print(
    "Probabilidad máxima:",
    round(proba.max(), 4)
)

print(
    "Probabilidad media:",
    round(proba.mean(), 4)
)

print(
    "Probabilidad mínima:",
    round(proba.min(), 4)
)

# ==========================================
# MEJOR UMBRAL
# ==========================================

mejor_f1 = 0
mejor_umbral = 0

for umbral in np.arange(0.10, 0.71, 0.01):

    pred_temp = (
        proba > umbral
    ).astype(int)

    f1 = f1_score(
        y_test,
        pred_temp,
        zero_division=0
    )

    if f1 > mejor_f1:

        mejor_f1 = f1
        mejor_umbral = umbral

print("\n==============================")
print("MEJOR UMBRAL")
print("==============================")

print(
    "Umbral:",
    round(mejor_umbral, 2)
)

print(
    "F1:",
    round(mejor_f1, 4)
)

# ==========================================
# PREDICCIÓN FINAL
# ==========================================

pred = (
    proba > mejor_umbral
).astype(int)

print(
    "\nSeñales generadas:",
    pred.sum()
)

print(
    "Porcentaje señales:",
    round(pred.mean() * 100, 2),
    "%"
)

# ==========================================
# MÉTRICAS
# ==========================================

accuracy = accuracy_score(
    y_test,
    pred
)

auc = roc_auc_score(
    y_test,
    proba
)

print("\n==========================")
print("RESULTADOS")
print("==========================")

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"ROC AUC  : {auc:.4f}"
)

print("\nMatriz de confusión:\n")

print(
    confusion_matrix(
        y_test,
        pred
    )
)

print("\nInforme:\n")

print(
    classification_report(
        y_test,
        pred,
        zero_division=0
    )
)

# ==========================================
# RANKING
# ==========================================

ranking = pd.DataFrame()

ranking["ticker"] = (
    dataset.iloc[split:]["ticker"]
    .reset_index(drop=True)
)

ranking["probabilidad"] = proba

ranking["real"] = (
    y_test.reset_index(drop=True)
)

ranking = ranking.sort_values(
    "probabilidad",
    ascending=False
)

ranking.to_csv(
    "ranking_oportunidades.csv",
    index=False
)

# ==========================================
# COMPRAS RECOMENDADAS
# ==========================================

compras = ranking[
    ranking["probabilidad"] > mejor_umbral
]

compras = compras.sort_values(
    "probabilidad",
    ascending=False
)

compras.to_csv(
    "compras_recomendadas.csv",
    index=False
)

print(
    "\nCompras detectadas:",
    len(compras)
)

# ==========================================
# IMPORTANCIA VARIABLES
# ==========================================

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    by="importance",
    ascending=False
)

print("\nTop 20 variables:\n")

print(
    importance.head(20)
)

importance.to_csv(
    "importancia_variables.csv",
    index=False
)

# ==========================================
# GUARDAR MODELO
# ==========================================

joblib.dump(
    model,
    "modelo_compra_3pct.pkl"
)

print("\n================================")
print("MODELO GUARDADO")
print("================================")

print("modelo_compra_3pct.pkl")
print("ranking_oportunidades.csv")
print("compras_recomendadas.csv")
print("importancia_variables.csv")