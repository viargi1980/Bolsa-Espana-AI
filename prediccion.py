import pandas as pd
import numpy as np
import yfinance as yf
import joblib

# ==========================================
# MODELO
# ==========================================

modelo = joblib.load(
    "modelo_compra_3pct.pkl"
)


# ==========================================
# TICKERS DESDE EL DATASET
# ==========================================

dataset = pd.read_csv(
    "acciones_espanolas_dataset.csv"
)

tickers = (
    dataset["ticker"]
    .dropna()
    .unique()
    .tolist()
)

print(
    f"Tickers encontrados: {len(tickers)}"
)

# ==========================================
# IBEX
# ==========================================

ibex = yf.download(
    "^IBEX",
    start="2010-01-01",
    auto_adjust=True,
    progress=False
)

if isinstance(ibex.columns, pd.MultiIndex):
    ibex.columns = ibex.columns.get_level_values(0)

ibex["ibex_ret_1d"] = ibex["Close"].pct_change(1)

ibex["ibex_ret_5d"] = ibex["Close"].pct_change(5)

ibex["ibex_ret_20d"] = ibex["Close"].pct_change(20)

ibex["ibex_ma50"] = (
    ibex["Close"]
    .rolling(50)
    .mean()
)

ibex["ibex_dist_ma50"] = (
    ibex["Close"]
    /
    ibex["ibex_ma50"]
    - 1
)

ibex = ibex[
    [
        "ibex_ret_1d",
        "ibex_ret_5d",
        "ibex_ret_20d",
        "ibex_dist_ma50"
    ]
]

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
# SP500
# ==========================================

sp500 = yf.download(
    "^GSPC",
    start="2010-01-01",
    auto_adjust=True,
    progress=False
)

if isinstance(sp500.columns, pd.MultiIndex):
    sp500.columns = sp500.columns.get_level_values(0)

sp500["sp_ret_1d"] = sp500["Close"].pct_change(1)
sp500["sp_ret_5d"] = sp500["Close"].pct_change(5)
sp500["sp_ret_20d"] = sp500["Close"].pct_change(20)

sp500 = sp500[
    [
        "sp_ret_1d",
        "sp_ret_5d",
        "sp_ret_20d"
    ]
]

# ==========================================
# VIX
# ==========================================

vix = yf.download(
    "^VIX",
    start="2010-01-01",
    auto_adjust=True,
    progress=False
)

if isinstance(vix.columns, pd.MultiIndex):
    vix.columns = vix.columns.get_level_values(0)

vix["vix_ret"] = (
    vix["Close"].pct_change(5)
)

vix["vix_level"] = (
    vix["Close"]
)

vix = vix[
    [
        "vix_ret",
        "vix_level"
    ]
]

# ==========================================
# CALCULAR FEATURES
# ==========================================

def crear_fila_actual(ticker):

    df = yf.download(
        ticker,
        start="2010-01-01",
        auto_adjust=True,
        progress=False
    )

    if len(df) < 260:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.join(
        ibex,
        how="left"
    )
    df = df.join(sp500, how="left")
    df = df.join(vix, how="left")


    df["ret_1d"] = df["Close"].pct_change(1)
    df["ret_3d"] = df["Close"].pct_change(3)
    df["ret_5d"] = df["Close"].pct_change(5)
    df["ret_20d"] = df["Close"].pct_change(20)

    df["vol_5d"] = (
        df["ret_1d"]
        .rolling(5)
        .std()
    )

    df["vol_20d"] = (
        df["ret_1d"]
        .rolling(20)
        .std()
    )

    df["volume_ratio"] = (
        df["Volume"]
        /
        df["Volume"].rolling(20).mean()
    )

    df["volume_spike"] = (
        df["Volume"]
        /
        df["Volume"].rolling(50).mean()
    )

    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    df["rsi"] = (
        100 - (100 / (1 + rs))
    )

    ema12 = (
        df["Close"]
        .ewm(span=12)
        .mean()
    )

    ema26 = (
        df["Close"]
        .ewm(span=26)
        .mean()
    )

    macd = ema12 - ema26

    signal = (
        macd
        .ewm(span=9)
        .mean()
    )

    df["macd_hist"] = (
        macd - signal
    )

    df["atr"] = (
        (df["High"] - df["Low"])
        .rolling(14)
        .mean()
    )

    df["atr_pct"] = (
        df["atr"]
        /
        df["Close"]
    )

    df["max20"] = (
        df["Close"]
        .rolling(20)
        .max()
    )

    df["max250"] = (
        df["Close"]
        .rolling(250)
        .max()
    )

    df["breakout20"] = (
        df["Close"]
        >= df["max20"]
    ).astype(int)

    df["dist_max250"] = (
        df["Close"]
        /
        df["max250"]
        - 1
    )

    df["gap"] = (
        df["Open"]
        /
        df["Close"].shift(1)
        - 1
    )

    df["relative_strength"] = (
        df["ret_20d"]
        -
        df["ibex_ret_20d"]
    )

    df["body_pct"] = (
        abs(df["Close"] - df["Open"])
        /
        df["Close"]
    )

    df["upper_shadow"] = (
        (
            df["High"]
            -
            df[["Open", "Close"]].max(axis=1)
        )
        /
        df["Close"]
    )

    df["lower_shadow"] = (
        (
            df[["Open", "Close"]].min(axis=1)
            -
            df["Low"]
        )
        /
        df["Close"]
    )

    df["close_position"] = (
        (df["Close"] - df["Low"])
        /
        (df["High"] - df["Low"])
    )
    # Medias

    df["ma5"] = df["Close"].rolling(5).mean()
    df["ma10"] = df["Close"].rolling(10).mean()
    df["ma20"] = df["Close"].rolling(20).mean()
    df["ma50"] = df["Close"].rolling(50).mean()
    df["ma200"] = df["Close"].rolling(200).mean()

    # Distancias

    df["dist_ma5"] = df["Close"] / df["ma5"] - 1
    df["dist_ma10"] = df["Close"] / df["ma10"] - 1
    df["dist_ma20"] = df["Close"] / df["ma20"] - 1
    df["dist_ma50"] = df["Close"] / df["ma50"] - 1
    df["dist_ma200"] = df["Close"] / df["ma200"] - 1

    # Retornos

    df["ret_10d"] = df["Close"].pct_change(10)

    # Volumen

    df["volume_trend"] = (
        df["Volume"]
        .rolling(20)
        .mean()
        .pct_change(5)
    )

    # MACD

    df["macd"] = macd

    df["macd_signal"] = signal

    # IBEX

    df["ibex_ret_5d"] = (
        df["Close"] * 0
    )
    df["ibex_ret_5d"] = ibex["ibex_ret_5d"]

    # Fuerza relativa USA

    df["relative_us"] = (
        df["ret_20d"]
        -
        df["sp_ret_20d"]
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.fillna(
        df.median(numeric_only=True)
    )

    ultima = df.iloc[-1]

    fila = pd.DataFrame(
        [ultima[features]]
    )

    return fila

# ==========================================
# PREDICCIÓN
# ==========================================

resultados = []

for ticker in tickers:

    try:

        fila = crear_fila_actual(
            ticker
        )

        if fila is None:
            continue

        prob = modelo.predict_proba(
            fila
        )[0][1]

        resultados.append(
            [
                ticker,
                prob
            ]
        )

    except Exception as e:

        print(
            ticker,
            e
        )

# ==========================================
# RANKING
# ==========================================

ranking = pd.DataFrame(
    resultados,
    columns=[
        "ticker",
        "probabilidad"
    ]
)

ranking = ranking.sort_values(
    "probabilidad",
    ascending=False
)

print()
print("==========================")
print("TOP OPORTUNIDADES HOY")
print("==========================")

print(ranking)

ranking.to_csv(
    "ranking_hoy.csv",
    index=False
)

print(
    "\nGuardado: ranking_hoy.csv"
)