import pandas as pd
import yfinance as yf

# ==========================================
# ACCIONES ESPAÑOLAS
# ==========================================

tickers = [

    # Bancos
    "SAN.MC",
    "BBVA.MC",
    "CABK.MC",
    "SAB.MC",
    "BKT.MC",
    "UNI.MC",

    # Energía
    "IBE.MC",
    "REP.MC",
    "ENG.MC",
    "NTGY.MC",
    "RED.MC",
    "SLR.MC",
    "SOL.MC",

    # Telecom
    "TEF.MC",

    # Construcción e infraestructuras
    "ACS.MC",
    "FER.MC",
    "OHLA.MC",

    # Industria
    "ANA.MC",
    "MTS.MC",
    "CAF.MC",
    "TLGO.MC",

    # Tecnología
    
    "AMP.MC",

    # Consumo
    "ITX.MC",
    "EBRO.MC",
    "VIS.MC",

    # Turismo
    "AENA.MC",
    "MEL.MC",
    "IAG.MC",

    # Inmobiliario
    "MRL.MC",
    "COL.MC",
    "HOME.MC",

    # Medios
    "PRS.MC",

    # Alimentación
    "EBRO.MC",

    # Farmacia / Biotech
    "GRF.MC",
    "ALM.MC",
    "FAE.MC",
    "ORY.MC",

    # Renovables
    "GRE.MC",

    # Continuo
    "DIA.MC",
    "LOG.MC",
    "VID.MC",
    "GAM.MC",
    "ENO.MC",
    "ELE.MC",

    # Growth
    "FACE.MC",
    "NEA.MC",
    "SNG.MC"
]
# ==========================================
# IBEX
# ==========================================

print("Descargando IBEX...")

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
# SP500
# ==========================================

print("Descargando SP500...")

sp500 = yf.download(
    "^GSPC",
    start="2010-01-01",
    auto_adjust=True,
    progress=False
)

if isinstance(sp500.columns, pd.MultiIndex):
    sp500.columns = sp500.columns.get_level_values(0)

sp500["sp_ret_1d"] = (
    sp500["Close"].pct_change(1)
)

sp500["sp_ret_5d"] = (
    sp500["Close"].pct_change(5)
)

sp500["sp_ret_20d"] = (
    sp500["Close"].pct_change(20)
)

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

print("Descargando VIX...")

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
# FUNCIÓN
# ==========================================

def crear_features(ticker):

    print(f"Descargando {ticker}")

    df = yf.download(
        ticker,
        start="2010-01-01",
        auto_adjust=True,
        progress=False
    )

    if len(df) == 0:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.join(
         ibex,
         how="left"
    )
    df = df.join(
        sp500,
        how="left"
    )
    df = df.join(
        vix,
        how="left"
    )


    # ==================================
    # MEDIAS
    # ==================================

    df["ma5"] = df["Close"].rolling(5).mean()
    df["ma10"] = df["Close"].rolling(10).mean()
    df["ma20"] = df["Close"].rolling(20).mean()
    df["ma50"] = df["Close"].rolling(50).mean()
    df["ma200"] = df["Close"].rolling(200).mean()

    # ==================================
    # DISTANCIAS
    # ==================================

    df["dist_ma5"] = (
        df["Close"] / df["ma5"] - 1
    )

    df["dist_ma10"] = (
        df["Close"] / df["ma10"] - 1
    )

    df["dist_ma20"] = (
        df["Close"] / df["ma20"] - 1
    )

    df["dist_ma50"] = (
        df["Close"] / df["ma50"] - 1
    )

    df["dist_ma200"] = (
        df["Close"] / df["ma200"] - 1
    )

    # ==================================
    # RETORNOS
    # ==================================

    df["ret_1d"] = df["Close"].pct_change(1)
    df["ret_3d"] = df["Close"].pct_change(3)
    df["ret_5d"] = df["Close"].pct_change(5)
    df["ret_10d"] = df["Close"].pct_change(10)
    df["ret_20d"] = df["Close"].pct_change(20)

    # ==================================
    # VOLATILIDAD
    # ==================================

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

    # ==================================
    # VOLUMEN
    # ==================================

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

    df["volume_trend"] = (
        df["Volume"]
        .rolling(20)
        .mean()
        .pct_change(5)
    )

    # ==================================
    # RSI
    # ==================================

    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    df["rsi"] = (
        100 - (100 / (1 + rs))
    )

    # ==================================
    # MACD
    # ==================================

    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()

    df["macd"] = ema12 - ema26

    df["macd_signal"] = (
        df["macd"]
        .ewm(span=9)
        .mean()
    )

    df["macd_hist"] = (
        df["macd"]
        - df["macd_signal"]
    )

    # ==================================
    # ATR
    # ==================================

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

    # ==================================
    # RUPTURAS
    # ==================================

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
        df["Close"] >= df["max20"]
    ).astype(int)

    df["dist_max250"] = (
        df["Close"]
        /
        df["max250"]
        - 1
    )

    # ==================================
    # GAP
    # ==================================

    df["gap"] = (
        df["Open"]
        /
        df["Close"].shift(1)
        - 1
    )

    # ==================================
    # FUERZA RELATIVA
    # ==================================

    df["relative_strength"] = (
        df["ret_20d"]
        -
        df["ibex_ret_20d"]
    )

    df["relative_us"] = (
        df["ret_20d"]
        -
        df["sp_ret_20d"]
    )

    # Velas
    # ==================================

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
        (df["Close"]  -     df["Low"]   )
        /
        ( df["High"]      -         df["Low"])
    ).fillna(0)
   


    # ==================================
    # TARGETS
    # ==================================

    df["target_next_day"] = (
        df["Close"].shift(-1)
        /
        df["Close"]
        - 1
    )

    df["target_up"] = (
        df["target_next_day"] > 0
    ).astype(int)

    df["target_buy"] = (
        df["target_next_day"] > 0.03
    ).astype(int)

    df["ticker"] = ticker

    df = df.dropna()

    return df

# ==========================================
# CREAR DATASET
# ==========================================

lista_df = []

for ticker in tickers:

    try:

        datos = crear_features(ticker)

        if datos is not None:

            lista_df.append(datos)

    except Exception as e:

        print(f"Error en {ticker}: {e}")

dataset = pd.concat(
    lista_df,
    ignore_index=False
)

dataset = dataset.reset_index()

# ==========================================
# ESTADÍSTICAS
# ==========================================

print("\nFilas:", len(dataset))
print("Columnas:", len(dataset.columns))

print("\nTarget subida:")
print(
    dataset["target_up"]
    .value_counts(normalize=True)
)

print("\nTarget +3%:")
print(
    dataset["target_buy"]
    .value_counts(normalize=True)
)

# ==========================================
# GUARDAR
# ==========================================

dataset.to_csv(
    "acciones_espanolas_dataset.csv",
    index=False,
    encoding="utf-8"
)

print("\nCSV generado:")
print("acciones_espanolas_dataset.csv")
