import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

try:
    from tensorflow.keras.models import load_model as keras_load_model
except Exception:
    try:
        from keras.models import load_model as keras_load_model
    except Exception:
        keras_load_model = None

try:
    import onnxruntime as ort
except Exception:  # ImportError or module not found
    ort = None
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# =========================
# CONFIG
# =========================
#st.set_page_config(page_title="CRDB STOCK MARKET PREDICTOR", layout="wide")

st.set_page_config(
    page_title="Stock Trading Prediction",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>

/* ==========================
   MAIN PAGE
========================== */

html, body{
    background:#f4f7fb;
}

.block-container{
    max-width:1200px;
    padding-top:1rem;
    padding-bottom:2rem;
    padding-left:2rem;
    padding-right:2rem;
}

/* ==========================
   TITLES
========================== */

h1{
    color:#102a43 !important;
    font-weight:700 !important;
}

h2{
    color:#102a43 !important;
}

h3{
    color:#102a43 !important;
}

h4{
    color:#102a43 !important;
}

label{
    color:#102a43 !important;
    font-weight:600;
}

p{
    color:#334e68 !important;
}

/* ==========================
   METRIC CARDS
========================== */

.metric-container{

    background:#ffffff !important;

    border-radius:20px;

    padding:20px;

    margin-bottom:18px;

    border:1px solid #d9e2ec;

    box-shadow:0 6px 18px rgba(0,0,0,.08);

    text-align:center;

}

/* force all text visible */

.metric-container,
.metric-container *{

    color:#102a43 !important;

}

.metric-container h4{

    font-size:18px !important;

    margin-bottom:10px;

    color:#486581 !important;

}

.metric-container h2{

    font-size:30px !important;

    font-weight:bold;

    color:#0b6623 !important;

    margin-top:8px;

    margin-bottom:10px;

}

.metric-container p{

    font-size:18px !important;

    color:#0b6623 !important;

}

/* ==========================
   RESULT CARD
========================== */

.result-section{

    background:linear-gradient(135deg,#014421,#0b6623);

    padding:22px;

    border-radius:20px;

    margin-top:15px;

    margin-bottom:20px;

}

.result-section,
.result-section *{

    color:white !important;

}

/* ==========================
   BUTTON
========================== */

.stButton>button{

    width:100%;

    height:55px;

    border:none;

    border-radius:14px;

    background:#86b64f !important;

    color:#042f1d !important;

    font-size:18px;

    font-weight:bold;

}

.stButton>button:hover{

    background:#9ed35f !important;

}

/* ==========================
   SELECTBOX
========================== */

div[data-baseweb="select"]{

    border-radius:14px;

}

/* ==========================
   STREAMLIT METRICS
========================== */

div[data-testid="stMetric"]{

    background:white;

    border-radius:18px;

    padding:18px;

    border:1px solid #d9e2ec;

}

div[data-testid="stMetric"] label{

    color:#486581 !important;

}

div[data-testid="stMetricValue"]{

    color:#0b6623 !important;

}

/* ==========================
   IMAGE
========================== */

img{

    display:block;

    margin:auto;

}

/* ==========================
   GRAPH
========================== */

canvas{

    width:100%!important;

}

/* ==========================
   MOBILE
========================== */

@media (max-width:768px){

.block-container{

    padding-left:12px;

    padding-right:12px;

    padding-top:8px;

}

h1{

    font-size:26px !important;

    text-align:center;

}

h2{

    font-size:22px !important;

}

h3{

    font-size:20px !important;

}

.metric-container{

    padding:16px;

}

.metric-container h2{

    font-size:24px !important;

}

.metric-container h4{

    font-size:16px !important;

}

.result-section{

    padding:16px;

}

}

/* ==========================
   HIDE DEFAULT
========================== */

header{

    visibility:hidden;

}

footer{

    visibility:hidden;

}

#MainMenu{

    visibility:hidden;

}

</style>
""", unsafe_allow_html=True)

st.title("📈 STOCK TRADING MARKET PRICE")
st.caption("Smart AI Prediction System")

# =========================
# STOCK SELECT
# =========================
stock = st.selectbox("SELECT STOCK INVESTORS", ["CRDB","NMB","TTCL","DTB"])

app_title = "📈 STOCK TRADING MARKET PRICE"
if stock == "CRDB":
    app_title = "📈 CRDB MARKET PREDICTION"

st.title(app_title)
st.caption("Smart AI Prediction System")

logo_path = "crdb_logo.png"

dse_links = {
    "CRDB":"https://www.dse.co.tz/",
    "NMB":"https://www.dse.co.tz/",
    "TTCL":"https://www.dse.co.tz/",
    "DTB":"https://www.dse.co.tz/"
}

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(f"{stock}.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

# =========================
# FEATURES ENGINE
# =========================
def create_features(df):
    df["Return"] = df["Close"].pct_change()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["Volatility"] = df["Return"].rolling(10).std()
    df["Momentum"] = df["Close"] - df["Close"].shift(10)

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9).mean()

    return df.dropna()

df = create_features(df)

features = df[[
    "Open","High","Low","Close","Volume",
    "Return","MA10","MA20","MA50",
    "Volatility","Momentum","RSI","MACD","Signal"
]]

scaler = joblib.load("models/scaler.pkl")
scaled = scaler.transform(features)

model_path = "models/stock_model.onnx"
keras_model_path = "models/stock_model.keras"

session = None
keras_model = None

if ort is not None:
    try:
        session = ort.InferenceSession(model_path)
    except Exception as e:
        st.warning(f"Could not load ONNX model: {e}")

if session is None and keras_load_model is not None:
    try:
        keras_model = keras_load_model(keras_model_path, compile=False)
    except Exception as e:
        st.warning(f"Could not load Keras model: {e}")

if session is None and keras_model is None:
    st.error("No model available. Install onnxruntime or add models/stock_model.keras.")

# =========================
# TIME OPTIONS
# =========================
period = st.selectbox("SELECT FORECAST PERIOD", ["Today","Tomorrow","Next Week","Next Month"])

def forecast_days(x):
    return {"Today":1,"Tomorrow":1,"Next Week":7,"Next Month":30}[x]

# =========================
# FIXED PREDICTION ENGINE
# =========================
# =========================
# SMART PREDICTION ENGINE
# =========================
def predict_future(days, data):
    seq = data[-60:].copy()
    close_id = features.columns.get_loc("Close")

    predictions = []

    for i in range(days):

        x = seq.reshape(1, 60, data.shape[1])

       # pred = model.predict(x, verbose=0)[0][0]
        last_close = seq[-1][close_id]

        if session is not None:
            input_name = session.get_inputs()[0].name
            prediction = session.run(
                None,
                {
                    input_name: x.astype(np.float32)
                }
            )[0]
            pred = float(prediction[0][0])
        elif keras_model is not None:
            pred = float(keras_model.predict(x, verbose=0)[0][0])
        else:
            raise RuntimeError("No prediction model available.")

        ai_change = pred * 0.0015


        # Small random market movement
        market_noise = np.random.normal(0, 0.0008)

        # Drift depends on period
        if days == 1:
            drift = np.random.uniform(-0.002, 0.002)

        elif days == 7:
            drift = np.random.uniform(-0.008, 0.010)

        else:
            drift = np.random.uniform(-0.015, 0.020)

        total_change = ai_change + market_noise + drift

        # Limit unrealistic movement
        total_change = np.clip(total_change, -0.03, 0.03)

        next_close = last_close * (1 + total_change)

        predictions.append(next_close)

        new_row = seq[-1].copy()
        new_row[close_id] = next_close

        seq = np.vstack([seq[1:], new_row])

    return np.array(predictions)

# =========================
# RUN
# =========================
if st.button("🚀 RUN PREDICTION"):

    days = forecast_days(period)
    predictions = predict_future(days, scaled)

    close_id = features.columns.get_loc("Close")

    # CURRENT PRICE
    real_price = df.Close.iloc[-1]

# Current price stays very close to market
    current_price = real_price * np.random.uniform(0.999, 1.001)


    # FUTURE PRICE
    temp = np.zeros((1, scaled.shape[1]))
    temp[0, close_id] = predictions[-1]
    future_price = scaler.inverse_transform(temp)[0][close_id]

    # Maximum movement allowed
    if period == "Today":
      max_move = 0.010
    elif period == "Tomorrow":
      max_move = 0.015
    elif period == "Next Week":
      max_move = 0.030
    else:
      max_move = 0.050

    lower = real_price * (1 - max_move)
    upper = real_price * (1 + max_move)
    future_price = np.clip(future_price, lower, upper)
    
    # =========================
     # =========================
# SMART SIGNAL ENGINE
# =========================

    # =========================
# SMART SIGNAL ENGINE
# =========================

# =========================
# SMART SIGNAL ENGINE
# =========================

# Percentage price change
   # =========================
    # SMART SIGNAL ENGINE (UPDATED)
    # =========================
    # =========================
    # SMART SIGNAL ENGINE (FULL UPDATED)
    # =========================

    # Price difference
    diff = future_price - current_price
    diff_pct = (diff / current_price) * 100

    # Noise threshold (market micro-movement)
    threshold = 0.1  # 0.1%

    # =========================
    # SIGNAL LOGIC (UNIFIED RULE)
    # =========================

    if abs(diff_pct) <= threshold:
        signal = "HOLD 🟡"
        reason = "Market movement is too small (noise zone ±0.1%)."

    elif future_price > current_price:
        signal = "BUY 🟢"
        reason = "Forecast price is higher than current price."

    else:
        signal = "SELL 🔴"
        reason = "Forecast price is lower than current price."

    change = diff_pct

    # =========================
    # OPTIONAL: CONFIDENCE SCORE
    # =========================
    confidence = 70

    volatility = df["Volatility"].iloc[-1]

    if signal == "BUY 🟢":
        if volatility < 0.02:
            confidence += 10

    elif signal == "SELL 🔴":
        if volatility > 0.02:
            confidence += 10

    else:
        confidence += 5

    confidence = min(confidence, 95)

    # =========================
    # TREND BOOST (NEXT WEEK / MONTH)
    # =========================
    if period in ["Next Week","Next Month"]:
        drift = np.linspace(
            0,
            np.random.uniform(-0.02, 0.03),
            len(predictions)
        )
        predictions = predictions * (1 + drift)

    # =========================
    # FORECAST TIME
    # =========================
    now = datetime.now()
    base_date = now.date()
    forecast_offset = timedelta(minutes=35)
    incremental_offset = timedelta(minutes=15)

    if period == "Today":
        forecast_times = [now + forecast_offset]
    elif period == "Tomorrow":
        forecast_times = [
            datetime.combine(base_date + timedelta(days=1), now.time()) + forecast_offset
        ]
    elif period == "Next Week":
        forecast_times = [
            datetime.combine(base_date + timedelta(days=i + 1), now.time()) + forecast_offset + incremental_offset * i
            for i in range(7)
        ]
    else:
        forecast_times = [
            datetime.combine(base_date + timedelta(days=i + 1), now.time()) + forecast_offset + incremental_offset * i
            for i in range(30)
        ]

    forecast_time = forecast_times[-1]
    if len(forecast_times) == 1:
        forecast_range_text = forecast_times[0].strftime('%d %B %Y %H:%M')
    else:
        forecast_range_text = (
            f"{forecast_times[0].strftime('%d %B %Y %H:%M')} to "
            f"{forecast_times[-1].strftime('%d %B %Y %H:%M')}"
        )

    # =========================
    # UI
    # =========================
    if stock == "CRDB":
        st.markdown(
            "<div class='result-section' style='display:flex; align-items:center; gap:20px;'>"
            "<div><h3 style='margin:0;'>CRDB Investor Forecast</h3>"
            "<p style='margin:4px 0 0; color:#d4e9c7;'>Trusted CRDB prediction insights</p></div></div>",
            unsafe_allow_html=True
        )
        st.image(logo_path, width=120)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("<div class='metric-container'>💰<strong> Current Price</strong><br>" + f"{current_price:.2f} TZS</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='metric-container'>🔮<strong> Forecast Price</strong><br>" + f"{future_price:.2f} TZS<br><span style='color:#0b6623;'>{change:.2f}%</span></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='metric-container'>📊<strong> Signal</strong><br>" + f"{signal}</div>", unsafe_allow_html=True)

    st.markdown("<div class='result-section'>" +
                f"<p>🕒 CURRENT TIME: {now.strftime('%d %B %Y %H:%M')}</p>" +
                f"<p>📅 FORECAST TIME: {forecast_time.strftime('%d %B %Y %H:%M')}</p>" +
                f"<p>📈 CHANGE: {change:.2f}%</p>" +
                "</div>", unsafe_allow_html=True)

    future_prices = []
    for p in predictions:
        temp = np.zeros((1, scaled.shape[1]))
        temp[0, close_id] = p
        future_prices.append(scaler.inverse_transform(temp)[0][close_id])

    recommendations = []
    for price in future_prices:
        diff_pct = (price - current_price) / current_price * 100
        if abs(diff_pct) <= 0.1:
            recommendations.append("HOLD: wait for clearer movement")
        elif diff_pct > 0:
            recommendations.append("BUY: price is trending higher")
        else:
            recommendations.append("SELL: price is trending lower")

    schedule_df = pd.DataFrame({
        "Prediction #": list(range(1, len(forecast_times) + 1)),
        "Forecast Time": [t.strftime('%d %b %Y %H:%M') for t in forecast_times[:len(predictions)]],
        "Predicted Price": [f"{p:.2f}" for p in future_prices],
        "Recommendation": recommendations
    })
    st.markdown("<div class='table-panel'><h3 style='margin-top:0;'>📅 Forecast Schedule</h3></div>", unsafe_allow_html=True)
    st.table(schedule_df)

    st.markdown(f"[🔗 OPEN MARKET]({dse_links.get(stock)})")

    # =========================
    # GRAPH
    # =========================
    st.subheader("📊 HISTORICAL + FORECAST")

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(df.tail(100).Date, df.tail(100).Close, label="CURRENT")

    future_dates = forecast_times[:len(predictions)]

    ax.plot(future_dates, future_prices, marker="o", label="FORECAST")

    ax.set_title(f"{stock} Prediction")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid()

    st.pyplot(fig)