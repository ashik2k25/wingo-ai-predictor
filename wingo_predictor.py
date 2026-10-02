import streamlit as st
import numpy as np
import pandas as pd
import cloudscraper
import time
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="WinGo AI Predictor | Strongest Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS (Colorful UI) ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Poppins:wght@300;400;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #ffffff;
    }
    
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif !important;
        background: linear-gradient(90deg, #00f5ff, #ff00ff, #00ff88);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    
    .main-header {
        font-size: 2.8rem !important;
        font-weight: 900 !important;
        text-shadow: 0 0 20px rgba(0, 245, 255, 0.5);
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-family: 'Poppins', sans-serif !important;
        color: #a0a0ff !important;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(145deg, rgba(30,30,60,0.9), rgba(50,20,80,0.9));
        border: 1px solid rgba(0, 245, 255, 0.3);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0, 245, 255, 0.15);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(255, 0, 255, 0.25);
        border-color: #ff00ff;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #ff00ff22, #00f5ff22);
        border: 2px solid #00f5ff;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 0 40px rgba(0, 245, 255, 0.3);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 20px rgba(0, 245, 255, 0.3); }
        50% { box-shadow: 0 0 40px rgba(255, 0, 255, 0.5); }
    }
    
    .pred-number {
        font-size: 5rem !important;
        font-weight: 900 !important;
        font-family: 'Orbitron', sans-serif !important;
        background: linear-gradient(90deg, #00ff88, #00f5ff, #ff00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #ff00ff, #00f5ff) !important;
        color: white !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 28px !important;
        font-family: 'Orbitron', sans-serif !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(255, 0, 255, 0.4) !important;
    }
    
    .stButton > button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 6px 25px rgba(0, 245, 255, 0.6) !important;
    }
    
    div[data-testid="stMetricValue"] {
        font-family: 'Orbitron', sans-serif !important;
        color: #00f5ff !important;
    }
    
    .calc-result {
        background: linear-gradient(145deg, #1a1a3e, #2a1a4e);
        border-left: 4px solid #00ff88;
        padding: 15px 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .warning-box {
        background: linear-gradient(90deg, #ff444422, #ffaa0022);
        border: 1px solid #ffaa00;
        border-radius: 12px;
        padding: 15px;
        color: #ffcc00;
        font-size: 0.9rem;
    }
    
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #a0a0ff !important;
        font-weight: 600 !important;
    }
    
    .footer {
        text-align: center;
        color: #6666aa;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding: 20px;
        border-top: 1px solid rgba(0, 245, 255, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# ==================== API & DATA ====================
API_URL = "https://draw.ar-lottery01.com/WinGo/WinGo_1M/GetHistoryIssuePage.json"

@st.cache_data(ttl=25, show_spinner=False)
def fetch_history_data(limit=120):
    scraper = cloudscraper.create_scraper(
        browser={'browser': 'chrome', 'platform': 'android', 'mobile': True}
    )
    try:
        ts = int(time.time() * 1000)
        r = scraper.get(f"{API_URL}?ts={ts}", timeout=12)
        if r.status_code == 200:
            items = r.json().get("data", {}).get("list", [])
            numbers = [int(item["number"]) for item in reversed(items) if "number" in item]
            return numbers[-limit:] if numbers else []
    except Exception as e:
        st.error(f"Fetch Error: {e}")
    return []

# ==================== STRONGEST LIGHTWEIGHT ENGINE ====================
def create_advanced_features(numbers, seq_length=8):
    """Rich feature engineering for strong prediction"""
    X, y = [], []
    for i in range(len(numbers) - seq_length):
        seq = numbers[i:i + seq_length]
        
        features = list(seq)
        
        # Statistical features
        features.append(np.mean(seq))
        features.append(np.std(seq) if np.std(seq) > 0 else 0.1)
        features.append(np.median(seq))
        features.append(max(seq) - min(seq))
        features.append(seq[-1] - seq[0])
        features.append(seq[-1])
        features.append(seq[-1] - seq[-2] if len(seq) > 1 else 0)
        
        # Parity & modulo features
        features.append(sum(1 for x in seq if x % 2 == 0))
        features.append(sum(1 for x in seq if x >= 5))
        features.append(seq[-1] % 2)
        features.append(seq[-1] % 3)
        
        # Frequency in window
        for n in range(10):
            features.append(seq.count(n))
        
        X.append(features)
        y.append(numbers[i + seq_length])
    
    return np.array(X), np.array(y)

def build_strongest_model():
    """Ensemble of powerful tree-based models (no TensorFlow)"""
    gb = GradientBoostingClassifier(
        n_estimators=180,
        learning_rate=0.08,
        max_depth=5,
        subsample=0.85,
        random_state=42
    )
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )
    
    model = VotingClassifier(
        estimators=[('gb', gb), ('rf', rf)],
        voting='soft',
        weights=[1.3, 1.0]
    )
    return model

def train_and_predict(numbers, seq_length=8):
    if len(numbers) < seq_length + 15:
        return None, None
    
    X, y = create_advanced_features(numbers, seq_length)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = build_strongest_model()
    
    with st.spinner("🔥 Training Strongest Ensemble Engine (GradientBoosting + RandomForest)..."):
        model.fit(X_scaled, y)
    
    # Last sequence features
    last_seq = numbers[-seq_length:]
    features = list(last_seq)
    features.append(np.mean(last_seq))
    features.append(np.std(last_seq) if np.std(last_seq) > 0 else 0.1)
    features.append(np.median(last_seq))
    features.append(max(last_seq) - min(last_seq))
    features.append(last_seq[-1] - last_seq[0])
    features.append(last_seq[-1])
    features.append(last_seq[-1] - last_seq[-2] if len(last_seq) > 1 else 0)
    features.append(sum(1 for x in last_seq if x % 2 == 0))
    features.append(sum(1 for x in last_seq if x >= 5))
    features.append(last_seq[-1] % 2)
    features.append(last_seq[-1] % 3)
    for n in range(10):
        features.append(last_seq.count(n))
    
    features_scaled = scaler.transform([features])
    
    probs = model.predict_proba(features_scaled)[0]
    
    full_probs = np.zeros(10)
    for idx, cls in enumerate(model.classes_):
        full_probs[int(cls)] = probs[idx]
    
    if full_probs.sum() > 0:
        full_probs = full_probs / full_probs.sum()
    
    predicted = int(np.argmax(full_probs))
    return predicted, full_probs

# ==================== CALCULATOR ====================
def calculate_ev(prob, bet_amount, multiplier=9.0):
    win_amount = bet_amount * multiplier
    loss_amount = bet_amount
    ev = (prob * win_amount) - ((1 - prob) * loss_amount)
    roi = (ev / bet_amount) * 100 if bet_amount > 0 else 0
    return ev, roi, win_amount

# ==================== MAIN UI ====================
def main():
    st.markdown('<h1 class="main-header">🎯 WinGo AI Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">⚡ Strongest Ensemble Engine • GradientBoosting + RandomForest • Smart Calculator</p>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### ⚙️ Engine Settings")
        seq_length = st.slider("Sequence Window", 5, 12, 8, help="More history = better pattern detection")
        data_limit = st.slider("History Points", 60, 150, 100)
        
        st.markdown("---")
        st.markdown("### 💰 Calculator Settings")
        bet_amount = st.number_input("Bet Amount (₹)", min_value=1.0, value=100.0, step=10.0)
        multiplier = st.number_input("Payout Multiplier", min_value=1.0, value=9.0, step=0.5,
                                     help="Usually 9x for exact number")
        
        st.markdown("---")
        st.markdown("### 🎨 Theme")
        st.info("Neon Cyberpunk Mode Active 🔥")
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    with st.spinner("📡 Fetching live lottery data..."):
        numbers = fetch_history_data(data_limit)
    
    if len(numbers) < 25:
        st.error("❌ Not enough historical data. Need at least 25 points.")
        st.stop()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color:#00f5ff;margin:0;">📊 Total History</h4>
            <h2 style="color:#fff;margin:5px 0;">{len(numbers)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        last5 = numbers[-5:]
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color:#ff00ff;margin:0;">🔢 Last 5</h4>
            <h2 style="color:#fff;margin:5px 0;">{' - '.join(map(str, last5))}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg = np.mean(numbers[-20:])
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color:#00ff88;margin:0;">📈 Avg (20)</h4>
            <h2 style="color:#fff;margin:5px 0;">{avg:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        even_pct = sum(1 for x in numbers[-20:] if x % 2 == 0) / 20 * 100
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color:#ffaa00;margin:0;">⚖️ Even %</h4>
            <h2 style="color:#fff;margin:5px 0;">{even_pct:.0f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚀 RUN STRONGEST ENGINE", use_container_width=True, type="primary"):
        predicted, probs = train_and_predict(numbers, seq_length)
        
        if predicted is None:
            st.error("Training failed. Not enough data.")
            st.stop()
        
        st.session_state['predicted'] = predicted
        st.session_state['probs'] = probs
    
    if 'predicted' in st.session_state:
        predicted = st.session_state['predicted']
        probs = st.session_state['probs']
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        conf = probs[predicted] * 100
        st.markdown(f"""
        <div class="prediction-box">
            <h3 style="color:#a0a0ff;margin:0;">🎯 TARGET PREDICTED NUMBER</h3>
            <div class="pred-number">{predicted}</div>
            <h3 style="color:#00ff88;">Confidence: {conf:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_left, col_right = st.columns([1.2, 1])
        
        with col_left:
            st.markdown("### 📊 Probability Distribution")
            df_prob = pd.DataFrame({
                'Number': list(range(10)),
                'Probability': probs * 100
            })
            
            colors = ['#ff00ff' if i == predicted else '#00f5ff' for i in range(10)]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=df_prob['Number'],
                    y=df_prob['Probability'],
                    marker_color=colors,
                    text=[f"{p:.1f}%" for p in df_prob['Probability']],
                    textposition='outside',
                    marker_line_width=0
                )
            ])
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                xaxis=dict(title='Number (0-9)', tickmode='linear', dtick=1),
                yaxis=dict(title='Probability %', gridcolor='rgba(255,255,255,0.1)'),
                height=380,
                margin=dict(t=20, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col_right:
            st.markdown("### 🏆 Top 3 Predictions")
            top3_idx = np.argsort(probs)[::-1][:3]
            for rank, idx in enumerate(top3_idx, 1):
                pct = probs[idx] * 100
                medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
                st.markdown(f"""
                <div class="metric-card" style="margin-bottom:12px;">
                    <h3 style="margin:0;">{medal} Number {idx}</h3>
                    <h2 style="color:#00f5ff;margin:5px 0;">{pct:.2f}%</h2>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("## 💰 Smart Betting Calculator")
        
        c1, c2, c3 = st.columns(3)
        
        ev, roi, potential_win = calculate_ev(probs[predicted], bet_amount, multiplier)
        
        with c1:
            st.markdown(f"""
            <div class="calc-result">
                <h4 style="color:#00f5ff;margin:0;">💵 Potential Win</h4>
                <h2 style="color:#00ff88;margin:5px 0;">₹{potential_win:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with c2:
            color = "#00ff88" if ev > 0 else "#ff4444"
            st.markdown(f"""
            <div class="calc-result">
                <h4 style="color:#00f5ff;margin:0;">📈 Expected Value</h4>
                <h2 style="color:{color};margin:5px 0;">₹{ev:,.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with c3:
            color = "#00ff88" if roi > 0 else "#ff4444"
            st.markdown(f"""
            <div class="calc-result">
                <h4 style="color:#00f5ff;margin:0;">📊 ROI</h4>
                <h2 style="color:{color};margin:5px 0;">{roi:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        if conf >= 22:
            advice = "🟢 High confidence signal. Consider moderate stake."
        elif conf >= 15:
            advice = "🟡 Medium confidence. Use small stake only."
        else:
            advice = "🔴 Low confidence. Better to skip this round."
        
        st.markdown(f"""
        <div class="warning-box">
            <b>💡 AI Advice:</b> {advice}<br>
            <b>Risk Note:</b> Lottery is pure RNG. This model finds statistical patterns only. 
            Never bet more than you can afford to lose.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📉 Recent Number Trend")
        df_hist = pd.DataFrame({
            'Index': list(range(len(numbers[-40:]))),
            'Number': numbers[-40:]
        })
        fig2 = px.line(df_hist, x='Index', y='Number', markers=True)
        fig2.update_traces(line_color='#00f5ff', marker=dict(size=8, color='#ff00ff'))
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            yaxis=dict(dtick=1, gridcolor='rgba(255,255,255,0.1)'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            height=300,
            margin=dict(t=10)
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    else:
        st.info("👆 Click **RUN STRONGEST ENGINE** button to start prediction")
    
    st.markdown("""
    <div class="footer">
        WinGo AI Predictor v3.0 | Strongest Ensemble (GradientBoosting + RandomForest)<br>
        ⚠️ For educational & entertainment purposes only. Gambling involves risk.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
