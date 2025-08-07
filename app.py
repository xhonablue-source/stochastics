# mathcraft_stochastics.py
# MathCraft: The Math of Stochastics — Randomness, Probability & Markets
# Now with Dr. X (OpenAI) + Progress Badges
import os
import time
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from math import comb

# Optional live data (falls back gracefully)
try:
    import yfinance as yf
    YF_AVAILABLE = True
except Exception:
    YF_AVAILABLE = False

# =========================
# Page Config & Branding
# =========================
st.set_page_config(
    page_title="MathCraft: The Math of Stochastics",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.concept-box{background:#f8f9ff;border:1px solid #e2e8f0;padding:1rem;border-radius:10px;margin:1rem 0}
.success-box{background:#ecfeff;border:1px solid #a5f3fc;padding:1rem;border-radius:10px;margin:1rem 0}
.warning-box{background:#fff7ed;border:1px solid #ffd7aa;padding:1rem;border-radius:10px;margin:1rem 0}
.module-card{background:#ffffff;border:1px solid #e5e7eb;border-radius:12px;padding:1rem;margin:0.5rem 0}
.kpi{background:#111827;color:#fff;border-radius:12px;padding:1rem;text-align:center}
.small{font-size:0.9rem;color:#555}
.badge{display:inline-block;padding:.35rem .6rem;border-radius:999px;border:1px solid #e5e7eb;background:#fff;margin-right:.35rem;margin-bottom:.35rem}
.badge-earned{background:#DCFCE7;border-color:#16A34A;color:#065F46}
footer {visibility:hidden}
a { text-decoration: none; }
</style>
""", unsafe_allow_html=True)

col_a, col_b = st.columns([1,4])
with col_a:
    st.markdown("")
with col_b:
    st.markdown("### www.cognitivecloud.ai")
    st.markdown("**Built by Xavier Honablue M.Ed for CognitiveCloud.ai**")

st.markdown("---")
st.title(" MathCraft: The Math of Stochastics")

st.markdown("""
Welcome! This lesson explores **randomness** — from **coin flips and binomial probability** to **random walks**, **GBM**, and a markets tie-in with the **Stochastic Oscillator**.  
Includes a friendly **Wall Street 101** intro, curated **Explore More** links, **Dr. X** coaching, and **progress badges** for motivation.
""")

# =========================
# Progress & Badges
# =========================
SECTIONS = [
    "0) Wall Street 101 (Intro)",
    "0b) Chart Anatomy Lab",
    "1) Bernoulli & Binomial",
    "2) LLN & CLT",
    "3) Random Walks",
    "4) Geometric Brownian Motion",
    "5) Stochastic Oscillator (Markets)",
    "6) Quiz & Dr. X",
    "7) Reflection",
    "🌐 Explore More (Links)"
]
BADGE_RULES = {
    "Wall Street Explorer": "Complete Wall Street 101 actions",
    "Chart Reader": "Label two parts in Chart Anatomy",
    "Binomial Builder": "Compute at least one PMF value",
    "LLN Detective": "Run LLN/CLT simulation",
    "Random Walker": "Simulate multiple random walk paths",
    "GBM Voyager": "Simulate at least 5 GBM paths",
    "Oscillator Ops": "Compute %K/%D on any data",
    "Stochastics Apprentice": "Score ≥ 4/5 on the quiz",
    "Reflective Thinker": "Submit a reflection"
}

if "completed" not in st.session_state:
    st.session_state.completed = {key: False for key in SECTIONS}
if "badges" not in st.session_state:
    st.session_state.badges = set()
if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()

def award_badge(name: str):
    if name in BADGE_RULES:
        st.session_state.badges.add(name)

def set_completed(section: str):
    st.session_state.completed[section] = True

def progress_percent():
    done = sum(1 for v in st.session_state.completed.values() if v)
    return int(100 * done / len(SECTIONS))

# Sidebar — progress
st.sidebar.header("🏅 Progress")
st.sidebar.progress(progress_percent() / 100.0)
if st.session_state.badges:
    st.sidebar.caption("Badges earned:")
    st.sidebar.write(
        " ".join([f"<span class='badge badge-earned'>🏅 {b}</span>" for b in sorted(st.session_state.badges)]),
        unsafe_allow_html=True
    )
else:
    st.sidebar.caption("No badges yet — let’s earn some!")

# =================================
# Dr. X (OpenAI) — wiring
# =================================
def _get_openai_key():
    # Prefer st.secrets, fall back to env var
    try:
        return st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        pass
    return os.environ.get("OPENAI_API_KEY")

def dr_x_feedback(user_answer, correct_answer, rationale):
    """
    Uses OpenAI if key present; otherwise returns a friendly stub.
    """
    api_key = _get_openai_key()
    # If no key, provide stub
    if not api_key:
        return (
            "Dr. X (stub):\n"
            + rationale
            + f"\n\nYour answer: **{user_answer}**\nBest answer: **{correct_answer}**\n"
            "Tip: Re-check each step and units. (Add OPENAI_API_KEY to enable live coaching.)"
        )
    # Try OpenAI call (supports both new and legacy clients)
    prompt = (
        "You are Dr. X, a friendly growth mindset coach and math tutor for grades 6–12. "
        "Explain briefly, step-by-step, with kindness. If it's numeric, show the formula and a clean calculation. "
        "If it's conceptual, define terms and contrast distractors. Keep it under 150 words.\n\n"
        f"Student answer: {user_answer}\n"
        f"Target/correct: {correct_answer}\n"
        f"Rationale/solution path: {rationale}\n"
        "Now give the student-specific guidance."
    )
    try:
        # Prefer new SDK style if available
        try:
            from openai import OpenAI  # type: ignore
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role":"system","content":"You are Dr. X, a friendly, concise math coach for teens."},
                    {"role":"user","content":prompt}
                ],
                temperature=0.4,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            import openai  # legacy
            openai.api_key = api_key
            resp = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role":"system","content":"You are Dr. X, a friendly, concise math coach for teens."},
                    {"role":"user","content":prompt}
                ],
                temperature=0.4,
            )
            return resp.choices[0].message["content"].strip()
    except Exception as e:
        return f"Dr. X error: {e}. Using stub:\n\n{rationale}"

def within_tol(user_val, correct_val, pct=0.05):
    try:
        return abs(float(user_val) - float(correct_val)) <= abs(correct_val) * pct
    except Exception:
        return False

# =================================
# Sidebar — Standards & Navigation
# =================================
st.sidebar.header("📚 Common Core Alignment")
cc = st.sidebar.multiselect(
    "Select standards to emphasize:",
    [
        "HSF-IF.B.4 (interpret key features of functions)",
        "HSF-IF.C.7 (graph functions & show key features)",
        "HSS-IC.A (understand & evaluate random processes)",
        "HSS-MD.A (expected value & probability models)"
    ],
    default=["HSS-IC.A (understand & evaluate random processes)",
             "HSS-MD.A (expected value & probability models)"]
)

page = st.sidebar.radio("Navigate", SECTIONS, index=0)

# =========================
# 0) Wall Street 101
# =========================
if page == "0) Wall Street 101 (Intro)":
    st.header("0) Wall Street 101 — for Middle & High School")
    st.markdown("""
**What is Wall Street?**  
It’s the “home base” for U.S. stock markets — places where people **buy and sell pieces of companies** (called **stocks** or **shares**).

**What is a Stock?**  
Owning a stock means you own a **tiny piece** of a company. If the company grows, your piece can become more valuable. If it struggles, it can lose value.

**Why Do Prices Change?**  
Supply vs. demand, company news, earnings, new products, interest rates, and sometimes…human emotions.

**What is a Stock Chart?**  
A picture of price over time.  
- **Line chart**: just the closing price each day.  
- **Bar/Candlestick**: shows **Open, High, Low, Close** (“OHLC”).  
- **Volume**: how many shares traded (how “busy” it was).
""")

    st.markdown("""
<div class="concept-box">
<b>Why this matters in MathCraft:</b> Price changes can look random over short periods. That leads to
<b>probability</b>, <b>statistics</b>, and <b>stochastics</b> — the math we use to describe uncertainty.
</div>
""", unsafe_allow_html=True)

    col1, col2 = st.columns([2,1])
    with col1:
        src = st.radio("Data source", ["🔁 Synthetic demo", "⬆️ Upload CSV (Date,Open,High,Low,Close,Volume)", "🌐 Fetch with yfinance)"], index=0)
        if src == "🔁 Synthetic demo":
            np.random.seed(0)
            n = 120
            base = 100 + np.cumsum(np.random.normal(0, 1, size=n))
            close = pd.Series(base).rolling(3, min_periods=1).mean()
            high = close + np.random.uniform(0.5, 1.5, size=n)
            low  = close - np.random.uniform(0.5, 1.5, size=n)
            open_ = close.shift(1).fillna(close.iloc[0])
            vol  = (np.random.uniform(1e5, 5e5, size=n)).astype(int)
            df = pd.DataFrame({"Open":open_, "High":high, "Low":low, "Close":close, "Volume":vol})
            set_completed(page)
            award_badge("Wall Street Explorer")
        elif src.startswith("⬆️"):
            file = st.file_uploader("Upload CSV", type=["csv"])
            if file:
                df = pd.read_csv(file)
                set_completed(page)
                award_badge("Wall Street Explorer")
            else:
                st.stop()
        else:
            ticker = st.text_input("Ticker (e.g., AAPL)", value="AAPL")
            period = st.selectbox("Period", ["1mo","3mo","6mo","1y"], index=1)
            if not YF_AVAILABLE:
                st.warning("`yfinance` not installed. Using synthetic demo instead.")
                np.random.seed(0)
                n = 120
                base = 100 + np.cumsum(np.random.normal(0, 1, size=n))
                close = pd.Series(base).rolling(3, min_periods=1).mean()
                high = close + np.random.uniform(0.5, 1.5, size=n)
                low  = close - np.random.uniform(0.5, 1.5, size=n)
                open_ = close.shift(1).fillna(close.iloc[0])
                vol  = (np.random.uniform(1e5, 5e5, size=n)).astype(int)
                df = pd.DataFrame({"Open":open_, "High":high, "Low":low, "Close":close, "Volume":vol})
                set_completed(page)
                award_badge("Wall Street Explorer")
            else:
                data = yf.download(ticker, period=period, interval="1d")
                if data is None or data.empty:
                    st.error("Could not fetch data. Try another ticker or use synthetic demo.")
                    st.stop()
                df = data[["Open","High","Low","Close","Volume"]].reset_index(drop=True)
                set_completed(page)
                award_badge("Wall Street Explorer")

        fig = go.Figure()
        fig.add_trace(go.Candlestick(open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="OHLC"))
        fig.update_layout(title="Candlestick (OHLC)", height=420, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("""
**How to read a candle:**  
- **Body**: open → close  
- **Wicks**: high/low extremes  
- **Green**: close above open (up day)  
- **Red**: close below open (down day)
""")
        st.info("Volume (bars) shows how many shares traded — high volume = lots of interest.")

# =========================
# 0b) Chart Anatomy Lab
# =========================
elif page == "0b) Chart Anatomy Lab":
    st.header("0b) Stock Chart Anatomy — Practice Lab")
    st.markdown("Use the toggles to label parts of a candlestick chart.")

    np.random.seed(7)
    n = 60
    base = 100 + np.cumsum(np.random.normal(0, 1, size=n))
    close = pd.Series(base).rolling(2, min_periods=1).mean()
    high = close + np.random.uniform(0.4, 1.2, size=n)
    low  = close - np.random.uniform(0.4, 1.2, size=n)
    open_ = close.shift(1).fillna(close.iloc[0])
    df = pd.DataFrame({"Open":open_, "High":high, "Low":low, "Close":close})

    show_open = st.checkbox("Label: Open", value=True)
    show_close = st.checkbox("Label: Close", value=True)
    show_high = st.checkbox("Label: High", value=False)
    show_low  = st.checkbox("Label: Low", value=False)

    if sum([show_open, show_close, show_high, show_low]) >= 2:
        set_completed(page)
        award_badge("Chart Reader")

    fig = go.Figure()
    fig.add_trace(go.Candlestick(open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="OHLC"))
    if show_open:
        fig.add_annotation(x=5, y=df.iloc[5]["Open"], text="Open", showarrow=True, arrowhead=2)
    if show_close:
        fig.add_annotation(x=10, y=df.iloc[10]["Close"], text="Close", showarrow=True, arrowhead=2)
    if show_high:
        fig.add_annotation(x=20, y=df.iloc[20]["High"], text="High", showarrow=True, arrowhead=2)
    if show_low:
        fig.add_annotation(x=30, y=df.iloc[30]["Low"], text="Low", showarrow=True, arrowhead=2)
    fig.update_layout(title="Candlestick Anatomy (Click checkboxes to label)", height=450, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# =========================
# 1) Bernoulli & Binomial
# =========================
elif page == "1) Bernoulli & Binomial":
    st.header("1) Bernoulli Trial & Binomial Distribution")
    st.markdown("""
A **Bernoulli trial** has two outcomes (success/failure) with probability **p** of success.  
**Binomial(n, p)** counts successes in **n** independent Bernoulli trials.
""")

    col1, col2 = st.columns([1,2])
    with col1:
        p = st.slider("Success probability p", 0.0, 1.0, 0.5, 0.01)
        n = st.slider("Number of trials n", 1, 100, 20, 1)
        k_vals = np.arange(0, n+1)
        pmf = np.array([comb(n, k) * (p**k) * ((1-p)**(n-k)) for k in k_vals])

        st.markdown(f"**Mean:** μ = np = **{n*p:.2f}**")
        st.markdown(f"**Variance:** σ² = np(1–p) = **{n*p*(1-p):.2f}**")

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=k_vals, y=pmf, name="Binomial PMF"))
        fig.update_layout(title=f"Binomial(n={n}, p={p:.2f}) — PMF",
                          xaxis_title="k successes", yaxis_title="P(X=k)",
                          bargap=0.2, height=420)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("🧠 Try it: Compute a binomial probability"):
        colq1, colq2 = st.columns([2,1])
        with colq1:
            kq = st.number_input("Compute P(X = k) for k =", min_value=0, max_value=int(n), value=int(n//2))
        with colq2:
            attempt = st.button("Check P(X=k)")
        if attempt:
            correct = comb(n, kq) * (p**kq) * ((1-p)**(n-kq))
            st.write(f"**Exact:** {correct:.6f}")
            set_completed(page)
            award_badge("Binomial Builder")
            st.success("Nice! Try changing n and p.")

# ==============================
# 2) Law of Large Numbers & CLT
# ==============================
elif page == "2) LLN & CLT":
    st.header("2) Law of Large Numbers (LLN) & Central Limit Theorem (CLT)")
    st.markdown("""
- **LLN:** The sample mean approaches the true mean as sample size grows.  
- **CLT:** Sample means of many i.i.d. variables become **approximately normal**, even if the base distribution isn’t.
""")

    dist = st.selectbox("Choose a base distribution:", ["Bernoulli(0.5)", "Uniform(0,1)", "Exponential(λ=1)"])
    samples = st.slider("Samples per experiment", 10, 5000, 500, 10)
    experiments = st.slider("Number of experiments", 50, 2000, 400, 50)

    rng = np.random.default_rng(42)
    if dist.startswith("Bernoulli"):
        base = rng.binomial(1, 0.5, size=(experiments, samples))
        true_mean = 0.5
    elif dist.startswith("Uniform"):
        base = rng.uniform(0, 1, size=(experiments, samples))
        true_mean = 0.5
    else:
        base = rng.exponential(1, size=(experiments, samples))
        true_mean = 1.0

    sample_means = base.mean(axis=1)

    col1, col2 = st.columns(2)
    with col1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(y=np.cumsum(base[0])/np.arange(1, samples+1),
                                  mode="lines", name="Running mean (1 path)"))
        fig1.add_hline(y=true_mean, line_dash="dot", annotation_text="True Mean")
        fig1.update_layout(title="LLN: Running Mean (Single Experiment)",
                           xaxis_title="n", yaxis_title="Mean", height=380)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=sample_means, nbinsx=40, name="Means"))
        fig2.update_layout(title="CLT: Distribution of Sample Means",
                           xaxis_title="Mean", yaxis_title="Count", height=380)
        st.plotly_chart(fig2, use_container_width=True)

    set_completed(page)
    award_badge("LLN Detective")

# =====================
# 3) Random Walks
# =====================
elif page == "3) Random Walks":
    st.header("3) Symmetric Random Walk")
    st.markdown("""
A **random walk** starts at 0 and moves +1 or −1 each step with equal probability.  
It’s the discrete backbone of continuous processes like Brownian motion.
""")
    steps = st.slider("Steps", 10, 2000, 200, 10)
    paths = st.slider("Paths to simulate", 1, 50, 5, 1)
    rng = np.random.default_rng(7)

    fig = go.Figure()
    for i in range(paths):
        increments = rng.choice([-1, 1], size=steps)
        walk = np.cumsum(increments)
        fig.add_trace(go.Scatter(x=np.arange(steps), y=walk, mode="lines", name=f"path {i+1}", opacity=0.7))
    fig.update_layout(title="Random Walk Paths", xaxis_title="Step", yaxis_title="Position", height=450)
    st.plotly_chart(fig, use_container_width=True)

    if paths >= 3:
        set_completed(page)
        award_badge("Random Walker")

    with st.expander("🧠 Question: Expected value and variance"):
        st.write("For a symmetric walk with step ±1, what is **E[X_n]** and **Var(X_n]**?")
        a1 = st.text_input("Type your answer like: E=0, Var=n", key="rw_q")
        if st.button("Check", key="rw_check"):
            correct_text = "E[X_n]=0, Var(X_n)=n"
            st.info(f"✅ **Answer:** {correct_text}")
            if a1.strip().lower().replace(" ", "") in ["e=0,var=n","e[x_n]=0,var(x_n)=n"]:
                st.success("Correct!")
            else:
                st.error("Close! Ask Dr. X to break it down.")
            if st.button("🤖 Dr. X Explain", key="rw_x"):
                st.write(dr_x_feedback(a1, correct_text,
                    "Each step has mean 0 and variance 1; sums of independent steps add means and variances."))

# =================================
# 4) Geometric Brownian Motion
# =================================
elif page == "4) Geometric Brownian Motion":
    st.header("4) Geometric Brownian Motion (GBM)")
    st.markdown(r"""
**GBM** is a common model for stock prices:  
\( dS_t = \mu S_t \, dt + \sigma S_t \, dW_t \).  
Discrete simulation (Euler–Maruyama):  
\( S_{t+\Delta t} = S_t \exp\big((\mu - \tfrac{\sigma^2}{2})\Delta t + \sigma \sqrt{\Delta t} Z\big) \).
""")

    S0 = st.number_input("Initial price S0", 1.0, 10000.0, 100.0, 1.0)
    mu = st.slider("Drift μ (annual)", -0.10, 0.30, 0.08, 0.01)
    sigma = st.slider("Volatility σ (annual)", 0.01, 1.00, 0.25, 0.01)
    T = st.slider("Years to simulate", 0.1, 5.0, 1.0, 0.1)
    N = st.slider("Steps per year", 50, 252, 100, 1)
    paths = st.slider("Paths", 1, 50, 10, 1)

    total_steps = int(N*T)
    rng = np.random.default_rng(10)
    dt = 1.0 / N

    fig = go.Figure()
    for i in range(paths):
        S = np.zeros(total_steps+1)
        S[0] = S0
        Z = rng.standard_normal(total_steps)
        for t in range(total_steps):
            S[t+1] = S[t] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z[t])
        fig.add_trace(go.Scatter(y=S, mode="lines", name=f"path {i+1}", opacity=0.7))
    fig.update_layout(title="GBM Price Paths", xaxis_title="Step", yaxis_title="Price", height=450)
    st.plotly_chart(fig, use_container_width=True)

    if paths >= 5:
        set_completed(page)
        award_badge("GBM Voyager")

# ===========================================
# 5) Stochastic Oscillator (Markets tie-in)
# ===========================================
elif page == "5) Stochastic Oscillator (Markets)":
    st.header("5) Stochastic Oscillator — Math Link")
    st.markdown(r"""
The **Stochastic Oscillator** measures where today’s close sits within the recent high-low window:  
\( \%K = 100 \times \frac{C - L_n}{H_n - L_n} \).  
It’s basically a **scaled rank** of the close over the lookback window (0–100).
""")

    st.markdown("Use synthetic prices, upload CSV, or fetch a ticker (if `yfinance` is available).")
    mode = st.radio("Data source", ["🔁 Synthetic", "⬆️ Upload CSV (Date,Close)", "🌐 yfinance"], index=0)
    lookback = st.slider("Lookback n", 5, 50, 14, 1)
    smooth = st.slider("SMA smoothing for %D", 1, 10, 3, 1)

    if mode.startswith("🔁"):
        T = 1.0; N = 252
        mu=0.10; sigma=0.25; S0=100
        rng = np.random.default_rng(22)
        Z = rng.standard_normal(N)
        dt = 1/N
        S = [S0]
        for t in range(N-1):
            S.append(S[-1]*np.exp((mu-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z[t]))
        close = pd.Series(S, name="Close")
        df = pd.DataFrame({"Close": close})
        df["High"] = df["Close"].rolling(lookback).max()
        df["Low"]  = df["Close"].rolling(lookback).min()
    elif mode.startswith("⬆️"):
        file = st.file_uploader("CSV with columns: Date, Close", type=["csv"])
        if file is None:
            st.stop()
        df = pd.read_csv(file)
        if "Close" not in df.columns:
            st.error("CSV must include a 'Close' column.")
            st.stop()
        df = df.copy()
        df["High"] = df["Close"].rolling(lookback).max()
        df["Low"]  = df["Close"].rolling(lookback).min()
    else:
        if not YF_AVAILABLE:
            st.warning("`yfinance` not installed. Switch to Synthetic or Upload CSV.")
            st.stop()
        ticker = st.text_input("Ticker (e.g., AAPL)", value="AAPL")
        period = st.selectbox("Period", ["3mo","6mo","1y","2y"], index=1)
        data = yf.download(ticker, period=period, interval="1d")
        if data is None or data.empty:
            st.error("Could not fetch data. Try another ticker or use synthetic.")
            st.stop()
        df = data[["High","Low","Close"]].reset_index(drop=True)

    df["%K"] = 100*(df["Close"] - df["Low"]) / (df["High"] - df["Low"])
    df["%D"] = df["%K"].rolling(smooth).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df["Close"], mode="lines", name="Close"))
    fig.update_layout(title="Price", height=300)
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(y=df["%K"], mode="lines", name="%K"))
    fig2.add_trace(go.Scatter(y=df["%D"], mode="lines", name="%D"))
    fig2.add_hline(y=80, line_dash="dot", annotation_text="Overbought-ish")
    fig2.add_hline(y=20, line_dash="dot", annotation_text="Oversold-ish")
    fig2.update_layout(title="Stochastic Oscillator", yaxis_title="%", height=300)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
**MathCraft note:** %K is the **normalized position** of the close within its window.  
If prices were i.i.d. within a window (toy assumption), %K acts like a **scaled order statistic**.
""")

    set_completed(page)
    award_badge("Oscillator Ops")

# ===========================
# 6) Quiz & Dr. X Coaching
# ===========================
elif page == "6) Quiz & Dr. X":
    st.header("6) Stochastics Quiz — Instant Feedback + Dr. X")
    st.markdown("Instant checks with a 5% tolerance for numeric questions.")
    st.markdown("---")

    score = 0
    total = 5

    # Q1 — Binomial mean/var
    st.subheader("Q1) Binomial mean/variance")
    st.write("For X ~ Binomial(n=40, p=0.25), enter the **mean** and **variance**.")
    q1_mean = st.number_input("Mean μ =", key="q1m", step=0.01)
    q1_var  = st.number_input("Variance σ² =", key="q1v", step=0.01)
    if st.button("Check Q1"):
        correct_mean, correct_var = 40*0.25, 40*0.25*0.75  # 10, 7.5
        st.write(f"Correct μ={correct_mean:.2f}, σ²={correct_var:.2f}")
        ok = within_tol(q1_mean, correct_mean) and within_tol(q1_var, correct_var)
        st.success("✅ Correct!") if ok else st.error("❌ Not quite. Try Dr. X.")
        st.session_state.quiz_q1_ok = ok
    if st.button("🤖 Dr. X for Q1"):
        st.info(dr_x_feedback(
            user_answer=f"μ={q1_mean}, σ²={q1_var}",
            correct_answer="μ=np, σ²=np(1−p)",
            rationale="Use μ=np and σ²=np(1−p). With n=40, p=0.25 → μ=10, σ²=7.5."
        ))

    st.markdown("---")

    # Q2 — LLN concept
    st.subheader("Q2) Law of Large Numbers")
    q2 = st.radio("Which is the best statement of LLN?",
        [
            "The sample mean equals the population mean for any sample size.",
            "The sample mean converges to the population mean as sample size grows.",
            "Any distribution becomes normal if you sample enough.",
            "Variance always decreases to zero with larger samples."
        ], key="q2")
    if st.button("Check Q2"):
        correct = "The sample mean converges to the population mean as sample size grows."
        ok = (q2 == correct)
        st.success("✅ Correct!") if ok else st.error("❌ Not quite.")
        st.session_state.quiz_q2_ok = ok
    if st.button("🤖 Dr. X for Q2"):
        st.info(dr_x_feedback(q2,
            "The sample mean converges to the population mean as sample size grows.",
            "LLN is about convergence of the sample mean; CLT is about the shape of the distribution of means."
        ))

    st.markdown("---")

    # Q3 — Random walk variance
    st.subheader("Q3) Random Walk Variance")
    st.write("For n steps with ±1 increments, **Var(X_n) = ?** (type the expression, e.g., `n`)")
    q3 = st.text_input("Var(X_n) =", key="q3")
    if st.button("Check Q3"):
        ok = (q3.strip().lower() == "n")
        st.info("**Correct expression:** Var(X_n)=n")
        st.success("✅ Correct!") if ok else st.error("❌ Not quite. Ask Dr. X if stuck.")
        st.session_state.quiz_q3_ok = ok
    if st.button("🤖 Dr. X for Q3"):
        st.info(dr_x_feedback(q3, "n",
            "Variance of a sum of independent ±1 steps (mean 0, var 1) is the sum of variances: n×1 = n."))

    st.markdown("---")

    # Q4 — GBM step
    st.subheader("Q4) GBM one-step update")
    st.write("Which formula updates S to the next step (Euler–Maruyama discretization)?")
    q4 = st.radio("", [
        "S_{t+Δt} = S_t + μΔt + σ√Δt·Z",
        "S_{t+Δt} = S_t · exp((μ − σ²/2)Δt + σ√Δt·Z)",
        "S_{t+Δt} = S_t · (1 + μ + σZ)",
        "S_{t+Δt} = μ + σZ"
    ], key="q4")
    if st.button("Check Q4"):
        correct = "S_{t+Δt} = S_t · exp((μ − σ²/2)Δt + σ√Δt·Z)"
        ok = (q4 == correct)
        st.success("✅ Correct!") if ok else st.error("❌ Not quite.")
        st.session_state.quiz_q4_ok = ok
    if st.button("🤖 Dr. X for Q4"):
        st.info(dr_x_feedback(q4,
            "S_{t+Δt} = S_t · exp((μ − σ²/2)Δt + σ√Δt·Z)",
            "GBM has lognormal increments; the Itô correction yields (μ − σ²/2)Δt in the exponent."))

    st.markdown("---")

    # Q5 — Stochastic Oscillator %K
    st.subheader("Q5) Stochastic %K")
    st.write("Given Close=94, High_n=100, Low_n=80, compute %K.")
    q5 = st.number_input("%K =", key="q5", step=0.1)
    if st.button("Check Q5"):
        correct = 100 * (94 - 80) / (100 - 80)  # 70
        st.write(f"Correct: **{correct:.1f}**")
        ok = within_tol(q5, correct, 0.01)
        st.success("✅ Correct!") if ok else st.error("❌ Not quite.")
        st.session_state.quiz_q5_ok = ok
    if st.button("🤖 Dr. X for Q5"):
        st.info(dr_x_feedback(q5, 70.0,
            "%K = 100*(C − L_n)/(H_n − L_n) = 100*(94−80)/(100−80) = 70."))

    # Tally + badge
    if st.button("🧮 Finalize Quiz Score"):
        oks = [st.session_state.get(k, False) for k in ["quiz_q1_ok","quiz_q2_ok","quiz_q3_ok","quiz_q4_ok","quiz_q5_ok"]]
        score = sum(1 for x in oks if x)
        st.session_state.quiz_score = score
        st.success(f"Your score: {score}/5")
        if score >= 4:
            award_badge("Stochastics Apprentice")
        set_completed(page)

# =================
# 7) Reflection
# =================
elif page == "7) Reflection":
    st.header("7) Reflection")
    st.markdown("""
- Where did **randomness** help you reason about uncertainty?  
- Which model (binomial, random walk, GBM, stochastic oscillator) felt most intuitive? Why?
""")
    r = st.text_area("Write a short reflection:")
    if st.button("Submit Reflection"):
        if r.strip():
            st.success("Great thinking! Keep connecting math to the world you observe.")
            set_completed(page)
            award_badge("Reflective Thinker")
            st.balloons()
        else:
            st.warning("Add a few sentences before submitting.")

# ============================
# 🌐 Explore More (Links)
# ============================
elif page == "🌐 Explore More (Links)":
    st.header("🌐 Explore More — Cool & Useful Resources for Teens")

    st.subheader("Stock Market Basics")
    st.markdown("- [Khan Academy: Stocks & Bonds](https://www.khanacademy.org/economics-finance-domain/core-finance/stock-and-bonds)")
    st.markdown("- [Investopedia: Stock Basics](https://www.investopedia.com/terms/s/stock.asp)")
    st.markdown("- [TeenVestor: Investing Education for Teens](http://www.teeninvestor.com/)")

    st.subheader("Interactive Market Tools")
    st.markdown("- [TradingView](https://www.tradingview.com/) — Free charts with drawing tools")
    st.markdown("- [Yahoo Finance](https://finance.yahoo.com/) — Prices, charts, and news")
    st.markdown("- [MarketWatch Markets](https://www.marketwatch.com/tools/markets) — Index overviews")

    st.subheader("Math in the Markets")
    st.markdown("- [Stochastic Oscillator](https://www.investopedia.com/terms/s/stochasticoscillator.asp)")
    st.markdown("- [Random Walk Theory](https://corporatefinanceinstitute.com/resources/capital-markets/random-walk-theory/)")
    st.markdown("- [Khan Academy: Probability & Statistics](https://www.khanacademy.org/math/statistics-probability)")

    set_completed(page)

st.markdown("---")
st.markdown(f"**📍 Standards Emphasized:** {', '.join(cc) if cc else 'Custom emphasis'}")
st.markdown("**© MathCraft | Built by Xavier Honablue M.Ed for CognitiveCloud.ai**")
