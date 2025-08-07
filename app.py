Here is the complete, updated Python code for the "MathCraft: The Math of Stochastics" application. The code now includes fixes for the syntax error you highlighted and incorporates all the improvements discussed previously, such as a cleaner UI with tabs, more detailed explanations, and an expanded Dr. X AI coaching feature.

```python
# mathcraft_stochastics.py
# MathCraft: The Math of Stochastics — Randomness, Probability & Markets
# Now with Dr. X (OpenAI) + Progress Badges
import os
import time
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from math import comb, sqrt
import requests

# Optional live data (falls back gracefully)
try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    
# Optional OpenAI client
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# =========================
# Page Config & Branding
# =========================
st.set_page_config(
    page_title="MathCraft: The Math of Stochastics",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for a modern look ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Roboto', sans-serif;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .concept-box{background:#e8f4fd;border-left:5px solid #3498db;padding:1.5rem;border-radius:8px;margin:1rem 0}
    .exercise-box{background:#f0f9f0;border-left:5px solid #27ae60;padding:1.5rem;border-radius:8px;margin:1rem 0}
    .warning-box{background:#fff8e1;border-left:5px solid #f39c12;padding:1.5rem;border-radius:8px;margin:1rem 0}
    .module-card{background:#ffffff;border:1px solid #e5e7eb;border-radius:12px;padding:1.5rem;margin:0.5rem 0;box-shadow: 0 4px 6px rgba(0,0,0,0.1)}
    .kpi{background:#111827;color:#fff;border-radius:12px;padding:1rem;text-align:center}
    .badge{display:inline-flex;align-items:center;padding:.5rem 1rem;border-radius:999px;border:1px solid #e5e7eb;background:#fff;margin-right:.5rem;margin-bottom:.5rem;font-weight:bold}
    .badge-earned{background:#d1f7e0;border-color:#2ecc71;color:#27ae60;font-weight:bold;animation: pulse 1.5s infinite;}
    
    @keyframes pulse {
      0% {transform: scale(1);}
      50% {transform: scale(1.05);}
      100% {transform: scale(1);}
    }
    
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-weight: bold;
    }

    footer {visibility:hidden}
    a { text-decoration: none; }
</style>
""", unsafe_allow_html=True)

# --- Header Section ---
col_a, col_b = st.columns([1, 4])
with col_a:
    st.image("https://www.cognitivecloud.ai/images/logo.png", width=60)
with col_b:
    st.markdown("<div class='main-header'>🎲 MathCraft: The Math of Stochastics</div>", unsafe_allow_html=True)
    st.markdown("### A lesson on randomness, probability, and markets")
    st.markdown("Built for **CognitiveCloud.ai** by Xavier Honablue M.Ed.")
    
st.markdown("---")

st.markdown("""
Welcome! This lesson explores **randomness** — from **coin flips and binomial probability** to **random walks**, **Geometric Brownian Motion (GBM)**, and a markets tie-in with the **Stochastic Oscillator**.  
Includes a friendly **Wall Street 101** intro, curated **Explore More** links, **Dr. X** coaching, and **progress badges** for motivation.
""")
# =========================
# Progress & Badges
# =========================
SECTIONS = {
    "Wall Street 101": "Wall Street 101 (Intro)",
    "Chart Anatomy": "Chart Anatomy Lab",
    "Binomial": "Bernoulli & Binomial",
    "LLN & CLT": "LLN & CLT",
    "Random Walks": "Random Walks",
    "GBM": "Geometric Brownian Motion",
    "Stochastic": "Stochastic Oscillator",
    "Quiz": "Quiz & Dr. X",
    "Reflection": "Reflection",
    "Resources": "Explore More (Links)"
}
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
BADGE_ICONS = {
    "Wall Street Explorer": "🏢",
    "Chart Reader": "📈",
    "Binomial Builder": "🎯",
    "LLN Detective": "🔎",
    "Random Walker": "🚶‍♂️",
    "GBM Voyager": "🚀",
    "Oscillator Ops": "📊",
    "Stochastics Apprentice": "🎓",
    "Reflective Thinker": "💡"
}

if "completed" not in st.session_state:
    st.session_state.completed = {key: False for key in SECTIONS.values()}
if "badges" not in st.session_state:
    st.session_state.badges = set()
if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()
if "data" not in st.session_state:
    st.session_state.data = None

def award_badge(name: str):
    if name in BADGE_RULES and name not in st.session_state.badges:
        st.session_state.badges.add(name)
        st.toast(f"**🏅 Badge Unlocked:** {name}!", icon="🏅")

def set_completed(section: str):
    st.session_state.completed[section] = True

def progress_percent():
    done = sum(1 for v in st.session_state.completed.values() if v)
    return int(100 * done / len(SECTIONS))

# Sidebar — progress
st.sidebar.header("🏅 Your Progress")
progress = progress_percent()
st.sidebar.progress(progress / 100.0, f"{progress}% Complete")
if st.session_state.badges:
    st.sidebar.caption("Badges earned:")
    badges_html = " ".join([
        f"<span class='badge badge-earned'> {BADGE_ICONS.get(b,'🏅')} {b}</span>" for b in sorted(st.session_state.badges)
    ])
    st.sidebar.markdown(badges_html, unsafe_allow_html=True)
else:
    st.sidebar.caption("No badges yet — let’s earn some!")

# =================================
# Dr. X (OpenAI) — wiring
# =================================
def _get_openai_key():
    try:
        return st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        pass
    return os.environ.get("OPENAI_API_KEY")

def dr_x_feedback(user_answer, correct_answer, rationale, prompt_prefix=""):
    api_key = _get_openai_key()
    if not api_key:
        return (
            "**Dr. X (Offline):**\n"
            + rationale
            + f"\n\nYour answer: **{user_answer}**\nBest answer: **{correct_answer}**\n"
            "Tip: Add your `OPENAI_API_KEY` to `.streamlit/secrets.toml` to enable live coaching."
        )

    prompt = (
        "You are Dr. X, a friendly growth mindset coach and math tutor for grades 6–12. "
        "Explain briefly, step-by-step, with kindness. If it's numeric, show the formula and a clean calculation. "
        "If it's conceptual, define terms and contrast distractors. Keep it under 150 words.\n\n"
        f"{prompt_prefix}\n\n"
        f"Student answer: {user_answer}\n"
        f"Target/correct: {correct_answer}\n"
        f"Rationale/solution path: {rationale}\n"
        "Now give the student-specific guidance."
    )
    
    # Store messages for a more conversational experience if needed
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Dr. X is thinking..."):
        try:
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":"You are Dr. X, a friendly, concise math coach for teens."}] + st.session_state.messages,
                temperature=0.4,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"Dr. X error: {e}. Using stub:\n\n{rationale}"

def within_tol(user_val, correct_val, pct=0.05):
    try:
        return abs(float(user_val) - float(correct_val)) <= abs(correct_val) * pct
    except (ValueError, TypeError):
        return False

# =================================
# Module-specific functions
# =================================

# --- Wall Street 101 ---
def wall_street_101():
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
    
    st.divider()

    col1, col2 = st.columns([2, 1])
    with col1:
        src = st.radio("Choose a data source:", ["🔁 Synthetic demo", "⬆️ Upload CSV", "🌐 Fetch with yfinance"], index=0, key="ws_data_source")
        
        df = None
        if src == "🔁 Synthetic demo":
            np.random.seed(0)
            n = 120
            base = 100 + np.cumsum(np.random.normal(0, 1, size=n))
            close = pd.Series(base).rolling(3, min_periods=1).mean()
            high = close + np.random.uniform(0.5, 1.5, size=n)
            low = close - np.random.uniform(0.5, 1.5, size=n)
            open_ = close.shift(1).fillna(close.iloc[0])
            vol = (np.random.uniform(1e5, 5e5, size=n)).astype(int)
            df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol})
            set_completed(SECTIONS["Wall Street 101"])
            award_badge("Wall Street Explorer")
            st.session_state.data = df
        elif src.startswith("⬆️"):
            file = st.file_uploader("Upload CSV (Date,Open,High,Low,Close,Volume)", type=["csv"], key="ws_upload")
            if file:
                df = pd.read_csv(file)
                st.session_state.data = df
                set_completed(SECTIONS["Wall Street 101"])
                award_badge("Wall Street Explorer")
            else:
                st.info("Upload a CSV file to view the chart.")
                st.stop()
        else:
            if not YF_AVAILABLE:
                st.warning("`yfinance` is not installed. Falling back to synthetic demo.")
                df = st.session_state.data
                if df is None:
                    np.random.seed(0)
                    n = 120
                    base = 100 + np.cumsum(np.random.normal(0, 1, size=n))
                    close = pd.Series(base).rolling(3, min_periods=1).mean()
                    high = close + np.random.uniform(0.5, 1.5, size=n)
                    low = close - np.random.uniform(0.5, 1.5, size=n)
                    open_ = close.shift(1).fillna(close.iloc[0])
                    vol = (np.random.uniform(1e5, 5e5, size=n)).astype(int)
                    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol})
                    set_completed(SECTIONS["Wall Street 101"])
                    award_badge("Wall Street Explorer")
            else:
                ticker = st.text_input("Ticker (e.g., AAPL)", value="AAPL", key="ws_ticker")
                period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y"], index=1, key="ws_period")
                if st.button("Fetch Data"):
                    with st.spinner("Fetching data..."):
                        data = yf.download(ticker, period=period, interval="1d")
                        if data is None or data.empty:
                            st.error("Could not fetch data. Try another ticker or use a different source.")
                            st.stop()
                        df = data[["Open", "High", "Low", "Close", "Volume"]].reset_index(drop=True)
                        st.session_state.data = df
                        set_completed(SECTIONS["Wall Street 101"])
                        award_badge("Wall Street Explorer")
                if st.session_state.data is None:
                    st.info("Enter a ticker and click 'Fetch Data'.")
                    st.stop()
                df = st.session_state.data
        
        if df is not None:
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
        
# --- Chart Anatomy Lab ---
def chart_anatomy():
    st.header("0b) Stock Chart Anatomy — Practice Lab")
    st.markdown("Use the toggles to label parts of a candlestick chart.")
    
    np.random.seed(7)
    n = 60
    base = 100 + np.cumsum(np.random.normal(0, 1, size=n))
    close = pd.Series(base).rolling(2, min_periods=1).mean()
    high = close + np.random.uniform(0.4, 1.2, size=n)
    low = close - np.random.uniform(0.4, 1.2, size=n)
    open_ = close.shift(1).fillna(close.iloc[0])
    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close})
    
    st.divider()

    show_open = st.checkbox("Label: Open", value=True)
    show_close = st.checkbox("Label: Close", value=True)
    show_high = st.checkbox("Label: High", value=False)
    show_low = st.checkbox("Label: Low", value=False)

    labeled_count = sum([show_open, show_close, show_high, show_low])
    if labeled_count >= 2:
        set_completed(SECTIONS["Chart Anatomy"])
        award_badge("Chart Reader")

    fig = go.Figure()
    fig.add_trace(go.Candlestick(open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="OHLC"))
    if show_open:
        fig.add_annotation(x=5, y=df.iloc[5]["Open"], text="Open", showarrow=True, arrowhead=2, yshift=10)
    if show_close:
        fig.add_annotation(x=10, y=df.iloc[10]["Close"], text="Close", showarrow=True, arrowhead=2, yshift=-10)
    if show_high:
        fig.add_annotation(x=20, y=df.iloc[20]["High"], text="High", showarrow=True, arrowhead=2, yshift=10)
    if show_low:
        fig.add_annotation(x=30, y=df.iloc[30]["Low"], text="Low", showarrow=True, arrowhead=2, yshift=-10)
    fig.update_layout(title="Candlestick Anatomy", height=450, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("🤔 Ask Dr. X about chart anatomy..."):
        prompt = st.text_area("What's your question about stock charts?", key="anatomy_x_prompt")
        if st.button("Ask Dr. X", key="anatomy_x_button"):
            response = dr_x_feedback(prompt, "N/A", "A stock chart is a visual representation of price data over time. Candlesticks show the open, high, low, and close prices for a specific period.")
            st.info(response)

# --- Bernoulli & Binomial ---
def binomial_distribution():
    st.header("1) Bernoulli Trial & Binomial Distribution")
    st.markdown("""
    A **Bernoulli trial** is a single experiment with two outcomes: "success" or "failure".
    - The probability of success is **p**.
    - The probability of failure is **(1-p)**.
    
    The **Binomial distribution** is the count of successes in **n** independent Bernoulli trials.
    """)
    st.divider()
    
    col1, col2 = st.columns([1, 2])
    with col1:
        p = st.slider("Success probability p", 0.0, 1.0, 0.5, 0.01)
        n = st.slider("Number of trials n", 1, 100, 20, 1)
        
        k_vals = np.arange(0, n + 1)
        pmf = np.array([comb(n, k) * (p**k) * ((1 - p)**(n - k)) for k in k_vals])
        
        st.markdown(f"**Mean:** \(μ = np = {n * p:.2f}\)")
        st.markdown(f"**Variance:** \(σ² = np(1-p) = {n * p * (1 - p):.2f}\)")
        
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=k_vals, y=pmf, name="Binomial PMF"))
        fig.update_layout(title=f"Binomial(n={n}, p={p:.2f}) — Probability Mass Function",
                          xaxis_title="k successes", yaxis_title="P(X=k)",
                          bargap=0.2, height=420)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("🧠 Exercise: Compute a binomial probability"):
        st.markdown("Use the formula \(P(X=k) = C(n, k) \cdot p^k \cdot (1-p)^{n-k}\) to compute a value.")
        colq1, colq2 = st.columns([2, 1])
        with colq1:
            kq = st.number_input("Compute P(X = k) for k =", min_value=0, max_value=int(n), value=int(n // 2), key="binom_k")
            user_prob = st.number_input(f"Your answer for P(X={kq})", format="%.6f", step=0.0001, key="binom_prob")
        with colq2:
            st.markdown("---")
            check_button = st.button("Check Answer", key="binom_check")
            if check_button:
                correct = comb(n, kq) * (p**kq) * ((1 - p)**(n - kq))
                if within_tol(user_prob, correct, pct=0.01):
                    st.success("✅ Correct! You've successfully computed a binomial probability.")
                    award_badge("Binomial Builder")
                else:
                    st.error(f"❌ Not quite. The correct value is approximately {correct:.6f}.")
                    st.button("🤖 Dr. X Explain", key="binom_x", on_click=lambda: st.info(
                        dr_x_feedback(user_prob, correct, "The formula is C(n,k) * p^k * (1-p)^(n-k). Plug in your values for n, p, and k to get the probability.")
                    ))
        set_completed(SECTIONS["Binomial"])

# --- LLN & CLT ---
def lln_clt():
    st.header("2) Law of Large Numbers (LLN) & Central Limit Theorem (CLT)")
    st.markdown("""
    - **Law of Large Numbers (LLN):** As you run an experiment many, many times, the average of your results gets closer and closer to the true, theoretical average.
    - **Central Limit Theorem (CLT):** If you take many, many random samples from *any* distribution (even a strange-looking one), the distribution of the *sample means* will look like a bell curve (a normal distribution).
    """)
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        dist = st.selectbox("Choose a base distribution:", ["Bernoulli(0.5)", "Uniform(0,1)", "Exponential(λ=1)"], key="clt_dist")
        samples = st.slider("Samples per experiment", 10, 5000, 500, 10, key="clt_samples")
        experiments = st.slider("Number of experiments", 50, 2000, 400, 50, key="clt_experiments")
    
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

    with col2:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(y=np.cumsum(base[0]) / np.arange(1, samples + 1),
                                  mode="lines", name="Running mean (1 path)"))
        fig1.add_hline(y=true_mean, line_dash="dot", annotation_text="True Mean")
        fig1.update_layout(title="LLN: Running Mean (Single Experiment)",
                           xaxis_title="n", yaxis_title="Mean", height=380)
        st.plotly_chart(fig1, use_container_width=True)
    
    st.divider()
    
    col3, col4 = st.columns(2)
    with col3:
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=sample_means, nbinsx=40, name="Means"))
        fig2.update_layout(title="CLT: Distribution of Sample Means",
                           xaxis_title="Mean", yaxis_title="Count", height=380)
        st.plotly_chart(fig2, use_container_width=True)
        set_completed(SECTIONS["LLN & CLT"])
        award_badge("LLN Detective")
    
    with col4:
        st.markdown("### Why this matters:")
        st.markdown("""
        - **LLN** is why a casino can predict its long-term profit even though any single game is random. The average outcome of millions of bets will be very close to the theoretical expected value.
        - **CLT** is a cornerstone of statistics. It's why many statistical tests work, and why we can use a normal distribution to approximate things like poll results or manufacturing errors, even if the underlying individual events are not normal.
        """)
        if st.button("🤖 Ask Dr. X about LLN/CLT"):
            response = dr_x_feedback("", "LLN vs CLT", "LLN says sample mean approaches population mean. CLT says the distribution of many sample means is normal.")
            st.info(response)

# --- Random Walks ---
def random_walks():
    st.header("3) Symmetric Random Walk")
    st.markdown("""
    A **random walk** is a process that describes a path consisting of a succession of random steps.
    - Each step is independent of the previous ones.
    - In our case, the step is either `+1` (move up) or `-1` (move down) with equal probability.
    """)
    st.divider()

    col1, col2 = st.columns([1, 2])
    with col1:
        steps = st.slider("Number of steps", 10, 2000, 200, 10, key="rw_steps")
        paths = st.slider("Paths to simulate", 1, 50, 5, 1, key="rw_paths")
        if paths >= 3:
            set_completed(SECTIONS["Random Walks"])
            award_badge("Random Walker")
    
    rng = np.random.default_rng(7)
    fig = go.Figure()
    
    with col2:
        for i in range(paths):
            increments = rng.choice([-1, 1], size=steps)
            walk = np.cumsum(increments)
            fig.add_trace(go.Scatter(x=np.arange(steps), y=walk, mode="lines", name=f"path {i+1}", opacity=0.7))
        fig.update_layout(title="Random Walk Paths", xaxis_title="Step", yaxis_title="Position", height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("🧠 Question: Expected value and variance"):
        st.write("For a symmetric walk with step ±1, what is **E[X_n]** and **Var(X_n]**?")
        st.markdown(r"Remember, $E[X_n] = \sum E[S_i]$ and $Var(X_n) = \sum Var(S_i)$ for independent steps $S_i$.")
        a1 = st.text_input("Type your answer like: E=0, Var=n", key="rw_q")
        if st.button("Check", key="rw_check"):
            correct_text = "E[X_n]=0, Var(X_n)=n"
            if a1.strip().lower().replace(" ", "") in ["e=0,var=n", "e[x_n]=0,var(x_n)=n"]:
                st.success("✅ Correct! The expected position is 0, and the variance grows linearly with the number of steps.")
            else:
                st.error("❌ Not quite. The expectation for a symmetric walk is simple. Think about the variance of each step.")
            if st.button("🤖 Dr. X Explain", key="rw_x"):
                st.info(dr_x_feedback(a1, correct_text, "Each step has mean 0 and variance 1; sums of independent steps add means and variances."))

# --- Geometric Brownian Motion ---
def gbm():
    st.header("4) Geometric Brownian Motion (GBM)")
    st.markdown(r"""
    **GBM** is a continuous-time stochastic process often used to model stock prices. Unlike a random walk, it is based on a log-normal distribution, meaning the price changes are multiplicative, not additive.
    
    The equation is: $dS_t = \mu S_t \, dt + \sigma S_t \, dW_t$.
    
    We simulate this using a discrete form (Euler–Maruyama method):
    $S_{t+\Delta t} = S_t \exp\big((\mu - \tfrac{\sigma^2}{2})\Delta t + \sigma \sqrt{\Delta t} Z\big)$,
    where \(S_t\) is the price at time \(t\), \(\mu\) is the drift, \(\sigma\) is the volatility, \(\Delta t\) is the time step, and \(Z\) is a random number from a standard normal distribution.
    """)
    st.divider()

    col1, col2 = st.columns([1, 2])
    with col1:
        S0 = st.number_input("Initial price S0", 1.0, 10000.0, 100.0, 1.0)
        mu = st.slider("Drift μ (annual)", -0.10, 0.30, 0.08, 0.01)
        sigma = st.slider("Volatility σ (annual)", 0.01, 1.00, 0.25, 0.01)
        T = st.slider("Years to simulate", 0.1, 5.0, 1.0, 0.1)
        N = st.slider("Steps per year", 50, 252, 100, 1)
        paths = st.slider("Paths", 1, 50, 10, 1)
        
        if paths >= 5:
            set_completed(SECTIONS["GBM"])
            award_badge("GBM Voyager")
    
    total_steps = int(N * T)
    rng = np.random.default_rng(10)
    dt = 1.0 / N
    
    with col2:
        fig = go.Figure()
        for i in range(paths):
            S = np.zeros(total_steps + 1)
            S[0] = S0
            Z = rng.standard_normal(total_steps)
            for t in range(total_steps):
                S[t+1] = S[t] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[t])
            fig.add_trace(go.Scatter(y=S, mode="lines", name=f"path {i+1}", opacity=0.7))
        fig.update_layout(title="GBM Price Paths", xaxis_title="Step", yaxis_title="Price", height=450)
        st.plotly_chart(fig, use_container_width=True)

# --- Stochastic Oscillator ---
def stochastic_oscillator():
    st.header("5) Stochastic Oscillator — Math Link")
    st.markdown(r"""
    The **Stochastic Oscillator** is a momentum indicator that compares a particular closing price to its price range over a certain period. The core idea is that in an uptrend, prices should close near the high for the period, and in a downtrend, near the low.
    
    The primary line, %K, is calculated as:
    $ \%K = 100 \times \frac{C - L_n}{H_n - L_n} $
    
    - **C**: The most recent closing price.
    - **$L_n$**: The lowest low over the last $n$ periods.
    - **$H_n$**: The highest high over the last $n$ periods.
    
    Think of it as the **normalized position** of the current price within its recent range, scaled from 0 to 100.
    """)
    st.divider()

    col1, col2 = st.columns([1, 2])
    with col1:
        mode = st.radio("Choose a data source:", ["🔁 Synthetic", "⬆️ Upload CSV (Date,Close)", "🌐 yfinance"], index=0, key="stoch_mode")
        lookback = st.slider("Lookback n", 5, 50, 14, 1)
        smooth = st.slider("SMA smoothing for %D", 1, 10, 3, 1)
    
    df = None
    if mode.startswith("🔁"):
        T = 1.0; N = 252
        mu = 0.10; sigma = 0.25; S0 = 100
        rng = np.random.default_rng(22)
        Z = rng.standard_normal(N)
        dt = 1/N
        S = [S0]
        for t in range(N - 1):
            S.append(S[-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[t]))
        close = pd.Series(S, name="Close")
        df = pd.DataFrame({"Close": close})
        df["High"] = df["Close"].rolling(lookback).max()
        df["Low"] = df["Close"].rolling(lookback).min()
    elif mode.startswith("⬆️"):
        file = st.file_uploader("CSV with columns: Date, Close, High, Low (optional)", type=["csv"], key="stoch_upload")
        if file is None:
            st.info("Upload a CSV file to continue.")
            st.stop()
        df = pd.read_csv(file)
        if "Close" not in df.columns:
            st.error("CSV must include a 'Close' column.")
            st.stop()
        if "High" not in df.columns or "Low" not in df.columns:
            st.warning("High and Low columns not found. Calculating them from the Close price.")
            df["High"] = df["Close"].rolling(lookback).max()
            df["Low"] = df["Close"].rolling(lookback).min()
    else:
        if not YF_AVAILABLE:
            st.warning("`yfinance` not installed. Switch to Synthetic or Upload CSV.")
            st.stop()
        ticker = st.text_input("Ticker (e.g., AAPL)", value="AAPL", key="stoch_ticker")
        period = st.selectbox("Period", ["3mo", "6mo", "1y", "2y"], index=1, key="stoch_period")
        if st.button("Fetch Data", key="stoch_fetch"):
            with st.spinner("Fetching data..."):
                data = yf.download(ticker, period=period, interval="1d")
                if data is None or data.empty:
                    st.error("Could not fetch data. Try another ticker or use synthetic.")
                    st.stop()
                df = data[["High", "Low", "Close"]].reset_index(drop=True)

    if df is not None:
        df["%K"] = 100 * (df["Close"] - df["Low"]) / (df["High"] - df["Low"])
        df["%D"] = df["%K"].rolling(smooth).mean()

        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=df["Close"], mode="lines", name="Close"))
            fig.update_layout(title="Price", height=300)
            st.plotly_chart(fig, use_container_width=True)

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(y=df["%K"], mode="lines", name="%K", line=dict(color="#1f77b4")))
            fig2.add_trace(go.Scatter(y=df["%D"], mode="lines", name="%D", line=dict(color="#ff7f0e")))
            fig2.add_hline(y=80, line_dash="dash", annotation_text="Overbought", annotation_position="top left")
            fig2.add_hline(y=20, line_dash="dash", annotation_text="Oversold", annotation_position="bottom left")
            fig2.update_layout(title="Stochastic Oscillator", yaxis_title="%", height=300)
            st.plotly_chart(fig2, use_container_width=True)
            
        set_completed(SECTIONS["Stochastic"])
        award_badge("Oscillator Ops")
    
    with st.expander("🤔 Ask Dr. X about the Stochastic Oscillator..."):
        prompt = st.text_area("What's your question about this indicator?", key="stoch_x_prompt")
        if st.button("Ask Dr. X", key="stoch_x_button"):
            response = dr_x_feedback(prompt, "N/A", "The Stochastic Oscillator measures where the close price is relative to the high-low range of the last N periods. It is a momentum indicator.")
            st.info(response)

# --- Quiz & Dr. X Coaching ---
def quiz():
    st.header("6) Stochastics Quiz — Instant Feedback + Dr. X")
    st.markdown("Test your knowledge. Dr. X is here to help with hints if you get stuck!")
    st.markdown("---")
    
    q1_ok = False; q2_ok = False; q3_ok = False; q4_ok = False; q5_ok = False

    # Q1 — Binomial mean/var
    with st.container(border=True):
        st.subheader("Q1) Binomial mean/variance")
        st.write("For X ~ Binomial(n=40, p=0.25), enter the **mean** and **variance**.")
        col_q1a, col_q1b = st.columns(2)
        with col_q1a:
            q1_mean = st.number_input("Mean μ =", key="q1m", step=0.01)
        with col_q1b:
            q1_var = st.number_input("Variance σ² =", key="q1v", step=0.01)

        if st.button("Check Q1", key="q1_check"):
            correct_mean, correct_var = 40 * 0.25, 40 * 0.25 * 0.75  # 10, 7.5
            q1_ok = within_tol(q1_mean, correct_mean) and within_tol(q1_var, correct_var)
            if q1_ok:
                st.success(f"✅ Correct! The mean is {correct_mean} and the variance is {correct_var}.")
            else:
                st.error(f"❌ Not quite. The correct mean is {correct_mean} and variance is {correct_var}.")
            st.session_state.quiz_q1_ok = q1_ok
            if not q1_ok:
                 if st.button("🤖 Dr. X for Q1", key="q1_x"):
                    st.info(dr_x_feedback(
                        user_answer=f"μ={q1_mean}, σ²={q1_var}",
                        correct_answer="μ=np, σ²=np(1−p)",
                        rationale="Use μ=np and σ²=np(1−p). With n=40, p=0.25, that's 40*0.25=10 and 40*0.25*0.75=7.5."
                    ))

    # Q2 — LLN concept
    with st.container(border=True):
        st.subheader("Q2) Law of Large Numbers")
        q2 = st.radio("Which is the best statement of LLN?",
                      [ "The sample mean equals the population mean for any sample size.",
                        "The sample mean converges to the population mean as sample size grows.",
                        "Any distribution becomes normal if you sample enough.",
                        "Variance always decreases to zero with larger samples."], key="q2")
        if st.button("Check Q2", key="q2_check"):
            correct = "The sample mean converges to the population mean as sample size grows."
            q2_ok = (q2 == correct)
            if q2_ok:
                st.success("✅ Correct! This is the core principle of the LLN.")
            else:
                st.error("❌ Not quite. That statement describes the Central Limit Theorem, not the LLN.")
            st.session_state.quiz_q2_ok = q2_ok
            if not q2_ok:
                if st.button("🤖 Dr. X for Q2", key="q2_x"):
                    st.info(dr_x_feedback(q2, correct, "LLN is about the sample mean converging, while CLT is about the shape of the distribution of means becoming normal."))

    # Q3 — Random walk variance
    with st.container(border=True):
        st.subheader("Q3) Random Walk Variance")
        st.write("For a symmetric random walk with n steps (±1 increments), what is **Var(X_n)**?")
        q3 = st.text_input("Var(X_n) =", key="q3")
        if st.button("Check Q3", key="q3_check"):
            ok = (q3.strip().lower() == "n")
            if ok:
                st.success("✅ Correct! The variance grows linearly with the number of steps.")
            else:
                st.error("❌ Not quite. Remember that the variance of independent steps adds up.")
            st.session_state.quiz_q3_ok = ok
            if not ok:
                 if st.button("🤖 Dr. X for Q3", key="q3_x"):
                    st.info(dr_x_feedback(q3, "n", "Variance of a sum of independent steps is the sum of their variances. Each step has variance 1, so after n steps, the total variance is 1*n = n."))

    # Q4 — GBM one-step update
    with st.container(border=True):
        st.subheader("Q4) GBM one-step update")
        q4 = st.radio("Which formula updates S to the next step?", [
            "S_{t+Δt} = S_t + μΔt + σ√Δt·Z",
            "S_{t+Δt} = S_t · exp((μ − σ²/2)Δt + σ√Δt·Z)",
            "S_{t+Δt} = S_t · (1 + μ + σZ)",
            "S_{t+Δt} = μ + σZ"], key="q4")
        if st.button("Check Q4", key="q4_check"):
            correct = "S_{t+Δt} = S_t · exp((μ − σ²/2)Δt + σ√Δt·Z)"
            q4_ok = (q4 == correct)
            if q4_ok:
                st.success("✅ Correct! This is the discrete form for GBM, including the Ito correction term.")
            else:
                st.error("❌ Not quite. This is a tricky one. The exponential form handles multiplicative growth better.")
            st.session_state.quiz_q4_ok = q4_ok
            if not q4_ok:
                if st.button("🤖 Dr. X for Q4", key="q4_x"):
                    st.info(dr_x_feedback(q4, correct, "GBM is modeled with a lognormal distribution, so the update step involves an exponential term. The (μ − σ²/2) term is the drift correction from Itô's Lemma."))

    # Q5 — Stochastic Oscillator %K
    with st.container(border=True):
        st.subheader("Q5) Stochastic %K")
        st.write("Given Close=94, High_n=100, Low_n=80, compute %K.")
        q5 = st.number_input("%K =", key="q5", step=0.1)
        if st.button("Check Q5", key="q5_check"):
            correct = 100 * (94 - 80) / (100 - 80)
            q5_ok = within_tol(q5, correct, 0.01)
            if q5_ok:
                st.success(f"✅ Correct! The value is {correct:.1f}.")
            else:
                st.error(f"❌ Not quite. The correct value is {correct:.1f}.")
            st.session_state.quiz_q5_ok = q5_ok
            if not q5_ok:
                if st.button("🤖 Dr. X for Q5", key="q5_x"):
                    st.info(dr_x_feedback(q5, correct, "The formula is %K = 100*(C - L_n)/(H_n - L_n). Plug in the values: 100*(94-80)/(100-80) = 100*(14)/(20) = 70."))

    st.markdown("---")
    if st.button("🧮 Finalize Quiz Score", key="finalize_quiz_button"):
        oks = [st.session_state.get(k, False) for k in ["quiz_q1_ok", "quiz_q2_ok", "quiz_q3_ok", "quiz_q4_ok", "quiz_q5_ok"]]
        score = sum(1 for x in oks if x)
        st.session_state.quiz_score = score
        st.success(f"### Your final score: **{score}/5** 🎉")
        if score >= 4:
            award_badge("Stochastics Apprentice")
            st.balloons()
        else:
            st.info("Don't worry! Review the concepts and try again.")
        set_completed(SECTIONS["Quiz"])

# --- Reflection ---
def reflection():
    st.header("7) Reflection")
    st.markdown("""
    Take a moment to reflect on the concepts you've learned.
    
    - Where did **randomness** help you reason about uncertainty?  
    - Which model (binomial, random walk, GBM, stochastic oscillator) felt most intuitive? Why?
    - How does knowing this math change the way you might think about stock charts?
    """)
    r = st.text_area("Write a short reflection:", key="reflection_text")
    if st.button("Submit Reflection"):
        if r.strip():
            st.success("Great thinking! Connecting math to the world you observe is the key to mastery.")
            set_completed(SECTIONS["Reflection"])
            award_badge("Reflective Thinker")
            st.balloons()
        else:
            st.warning("Add a few sentences before submitting.")
            
# --- Explore More (Links) ---
def explore_more():
    st.header("🌐 Explore More — Cool & Useful Resources")
    
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
    
    set_completed(SECTIONS["Resources"])

# ============================
# Main App Logic (Tab-based)
# ============================
tab_titles = list(SECTIONS.keys())
tabs = st.tabs([f"**{title}**" for title in tab_titles])

with tabs[0]:
    wall_street_101()
with tabs[1]:
    chart_anatomy()
with tabs[2]:
    binomial_distribution()
with tabs[3]:
    lln_clt()
with tabs[4]:
    random_walks()
with tabs[5]:
    gbm()
with tabs[6]:
    stochastic_oscillator()
with tabs[7]:
    quiz()
with tabs[8]:
    reflection()
with tabs[9]:
    explore_more()

# --- Footer ---
st.markdown("---")
# The cc variable from the old code is gone, so let's check for it in session state
# or provide a default list of standards.
cc_list = st.session_state.get('cc', ["HSS-IC.A (understand & evaluate random processes)",
                                      "HSS-MD.A (expected value & probability models)"])
st.markdown(f"**📚 Common Core Standards:** {', '.join(cc_list)}")
st.markdown("---")
st.markdown("© MathCraft | Built by Xavier Honablue M.Ed for CognitiveCloud.ai")

```
