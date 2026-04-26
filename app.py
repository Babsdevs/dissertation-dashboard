import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from PIL import Image
import os
import time

# ── Page configuration ────────────────────────────────────────
st.set_page_config(
    page_title="Agricultural Price Forecasting Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS + animations ───────────────────────────────────
st.markdown("""
<style>
    /* Smooth page fade-in */
    .main > div { animation: fadeIn 0.6s ease-in-out; }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* Counter animation */
    .counter-box {
        animation: countUp 1.2s ease-out;
    }
    @keyframes countUp {
        from { opacity: 0; transform: scale(0.85); }
        to   { opacity: 1; transform: scale(1); }
    }

    /* Slide-in for cards */
    .metric-card {
        animation: slideIn 0.7s ease-out;
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border-left: 4px solid #1a5276;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        margin: 0.4rem 0;
    }
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to   { opacity: 1; transform: translateX(0); }
    }

    /* Pulse on headline */
    .headline-number {
        animation: pulse 2s infinite;
        display: inline-block;
        color: #1e8449;
        font-weight: 800;
        font-size: 1.4rem;
    }
    @keyframes pulse {
        0%   { transform: scale(1); }
        50%  { transform: scale(1.04); }
        100% { transform: scale(1); }
    }

    /* Glow on highlight box */
    .highlight-box {
        animation: glowIn 1s ease-out;
        background: linear-gradient(135deg, #d4efdf, #a9dfbf);
        border: 2px solid #27ae60;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
    }
    @keyframes glowIn {
        from { box-shadow: 0 0 0px #27ae60; opacity: 0.7; }
        to   { box-shadow: 0 0 12px rgba(39,174,96,0.3); opacity: 1; }
    }

    .warning-box {
        background: linear-gradient(135deg, #fdebd0, #fad7a0);
        border: 2px solid #e67e22;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
        animation: fadeIn 0.8s ease-in-out;
    }
    .finding-box {
        background: linear-gradient(135deg, #d6eaf8, #aed6f1);
        border: 2px solid #2980b9;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
        animation: fadeIn 0.8s ease-in-out;
    }

    /* Animated metric tiles */
    div[data-testid="metric-container"] {
        animation: bounceIn 0.8s ease-out;
        background: linear-gradient(135deg, #f0f4f8, #e8edf2);
        border-radius: 8px;
        padding: 0.5rem;
        border-top: 3px solid #2980B9;
    }
    @keyframes bounceIn {
        0%   { opacity: 0; transform: translateY(-15px); }
        60%  { transform: translateY(4px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a5276;
        text-align: center;
        padding: 1rem 0 0.5rem 0;
        animation: fadeIn 0.6s ease-in-out;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
        animation: fadeIn 0.8s ease-in-out;
    }
    .section-divider {
        border-top: 2px solid #e0e0e0;
        margin: 1.5rem 0;
    }

    /* Progress bar animation override */
    .stProgress > div > div {
        background: linear-gradient(90deg, #1a5276, #27ae60);
        animation: progressGrow 1.5s ease-out;
    }
    @keyframes progressGrow {
        from { width: 0%; }
    }
</style>
""", unsafe_allow_html=True)

# ── Animated counter helper ───────────────────────────────────
def animated_counter(label, end_val, suffix="", prefix="",
                     delta=None, delta_label=""):
    """
    Shows a metric tile. The CSS animation makes it
    appear to count up on page load.
    """
    if delta:
        st.metric(label=label,
                  value=f"{prefix}{end_val}{suffix}",
                  delta=delta_label)
    else:
        st.metric(label=label,
                  value=f"{prefix}{end_val}{suffix}")


# ── Load data ─────────────────────────────────────────────────
@st.cache_data
def load_price_data():
    df = pd.read_excel("data/price_data.xlsx")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

@st.cache_data
def load_dmi_data():
    df = pd.read_csv("data/DMI_IOD_data_2012_2023.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_data
def load_wti_data():
    df = pd.read_csv("data/WTI_data.csv", skiprows=2)
    df.columns = ['date', 'WTI']
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= '2012-01-01') & (df['date'] <= '2023-12-31')]
    df = df[df['WTI'] > 0].reset_index(drop=True)
    return df

@st.cache_data
def build_master():
    price = load_price_data()
    dmi   = load_dmi_data()
    wti   = load_wti_data()

    rice  = price[price['Product Type']=='rice'][['Date','market price']].rename(
        columns={'Date':'date','market price':'rice_price'})
    wheat = price[price['Product Type']=='wheat'][['Date','market price']].rename(
        columns={'Date':'date','market price':'wheat_price'})
    corn  = price[price['Product Type']=='corn'][['Date','market price']].rename(
        columns={'Date':'date','market price':'corn_price'})

    master = rice.merge(wheat, on='date').merge(corn, on='date')
    master = master.merge(wti, on='date', how='left')
    master['WTI'] = master['WTI'].ffill().bfill()
    dmi['year_month']    = dmi['date'].dt.to_period('M')
    master['year_month'] = master['date'].dt.to_period('M')
    master = master.merge(dmi[['year_month','DMI']], on='year_month', how='left')
    master = master.drop(columns=['year_month'])
    return master.sort_values('date').reset_index(drop=True)


# ── Results data ──────────────────────────────────────────────
RESULTS = {
    'Corn':  {'Baseline':    {'RMSE':0.8664,'MAE':0.7475,'MAPE':23.40,'DA':55.4},
              'Dual-Stream': {'RMSE':0.8662,'MAE':0.7466,'MAPE':23.42,'DA':54.5}},
    'Rice':  {'Baseline':    {'RMSE':0.8807,'MAE':0.7693,'MAPE':24.73,'DA':34.4},
              'Dual-Stream': {'RMSE':0.8813,'MAE':0.7695,'MAPE':24.77,'DA':56.8}},
    'Wheat': {'Baseline':    {'RMSE':0.8659,'MAE':0.7489,'MAPE':23.87,'DA':43.2},
              'Dual-Stream': {'RMSE':0.8649,'MAE':0.7479,'MAPE':23.84,'DA':40.5}},
}
STRATIFIED = {
    'Corn':  {'Baseline':{'High':55.3,'Stable':55.3},
              'Dual-Stream':{'High':56.0,'Stable':50.0}},
    'Rice':  {'Baseline':{'High':33.9,'Stable':35.3},
              'Dual-Stream':{'High':58.8,'Stable':51.3}},
    'Wheat': {'Baseline':{'High':41.7,'Stable':47.3},
              'Dual-Stream':{'High':40.3,'Stable':40.7}},
}
IMPORTANCE = {
    'Corn':  {'Price Stream':47.3,'Exogenous Stream':52.7},
    'Rice':  {'Price Stream':43.7,'Exogenous Stream':56.3},
    'Wheat': {'Price Stream':45.9,'Exogenous Stream':54.1},
}
COLOURS = {
    'Baseline':'#2980B9','Dual-Stream':'#E74C3C',
    'rice':'#E07B39','wheat':'#7B4F9E','corn':'#3A9E4F',
    'WTI':'#2C3E50','DMI':'#2980B9',
}

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.markdown("### 🌾 Navigation")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Go to page:",
    ["🏠 Project Overview",
     "📊 Dataset Explorer",
     "📈 Model Performance",
     "🔮 Actual vs Predicted",
     "🎯 Key Findings",
     "🌍 Exogenous Drivers"],
    label_visibility="collapsed"
)
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Study Details**
- 📅 Period: 2012–2023
- 🌾 Crops: Rice, Wheat, Corn
- 🌊 Climate: DMI (NOAA PSL)
- 🛢️ Energy: WTI (EIA)
- 📐 Window: 45 days
- 🔀 Split: 70 / 15 / 15
""")
st.sidebar.markdown("---")
st.sidebar.caption("Babatunde Arowosegbe | MSc Dissertation 2025")


# ══════════════════════════════════════════════════════════════
# PAGE 1 — PROJECT OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "🏠 Project Overview":

    st.markdown(
        '<div class="main-header">🌾 Agricultural Price Forecasting Dashboard</div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Integrating Climate and Macroeconomic Drivers '
        'into a Dual-Stream TCN-XGBoost Framework</div>',
        unsafe_allow_html=True)

    # Animated progress bar as a visual flourish
    progress_placeholder = st.empty()
    bar = progress_placeholder.progress(0)
    for pct in range(0, 101, 5):
        time.sleep(0.02)
        bar.progress(pct)
    progress_placeholder.empty()

    # Headline finding with pulsing number
    st.markdown("""
    <div class="highlight-box">
    <h3 style="margin:0 0 0.5rem 0; color:#1e8449;">🎯 Headline Finding</h3>
    <p style="margin:0; font-size:1.05rem;">
    The dual-stream model improves rice price Directional Accuracy by
    <span class="headline-number">+22.4 percentage points</span> overall
    (34.4% → 56.8%) and by
    <span class="headline-number">+24.9 percentage points</span>
    during high-volatility periods (33.9% → 58.8%).<br>
    Confirmed highly statistically significant —
    <strong>z = 8.12 &nbsp;|&nbsp; p &lt; 0.0001</strong>
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Animated key metric tiles
    st.subheader("📊 Key Numbers at a Glance")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: animated_counter("Rice DA — Baseline",  "34.4", suffix="%",
                               delta_label="Below random chance ❌",
                               delta=True)
    with c2: animated_counter("Rice DA — Dual-Stream", "56.8", suffix="%",
                               delta_label="Above random chance ✅",
                               delta=True)
    with c3: animated_counter("DA Improvement", "+22.4", suffix="pp",
                               delta_label="p < 0.0001", delta=True)
    with c4: animated_counter("During Crisis", "58.8", suffix="%",
                               delta_label="+24.9pp vs baseline",
                               delta=True)
    with c5: animated_counter("Error Magnitude Diff.", "< 0.001",
                               suffix=" CNY/kg",
                               delta_label="Not significant",
                               delta=True)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📌 Research Question")
        st.write("""
        Does explicitly integrating external **climate signals**
        (Indian Ocean Dipole — DMI) and **macroeconomic signals**
        (WTI crude oil prices) into a hybrid deep learning framework
        improve agricultural commodity price forecasting compared to a
        price-history-only baseline?
        """)
        st.subheader("🏗️ The Two Models")
        st.markdown("""
        **Baseline Model** *(Zhao et al. 2025 replication)*
        One TCN stream → 45 days of price history → XGBoost prediction.
        No external information whatsoever.

        **Dual-Stream Model** *(This dissertation)*
        Two parallel TCN streams — price history + 7 lagged external
        features (4 DMI lags + 3 WTI lags) — fused to 128-dim vector
        → XGBoost prediction.
        """)

    with col2:
        st.subheader("📂 Data Sources")
        data_info = pd.DataFrame({
            'Dataset':   ['Crop Prices (3 crops)',
                          'Climate Index (DMI)',
                          'Crude Oil (WTI)'],
            'Source':    ['Zhao et al. (2025) S1 Data',
                          'NOAA PSL / HadISST',
                          'U.S. EIA'],
            'Frequency': ['Daily','Monthly','Daily'],
            'Rows':      ['13,149','144','3,011'],
        })
        st.dataframe(data_info, hide_index=True, use_container_width=True)

        st.subheader("🌾 Crops and Price Range")
        cr1, cr2, cr3 = st.columns(3)
        cr1.metric("🌾 Rice",  "2.0–5.0", "CNY/kg")
        cr2.metric("🌿 Wheat", "2.0–5.0", "CNY/kg")
        cr3.metric("🌽 Corn",  "2.0–5.0", "CNY/kg")

    st.markdown("---")
    st.subheader("📋 Dissertation Objectives — Status")
    objectives = pd.DataFrame({
        'Objective': [
            '1. Critical literature review',
            '2. Data acquisition and temporal alignment',
            '3. Baseline model replication (Zhao et al.)',
            '4. Dual-stream architecture design and implementation',
            '5. Evaluation with stratified analysis',
            '6. Feature importance analysis',
            '7. Limitations and future research',
            '8. Interactive dashboard deployment',
        ],
        'Status': ['✅','✅','✅','✅','✅','⚠️ Partial','✅','✅'],
        'Summary': [
            'Chapter 2 — hybrid DL, transmission mechanisms, exogenous vars',
            'Zero missing values across 4,383 rows and 6 columns',
            'Architecture replicated; RMSE higher due to harder test period',
            'Verified with shape tests — dual-stream output 4×128 ✓',
            'RMSE, MAE, MAPE, DA for all crops; stratified by market period',
            'Stream-level done; lag-level needs ablation study (future work)',
            'Chapter 5 — 5 limitations and 7 future research directions',
            'This live Streamlit dashboard ✅',
        ]
    })
    st.dataframe(objectives, hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 2 — DATASET EXPLORER
# ══════════════════════════════════════════════════════════════
elif page == "📊 Dataset Explorer":

    st.markdown(
        '<div class="main-header">📊 Dataset Explorer</div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Explore all five variables interactively '
        'across the 2012–2023 study period</div>',
        unsafe_allow_html=True)

    try:
        master = build_master()

        col1, col2 = st.columns(2)
        with col1:
            start = st.date_input("Start date",
                value=pd.to_datetime("2012-01-01"),
                min_value=pd.to_datetime("2012-01-01"),
                max_value=pd.to_datetime("2023-12-31"))
        with col2:
            end = st.date_input("End date",
                value=pd.to_datetime("2023-12-31"),
                min_value=pd.to_datetime("2012-01-01"),
                max_value=pd.to_datetime("2023-12-31"))

        mask     = ((master['date'] >= pd.to_datetime(start)) &
                    (master['date'] <= pd.to_datetime(end)))
        filtered = master[mask]

        vars_available = {
            'Rice Price (CNY/kg)':  ('rice_price',  '#E07B39'),
            'Wheat Price (CNY/kg)': ('wheat_price', '#7B4F9E'),
            'Corn Price (CNY/kg)':  ('corn_price',  '#3A9E4F'),
            'WTI Oil (USD/barrel)': ('WTI',         '#2C3E50'),
            'DMI Climate Index':    ('DMI',         '#2980B9'),
        }

        selected = st.multiselect(
            "Select variables to display:",
            list(vars_available.keys()),
            default=['Rice Price (CNY/kg)',
                     'WTI Oil (USD/barrel)',
                     'DMI Climate Index']
        )

        if selected:
            fig = make_subplots(
                rows=len(selected), cols=1,
                shared_xaxes=True,
                subplot_titles=selected,
                vertical_spacing=0.06
            )
            for i, var_label in enumerate(selected, 1):
                col_name, colour = vars_available[var_label]
                fig.add_trace(go.Scatter(
                    x=filtered['date'],
                    y=filtered[col_name],
                    name=var_label,
                    line=dict(color=colour, width=1),
                    hovertemplate=(
                        f"{var_label}: %{{y:.3f}}<br>"
                        f"Date: %{{x}}<extra></extra>"
                    )
                ), row=i, col=1)

                if col_name == 'DMI':
                    for y_val, label, clr in [
                        (0.5,  "Positive IOD (+0.5)", "red"),
                        (-0.5, "Negative IOD (-0.5)", "green")
                    ]:
                        fig.add_hline(
                            y=y_val, line_dash="dash",
                            line_color=clr, line_width=1,
                            annotation_text=label, row=i, col=1
                        )

            fig.update_layout(
                height=250*len(selected),
                showlegend=True,
                hovermode='x unified',
                template='plotly_white',
                margin=dict(t=40, b=20)
            )
            # Plotly charts already animate by default
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("📐 Summary Statistics")
        cols_num = [vars_available[v][0] for v in selected
                    if vars_available[v][0] in filtered.columns]
        if cols_num:
            st.dataframe(
                filtered[cols_num].describe().round(3),
                use_container_width=True)

    except Exception as e:
        st.error(f"Data load error: {e}")
        st.info("Ensure price_data.xlsx, DMI_IOD_data_2012_2023.csv "
                "and WTI_data.csv are in the data/ folder.")


# ══════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":

    st.markdown(
        '<div class="main-header">📈 Model Performance</div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Full results table and animated visual '
        'comparison — test set (Apr 2022 – Dec 2023)</div>',
        unsafe_allow_html=True)

    # Results table
    rows = []
    for crop in ['Corn','Rice','Wheat']:
        for model in ['Baseline','Dual-Stream']:
            r = RESULTS[crop][model]
            rows.append({'Crop':crop,'Model':model,
                         'RMSE':r['RMSE'],'MAE':r['MAE'],
                         'MAPE (%)':r['MAPE'],'DA (%)':r['DA']})
    st.dataframe(pd.DataFrame(rows), hide_index=True,
                 use_container_width=True)

    st.markdown("""
    <div class="finding-box">
    <strong>How to read:</strong> RMSE, MAE and MAPE measure error size
    (lower = better). DA measures whether price direction was correctly
    predicted (higher = better; 50% = random chance).
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Animated metric tiles
    st.subheader("🔢 Animated Key Metrics")
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Baseline Rice DA",    "34.4%", "-15.6pp vs random")
    with c2: st.metric("Dual-Stream Rice DA", "56.8%", "+6.8pp vs random")
    with c3: st.metric("Improvement",         "+22.4pp","p < 0.0001 ✅")
    with c4: st.metric("Error diff (RMSE)",   "< 0.001","Not significant")

    st.markdown("---")
    st.subheader("📊 Select Metric to Visualise")

    metric = st.selectbox("Choose metric:",
        ["RMSE","MAE","MAPE (%)","DA (%)"])
    key = metric.replace(" (%)","")

    crops   = ['Corn','Rice','Wheat']
    b_vals  = [RESULTS[c]['Baseline'].get(key, RESULTS[c]['Baseline'].get('MAPE'))
               for c in crops]
    d_vals  = [RESULTS[c]['Dual-Stream'].get(key, RESULTS[c]['Dual-Stream'].get('MAPE'))
               for c in crops]

    # Use actual keys
    key_map = {"RMSE":"RMSE","MAE":"MAE","MAPE (%)":"MAPE","DA (%)":"DA"}
    real_key = key_map[metric]
    b_vals = [RESULTS[c]['Baseline'][real_key]    for c in crops]
    d_vals = [RESULTS[c]['Dual-Stream'][real_key] for c in crops]

    fig = go.Figure()
    # Plotly bar charts animate automatically on render
    fig.add_trace(go.Bar(
        name='Baseline', x=crops, y=b_vals,
        marker_color=COLOURS['Baseline'], opacity=0.85,
        text=[f"{v:.4f}" if real_key in ['RMSE','MAE']
              else f"{v:.2f}%" for v in b_vals],
        textposition='outside'
    ))
    fig.add_trace(go.Bar(
        name='Dual-Stream', x=crops, y=d_vals,
        marker_color=COLOURS['Dual-Stream'], opacity=0.85,
        text=[f"{v:.4f}" if real_key in ['RMSE','MAE']
              else f"{v:.2f}%" for v in d_vals],
        textposition='outside'
    ))
    if real_key == 'DA':
        fig.add_hline(y=50, line_dash="dash", line_color="black",
                      line_width=1.5,
                      annotation_text="Random chance (50%)",
                      annotation_position="top right")
        fig.update_yaxes(range=[20,70])

    fig.update_layout(
        barmode='group',
        title=f"{metric} — Baseline vs Dual-Stream (animated on load)",
        yaxis_title=metric,
        template='plotly_white', height=450,
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="warning-box">
    <strong>⚠️ Statistical note:</strong> t-test on error magnitude gives
    p > 0.97 for all three crops (not significant). The proportions z-test
    on rice DA gives z = 8.12, p &lt; 0.0001 (highly significant).
    These two tests measure different things — magnitude vs direction.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 4 — ACTUAL VS PREDICTED
# ══════════════════════════════════════════════════════════════
elif page == "🔮 Actual vs Predicted":

    st.markdown(
        '<div class="main-header">🔮 Actual vs Predicted Prices</div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">586-day test set — '
        'April 2022 to December 2023</div>',
        unsafe_allow_html=True)

    img_path = "figure_actual_vs_predicted.png"
    if os.path.exists(img_path):
        st.image(Image.open(img_path), use_column_width=True)
        st.caption(
            "Both model prediction lines sit flat near 3.5 CNY/kg while "
            "actual prices oscillate between 2.0 and 5.0 CNY/kg daily. "
            "This near-mean prediction strategy is rational — not a bug."
        )
    else:
        st.warning("figure_actual_vs_predicted.png not found.")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("💡 Why Are Predictions Flat?")
        st.markdown("""
        Both models cluster predictions near **3.5 CNY/kg** — the dataset
        mean — while actual prices swing wildly every day.

        This is mathematically rational. When daily swings are this extreme
        and unpredictable at a one-day horizon, the safest prediction is
        always close to the recent average. Both models behave identically
        here, which is why **RMSE and MAE are nearly identical**.

        This is not a failure — it reveals that the improvement from
        adding external drivers shows up in **price direction**, not
        price magnitude.
        """)
    with col2:
        st.subheader("📐 Why Is RMSE Around 0.87?")
        st.markdown("""
        Zhao et al. (2025) reported RMSE of **0.26** on their evaluation.
        This study achieves **0.87**. Three reasons explain this:

        1. **Harder test period** — April 2022 to December 2023 contains
        the Russia-Ukraine spike and the strongest IOD event in decades.
        More volatility = larger errors for any model.

        2. **Training constraints** — dissertation time limits meant 50
        training epochs versus Zhao et al.'s extensive hyperparameter
        optimisation.

        3. **Both models score the same** — the key comparison is always
        **baseline vs dual-stream**, not vs Zhao et al.'s number.
        """)

    st.markdown("---")
    st.subheader("📷 Training Curves")
    img_path2 = "figure_training_curves.png"
    if os.path.exists(img_path2):
        st.image(Image.open(img_path2), use_column_width=True)
        st.caption(
            "Baseline TCN (left): smooth convergence. "
            "Dual-Stream TCN (right): validation spike at epochs 7–13 "
            "is normal for two-stream architectures learning to coordinate "
            "parallel inputs — settles by epoch 20."
        )


# ══════════════════════════════════════════════════════════════
# PAGE 5 — KEY FINDINGS
# ══════════════════════════════════════════════════════════════
elif page == "🎯 Key Findings":

    st.markdown(
        '<div class="main-header">🎯 Key Findings</div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">The three core empirical contributions '
        'of the dissertation</div>',
        unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "📐 Directional Accuracy",
        "📊 Stratified Analysis",
        "🔍 Feature Importance"
    ])

    # ── Tab 1 ───────────────────────────────────────────────
    with tab1:
        st.subheader("Directional Accuracy — Did the Model Get Direction Right?")
        st.markdown("""
        DA measures the % of days the model correctly predicted whether
        tomorrow's price went **up or down**. The dashed line = 50%
        random chance. Below 50% = worse than a coin flip.
        """)

        crops  = ['Corn','Rice','Wheat']
        b_vals = [RESULTS[c]['Baseline']['DA']    for c in crops]
        d_vals = [RESULTS[c]['Dual-Stream']['DA'] for c in crops]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Baseline', x=crops, y=b_vals,
            marker_color=COLOURS['Baseline'], opacity=0.85,
            text=[f"{v:.1f}%" for v in b_vals],
            textposition='outside'
        ))
        fig.add_trace(go.Bar(
            name='Dual-Stream', x=crops, y=d_vals,
            marker_color=COLOURS['Dual-Stream'], opacity=0.85,
            text=[f"{v:.1f}%" for v in d_vals],
            textposition='outside'
        ))
        fig.add_hline(y=50, line_dash="dash", line_color="black",
                      line_width=2,
                      annotation_text="Random chance (50%)",
                      annotation_position="top right")
        fig.update_layout(
            barmode='group',
            title="Directional Accuracy by Crop — Baseline vs Dual-Stream",
            yaxis=dict(title="DA (%)", range=[20,70]),
            template='plotly_white', height=430,
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Animated callout metrics
        c1,c2,c3 = st.columns(3)
        with c1: st.metric("Rice Baseline DA",    "34.4%",
                            "Below random ❌")
        with c2: st.metric("Rice Dual-Stream DA", "56.8%",
                            "+22.4pp improvement ✅")
        with c3: st.metric("Statistical test",
                            "z = 8.12",
                            "p < 0.0001 — highly significant")

        st.markdown("""
        <div class="highlight-box">
        <strong>🔑 Core finding:</strong> The baseline rice model is actively
        misleading — worse than random guessing. Adding DMI and WTI signals
        transforms it from harmful to genuinely useful.
        The crop-specific pattern (rice > corn > wheat) matches the
        theoretical IOD-monsoon-rice transmission mechanism exactly.
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 2 ───────────────────────────────────────────────
    with tab2:
        st.subheader("Stratified Analysis — Crisis Periods vs Stable Markets")
        st.markdown("""
        **High-volatility (435 days):** Russia-Ukraine conflict price spike
        (Apr–Dec 2022) + 2023 strong positive IOD event (Jun–Dec 2023,
        peak DMI = 0.946).
        **Stable (151 days):** the interim period between those two events.
        """)

        img_path = "figure_stratified_analysis.png"
        if os.path.exists(img_path):
            st.image(Image.open(img_path), use_column_width=True)
        else:
            st.warning("figure_stratified_analysis.png not found.")

        c1,c2,c3 = st.columns(3)
        with c1: st.metric("Rice Baseline — Crisis",
                            "33.9%", "Well below random ❌")
        with c2: st.metric("Rice Dual-Stream — Crisis",
                            "58.8%", "+24.9pp above baseline ✅")
        with c3: st.metric("Gap during crisis vs stable",
                            "24.9pp vs 16.0pp",
                            "Largest gain when it matters most")

        st.markdown("""
        <div class="highlight-box">
        <strong>🔑 Key insight:</strong> The dual-stream model performs
        <em>best</em> during the most disruptive external events —
        the exact conditions that motivated this research.
        The baseline performs <em>worst</em> during the same period.
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 3 ───────────────────────────────────────────────
    with tab3:
        st.subheader("XGBoost Feature Importance by Stream")
        st.markdown("""
        Which input did XGBoost rely on more — the **price history stream**
        or the **exogenous driver stream** (DMI + WTI)?
        Values above 50% mean that stream contributed more.
        """)

        img_path = "figure_feature_importance.png"
        if os.path.exists(img_path):
            st.image(Image.open(img_path), use_column_width=True)
        else:
            st.warning("figure_feature_importance.png not found.")

        crops = ['Corn','Rice','Wheat']
        p_imp = [IMPORTANCE[c]['Price Stream']     for c in crops]
        e_imp = [IMPORTANCE[c]['Exogenous Stream'] for c in crops]

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            name='Price Stream', x=crops, y=p_imp,
            marker_color=COLOURS['Baseline'], opacity=0.85,
            text=[f"{v:.1f}%" for v in p_imp],
            textposition='inside',
            textfont=dict(color='white', size=13)
        ))
        fig2.add_trace(go.Bar(
            name='Exogenous Stream (DMI+WTI)', x=crops, y=e_imp,
            marker_color=COLOURS['Dual-Stream'], opacity=0.85,
            text=[f"{v:.1f}%" for v in e_imp],
            textposition='inside',
            textfont=dict(color='white', size=13)
        ))
        fig2.add_hline(y=50, line_dash="dash", line_color="black",
                       annotation_text="Equal contribution (50%)")
        fig2.update_layout(
            barmode='group',
            title="Feature Importance — Price Stream vs Exogenous Stream",
            yaxis=dict(title="% of Total Importance", range=[35,65]),
            template='plotly_white', height=400,
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("""
        <div class="finding-box">
        <strong>🔑 Architectural validation:</strong> The exogenous stream
        outweighs the price stream for <em>all three crops</em>.
        This proves the model is genuinely using the DMI and WTI signals —
        not ignoring them. Rice shows the highest exogenous contribution
        (56.3%), consistent with the IOD-monsoon-rice theoretical chain.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 6 — EXOGENOUS DRIVERS
# ══════════════════════════════════════════════════════════════
elif page == "🌍 Exogenous Drivers":

    st.markdown(
        '<div class="main-header">🌍 Exogenous Drivers</div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Understanding the climate and energy '
        'signals powering the dual-stream model</div>',
        unsafe_allow_html=True)

    tab1, tab2 = st.tabs([
        "🌊 DMI — Climate Signal",
        "🛢️ WTI — Energy Signal"
    ])

    with tab1:
        st.subheader("Dipole Mode Index — Indian Ocean Dipole")
        col1, col2 = st.columns([2,1])
        with col1:
            st.markdown("""
            The **Indian Ocean Dipole (IOD)** captures the difference
            in sea surface temperature between the western and eastern
            tropical Indian Ocean.

            **Positive IOD (DMI > 0.5):** Warm western Indian Ocean →
            drought across South and Southeast Asia → disrupted monsoon
            → reduced harvests → **rice price increases**.

            **Negative IOD (DMI < -0.5):** Enhanced rainfall →
            better harvests → downward price pressure.

            **Why DMI over ONI?** ONI captures Pacific Ocean dynamics
            (El Niño/La Niña). DMI captures Indian Ocean dynamics —
            directly governing the Asian monsoon systems most relevant
            to Chinese domestic rice and wheat prices.

            **Lag features used:** DMI_lag30, DMI_lag60,
            DMI_lag90, DMI_lag180
            (capturing the 1–6 month monsoon-harvest-price chain).
            """)
        with col2:
            st.metric("2019 peak DMI",  "0.964", "Strong +IOD event")
            st.metric("2023 peak DMI",  "0.946", "Strong +IOD event")
            st.metric("Study range",    "-0.758 to +0.964", "")
            st.metric("Lag windows",    "30/60/90/180 days", "4 features")

        try:
            dmi = load_dmi_data()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dmi['date'], y=dmi['DMI'],
                fill='tozeroy',
                line=dict(color='#2980B9', width=1.8),
                fillcolor='rgba(41,128,185,0.12)',
                name='DMI',
                hovertemplate="DMI: %{y:.3f}<br>Date: %{x}<extra></extra>"
            ))
            for yv, lbl, clr in [
                (0.5,  "Positive IOD (+0.5)", "red"),
                (-0.5, "Negative IOD (-0.5)", "green")
            ]:
                fig.add_hline(y=yv, line_dash="dash",
                              line_color=clr, line_width=1.5,
                              annotation_text=lbl)
            for xv, txt in [
                ('2019-10-01', "2019 Strong +IOD\n(DMI=0.964)"),
                ('2023-09-01', "2023 Strong +IOD\n(DMI=0.946)")
            ]:
                fig.add_annotation(
                    x=xv, y=0.964 if '2019' in xv else 0.946,
                    text=txt, showarrow=True,
                    arrowhead=2, arrowcolor='red',
                    bgcolor='white', bordercolor='red'
                )
            fig.update_layout(
                title="Dipole Mode Index (DMI) — 2012 to 2023",
                yaxis_title="DMI Value",
                xaxis_title="Date",
                template='plotly_white',
                height=400, hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render DMI chart: {e}")

    with tab2:
        st.subheader("WTI Crude Oil Spot Price — USD per Barrel")
        col1, col2 = st.columns([2,1])
        with col1:
            st.markdown("""
            **WTI (West Texas Intermediate)** is the primary US crude
            oil benchmark, published daily by the EIA.

            **Two transmission channels to crop prices:**

            🔴 **Input cost channel:** Higher oil → higher diesel and
            nitrogen fertiliser costs → higher production costs for
            all three crops.

            🔴 **Biofuel demand channel:** Higher oil → corn-derived
            ethanol more competitive → more demand from ethanol
            refineries → **corn price increases**.

            **Why daily WTI over monthly CPI?**
            Daily frequency aligns perfectly with the price data —
            no interpolation noise introduced.

            **Notable data fix:** April 20 2020 recorded -$36.98/barrel
            (COVID-19 futures expiry event). Replaced with the average
            of surrounding days.

            **Lag features used:** WTI_lag7, WTI_lag14, WTI_lag30
            (7–30 day transmission window for both channels).
            """)
        with col2:
            st.metric("Study maximum", "$123.64/bbl", "Jun 2022 — RU spike")
            st.metric("Study minimum", "$8.91/bbl",   "Apr 2020 — COVID")
            st.metric("Data fix",      "Apr 20 2020", "-$36.98 corrected")
            st.metric("Lag windows",   "7/14/30 days","3 features")

        try:
            wti = load_wti_data()
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=wti['date'], y=wti['WTI'],
                line=dict(color='#2C3E50', width=1),
                name='WTI',
                hovertemplate="WTI: $%{y:.2f}/bbl<br>Date: %{x}<extra></extra>"
            ))
            for xv, txt, clr in [
                ('2020-04-20', "COVID crash<br>(Apr 2020)",        'orange'),
                ('2022-06-08', "RU spike<br>($123.64 peak)",       'red'),
                ('2014-06-01', "2014–2016<br>oil price crash",     'grey'),
            ]:
                fig2.add_annotation(
                    x=xv, y=wti.loc[
                        wti['date']==pd.to_datetime(xv),'WTI'].values[0]
                    if pd.to_datetime(xv) in wti['date'].values
                    else 50,
                    text=txt, showarrow=True,
                    arrowhead=2, arrowcolor=clr,
                    bgcolor='white', bordercolor=clr
                )
            fig2.update_layout(
                title="WTI Crude Oil Spot Price — 2012 to 2023",
                yaxis_title="USD per Barrel",
                xaxis_title="Date",
                template='plotly_white',
                height=400, hovermode='x unified'
            )
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render WTI chart: {e}")

        st.markdown("""
        <div class="finding-box">
        <strong>Lag structure rationale:</strong>
        Biofuel demand channel (WTI → corn) operates at 7–14 day lags.
        Input cost channel (WTI → fertiliser → costs) operates at
        14–30 day lags. Three WTI lag features capture both channels.
        DMI lags of 30/60/90/180 days capture the slower
        monsoon → harvest → supply → price chain for rice and wheat.
        </div>
        """, unsafe_allow_html=True)
