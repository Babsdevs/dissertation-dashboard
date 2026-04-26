import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from PIL import Image
import os

# ── Page configuration ────────────────────────────────────────
st.set_page_config(
    page_title="Agricultural Price Forecasting Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a5276;
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border-left: 4px solid #1a5276;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        margin: 0.4rem 0;
    }
    .highlight-box {
        background: linear-gradient(135deg, #d4efdf, #a9dfbf);
        border: 2px solid #27ae60;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #fdebd0, #fad7a0);
        border: 2px solid #e67e22;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
    }
    .finding-box {
        background: linear-gradient(135deg, #d6eaf8, #aed6f1);
        border: 2px solid #2980b9;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
    }
    .section-divider {
        border-top: 2px solid #e0e0e0;
        margin: 1.5rem 0;
    }
    .sidebar-title {
        font-size: 1rem;
        font-weight: 600;
        color: #1a5276;
    }
</style>
""", unsafe_allow_html=True)

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
    df = df[df['WTI'] > 0]
    df = df.reset_index(drop=True)
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

    dmi['year_month'] = dmi['date'].dt.to_period('M')
    master['year_month'] = master['date'].dt.to_period('M')
    master = master.merge(dmi[['year_month','DMI']], on='year_month', how='left')
    master = master.drop(columns=['year_month'])
    master = master.sort_values('date').reset_index(drop=True)
    return master

# ── Results data (hardcoded from your analysis) ───────────────
RESULTS = {
    'Corn':  {'Baseline': {'RMSE':0.8664,'MAE':0.7475,'MAPE':23.40,'DA':55.4},
              'Dual-Stream': {'RMSE':0.8662,'MAE':0.7466,'MAPE':23.42,'DA':54.5}},
    'Rice':  {'Baseline': {'RMSE':0.8807,'MAE':0.7693,'MAPE':24.73,'DA':34.4},
              'Dual-Stream': {'RMSE':0.8813,'MAE':0.7695,'MAPE':24.77,'DA':56.8}},
    'Wheat': {'Baseline': {'RMSE':0.8659,'MAE':0.7489,'MAPE':23.87,'DA':43.2},
              'Dual-Stream': {'RMSE':0.8649,'MAE':0.7479,'MAPE':23.84,'DA':40.5}},
}

STRATIFIED = {
    'Corn':  {'Baseline':   {'High':55.3,'Stable':55.3},
              'Dual-Stream':{'High':56.0,'Stable':50.0}},
    'Rice':  {'Baseline':   {'High':33.9,'Stable':35.3},
              'Dual-Stream':{'High':58.8,'Stable':51.3}},
    'Wheat': {'Baseline':   {'High':41.7,'Stable':47.3},
              'Dual-Stream':{'High':40.3,'Stable':40.7}},
}

IMPORTANCE = {
    'Corn':  {'Price Stream':47.3,'Exogenous Stream':52.7},
    'Rice':  {'Price Stream':43.7,'Exogenous Stream':56.3},
    'Wheat': {'Price Stream':45.9,'Exogenous Stream':54.1},
}

COLOURS = {
    'Baseline':    '#2980B9',
    'Dual-Stream': '#E74C3C',
    'rice':        '#E07B39',
    'wheat':       '#7B4F9E',
    'corn':        '#3A9E4F',
    'WTI':         '#2C3E50',
    'DMI':         '#2980B9',
}

# ── Sidebar navigation ────────────────────────────────────────
st.sidebar.markdown(
    '<p class="sidebar-title">🌾 Navigation</p>', unsafe_allow_html=True)
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
st.sidebar.markdown(
    "*Babatunde Arowosegbe | MSc Dissertation 2025*")

# ══════════════════════════════════════════════════════════════
# PAGE 1 — PROJECT OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "🏠 Project Overview":

    st.markdown(
        '<div class="main-header">🌾 Agricultural Price Forecasting</div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Integrating Climate and Macroeconomic Drivers '
        'into a Dual-Stream TCN-XGBoost Framework</div>',
        unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Headline finding
    st.markdown("""
    <div class="highlight-box">
    <h3 style="margin:0 0 0.5rem 0; color:#1e8449;">🎯 Headline Finding</h3>
    <p style="margin:0; font-size:1.05rem;">
    The dual-stream model improves rice price <strong>Directional Accuracy by
    +22.4 percentage points</strong> overall (34.4% → 56.8%) and by
    <strong>+24.9 percentage points during high-volatility periods</strong>
    (33.9% → 58.8%), confirmed highly statistically significant
    (z = 8.12, p &lt; 0.0001).
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📌 Research Question")
        st.write("""
        Does explicitly integrating external **climate signals** (Indian Ocean
        Dipole — DMI) and **macroeconomic signals** (WTI crude oil prices) into
        a hybrid deep learning framework improve agricultural commodity price
        forecasting compared to a price-history-only baseline?
        """)

        st.subheader("🏗️ The Two Models")
        st.markdown("""
        **Baseline Model (Zhao et al. 2025 replication)**
        One TCN stream processes 45 days of price history → XGBoost predicts
        tomorrow's price. No external information.

        **Dual-Stream Model (This dissertation)**
        Two parallel TCN streams — one for price history, one for 7 lagged
        external driver features (4 DMI lags + 3 WTI lags) — fused into a
        128-dimensional vector → XGBoost predicts tomorrow's price.
        """)

    with col2:
        st.subheader("📂 Data Sources")
        data_info = pd.DataFrame({
            'Dataset': ['Crop Prices', 'Climate (DMI)', 'Energy (WTI)'],
            'Source': ['Zhao et al. (2025) S1 Data',
                       'NOAA PSL / HadISST',
                       'U.S. EIA'],
            'Frequency': ['Daily', 'Monthly', 'Daily'],
            'Period': ['2012–2023', '2012–2023', '2012–2023'],
        })
        st.dataframe(data_info, hide_index=True, use_container_width=True)

        st.subheader("🌾 Crops Studied")
        c1, c2, c3 = st.columns(3)
        c1.metric("🌾 Rice",  "4,383 days", "CNY/kg")
        c2.metric("🌿 Wheat", "4,383 days", "CNY/kg")
        c3.metric("🌽 Corn",  "4,383 days", "CNY/kg")

    st.markdown("---")
    st.subheader("📋 Dissertation Objectives — Status")

    objectives = pd.DataFrame({
        'Objective': [
            '1. Critical literature review',
            '2. Data acquisition and alignment',
            '3. Baseline model replication',
            '4. Dual-stream architecture design',
            '5. Evaluation and stratified analysis',
            '6. Feature importance analysis',
            '7. Limitations and future research',
            '8. Interactive dashboard deployment',
        ],
        'Status': ['✅ Complete','✅ Complete','✅ Complete','✅ Complete',
                   '✅ Complete','⚠️ Partial','✅ Complete','✅ Complete'],
        'Note': [
            'Chapter 2 — all domains covered with real references',
            'Zero missing values across all 4,383 rows',
            'Architecture replicated; RMSE differs due to harder test period',
            'Dual-stream implemented and verified with shape tests',
            'All metrics, all crops, stratified by market period',
            'Stream-level done; lag-level needs ablation study',
            'Chapter 5 — 5 limitations, 7 future directions',
            'This dashboard — live on Streamlit Cloud',
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
        '<div class="sub-header">Explore all five variables across '
        'the 2012–2023 study period</div>',
        unsafe_allow_html=True)

    try:
        master = build_master()

        # Date range selector
        st.subheader("🗓️ Select Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start = st.date_input(
                "Start date",
                value=pd.to_datetime("2012-01-01"),
                min_value=pd.to_datetime("2012-01-01"),
                max_value=pd.to_datetime("2023-12-31")
            )
        with col2:
            end = st.date_input(
                "End date",
                value=pd.to_datetime("2023-12-31"),
                min_value=pd.to_datetime("2012-01-01"),
                max_value=pd.to_datetime("2023-12-31")
            )

        mask = (master['date'] >= pd.to_datetime(start)) & \
               (master['date'] <= pd.to_datetime(end))
        filtered = master[mask]

        # Variable selector
        st.subheader("📌 Select Variables to Display")
        vars_available = {
            'Rice Price (CNY/kg)':  'rice_price',
            'Wheat Price (CNY/kg)': 'wheat_price',
            'Corn Price (CNY/kg)':  'corn_price',
            'WTI Oil (USD/barrel)': 'WTI',
            'DMI Climate Index':    'DMI',
        }
        selected = st.multiselect(
            "Choose variables:",
            list(vars_available.keys()),
            default=['Rice Price (CNY/kg)', 'WTI Oil (USD/barrel)',
                     'DMI Climate Index']
        )

        if selected:
            colour_map = {
                'Rice Price (CNY/kg)':  '#E07B39',
                'Wheat Price (CNY/kg)': '#7B4F9E',
                'Corn Price (CNY/kg)':  '#3A9E4F',
                'WTI Oil (USD/barrel)': '#2C3E50',
                'DMI Climate Index':    '#2980B9',
            }

            fig = make_subplots(
                rows=len(selected), cols=1,
                shared_xaxes=True,
                subplot_titles=selected,
                vertical_spacing=0.06
            )

            for i, var_label in enumerate(selected, 1):
                col_name = vars_available[var_label]
                fig.add_trace(
                    go.Scatter(
                        x=filtered['date'],
                        y=filtered[col_name],
                        name=var_label,
                        line=dict(color=colour_map[var_label], width=1),
                        hovertemplate=f"{var_label}: %{{y:.3f}}<br>Date: %{{x}}<extra></extra>"
                    ),
                    row=i, col=1
                )

                # Add IOD threshold lines on DMI panel
                if col_name == 'DMI':
                    fig.add_hline(
                        y=0.5, line_dash="dash",
                        line_color="red", line_width=1,
                        annotation_text="Positive IOD (+0.5)",
                        row=i, col=1
                    )
                    fig.add_hline(
                        y=-0.5, line_dash="dash",
                        line_color="green", line_width=1,
                        annotation_text="Negative IOD (-0.5)",
                        row=i, col=1
                    )

            fig.update_layout(
                height=250 * len(selected),
                showlegend=True,
                hovermode='x unified',
                template='plotly_white',
                margin=dict(t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

        # Summary stats
        st.subheader("📐 Summary Statistics for Selected Period")
        cols_to_show = [vars_available[v] for v in selected
                        if vars_available[v] in filtered.columns]
        if cols_to_show:
            stats = filtered[cols_to_show].describe().round(3)
            st.dataframe(stats, use_container_width=True)

    except Exception as e:
        st.error(f"Could not load data: {e}")
        st.info("Make sure price_data.xlsx, DMI_IOD_data_2012_2023.csv "
                "and WTI_data.csv are in the data/ folder.")

# ══════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":

    st.markdown(
        '<div class="main-header">📈 Model Performance</div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Full results table and visual comparison '
        'for both models across all three crops</div>',
        unsafe_allow_html=True)

    # Results table
    st.subheader("📋 Complete Results Table — Test Set (Apr 2022 – Dec 2023)")

    rows = []
    for crop in ['Corn', 'Rice', 'Wheat']:
        for model in ['Baseline', 'Dual-Stream']:
            r = RESULTS[crop][model]
            rows.append({
                'Crop': crop, 'Model': model,
                'RMSE (CNY/kg)': r['RMSE'],
                'MAE (CNY/kg)':  r['MAE'],
                'MAPE (%)':      r['MAPE'],
                'DA (%)':        r['DA'],
            })
    df_results = pd.DataFrame(rows)
    st.dataframe(df_results, hide_index=True, use_container_width=True)

    st.markdown("""
    <div class="finding-box">
    <strong>How to read this table:</strong> RMSE, MAE and MAPE measure the
    size of prediction errors (lower is better). Directional Accuracy measures
    whether the model correctly predicted price direction — up or down
    (higher is better, 50% = random chance).
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Metric selector
    st.subheader("📊 Visual Comparison")
    metric = st.selectbox(
        "Select metric to visualise:",
        ["RMSE (CNY/kg)", "MAE (CNY/kg)", "MAPE (%)", "DA (%)"]
    )

    metric_key_map = {
        "RMSE (CNY/kg)": "RMSE",
        "MAE (CNY/kg)":  "MAE",
        "MAPE (%)":      "MAPE",
        "DA (%)":        "DA"
    }
    key = metric_key_map[metric]
    crops = ['Corn', 'Rice', 'Wheat']

    base_vals = [RESULTS[c]['Baseline'][key]    for c in crops]
    dual_vals = [RESULTS[c]['Dual-Stream'][key] for c in crops]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Baseline', x=crops, y=base_vals,
        marker_color=COLOURS['Baseline'], opacity=0.85,
        text=[f"{v:.4f}" if key in ['RMSE','MAE']
              else f"{v:.2f}%" for v in base_vals],
        textposition='outside'
    ))
    fig.add_trace(go.Bar(
        name='Dual-Stream', x=crops, y=dual_vals,
        marker_color=COLOURS['Dual-Stream'], opacity=0.85,
        text=[f"{v:.4f}" if key in ['RMSE','MAE']
              else f"{v:.2f}%" for v in dual_vals],
        textposition='outside'
    ))

    if key == 'DA':
        fig.add_hline(
            y=50, line_dash="dash", line_color="black",
            line_width=1.5,
            annotation_text="Random chance (50%)",
            annotation_position="right"
        )
        fig.update_yaxes(range=[20, 70])

    fig.update_layout(
        barmode='group',
        title=f"{metric} — Baseline vs Dual-Stream",
        yaxis_title=metric,
        template='plotly_white',
        height=450,
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Key metric callouts
    st.markdown("---")
    st.subheader("🔑 Key Numbers at a Glance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rice DA — Baseline",    "34.4%", "-15.6pp vs random")
    c2.metric("Rice DA — Dual-Stream", "56.8%", "+6.8pp vs random")
    c3.metric("DA Improvement (Rice)", "+22.4pp", "p < 0.0001")
    c4.metric("Error Magnitude Diff.", "< 0.001", "Not significant")

    st.markdown("""
    <div class="warning-box">
    <strong>⚠️ Statistical Note:</strong> The t-test on error magnitude
    (RMSE/MAE) returns p-values above 0.97 for all three crops — confirming
    no significant difference in error size. However the proportions z-test
    on rice Directional Accuracy returns z = 8.12, p &lt; 0.0001 — confirming
    the DA improvement is highly statistically significant.
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
        '<div class="sub-header">Test set predictions across the '
        '586-day evaluation period (April 2022 – December 2023)</div>',
        unsafe_allow_html=True)

    img_path = "figure_actual_vs_predicted.png"
    if os.path.exists(img_path):
        img = Image.open(img_path)
        st.image(img, use_column_width=True)
    else:
        st.warning("figure_actual_vs_predicted.png not found in the root folder.")

    st.markdown("---")
    st.subheader("💡 How to Read This Chart")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **The solid coloured lines** show actual daily prices oscillating
        between 2.0 and 5.0 CNY/kg throughout the test period.

        **The dashed lines** show both model predictions — both sit almost
        perfectly flat near the dataset mean of 3.5 CNY/kg.

        **Why are predictions flat?** Both models adopt a near-mean prediction
        strategy. When daily price swings are this extreme and erratic, the
        mathematically safest prediction is always close to the recent mean.
        This is rational behaviour, not a bug.
        """)
    with col2:
        st.markdown("""
        **Why is RMSE similar for both models?** Because both models are making
        the same near-mean prediction. RMSE measures the gap between prediction
        and actual price — and when both models predict 3.5 while actual prices
        swing to 4.8 or 2.1, both accumulate similar errors.

        **What RMSE misses:** Whether the model correctly predicted the
        *direction* of price movement — up or down. That is captured by
        Directional Accuracy, shown on the Key Findings page.

        **Test period significance:** This period contains the
        Russia-Ukraine commodity spike (2022) and the strongest positive
        IOD climate event in decades (2023 peak DMI = 0.946).
        """)

    st.markdown("---")
    st.subheader("📷 Training Curves — How Both Models Learned")

    img_path2 = "figure_training_curves.png"
    if os.path.exists(img_path2):
        img2 = Image.open(img_path2)
        st.image(img2, use_column_width=True)
        st.caption(
            "Figure: Training and validation loss curves over 50 epochs. "
            "Baseline (left) shows smooth convergence. Dual-stream (right) "
            "shows an initial validation spike at epochs 7–13 — normal for "
            "a two-stream model learning to coordinate two input types "
            "simultaneously — before stabilising by epoch 20."
        )

# ══════════════════════════════════════════════════════════════
# PAGE 5 — KEY FINDINGS
# ══════════════════════════════════════════════════════════════
elif page == "🎯 Key Findings":

    st.markdown(
        '<div class="main-header">🎯 Key Findings</div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">The three core empirical results '
        'of the dissertation</div>',
        unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "📐 Directional Accuracy",
        "📊 Stratified Analysis",
        "🔍 Feature Importance"
    ])

    # ── Tab 1: DA ───────────────────────────────────────────
    with tab1:
        st.subheader("Directional Accuracy — Baseline vs Dual-Stream")
        st.markdown("""
        Directional Accuracy measures the percentage of days the model
        correctly predicted whether tomorrow's price would go **up or down**.
        The dashed line at 50% represents random chance — any model below
        50% is worse than a coin flip.
        """)

        crops  = ['Corn', 'Rice', 'Wheat']
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
        fig.add_hline(
            y=50, line_dash="dash", line_color="black", line_width=2,
            annotation_text="Random chance (50%)",
            annotation_position="top right"
        )
        fig.update_layout(
            barmode='group',
            title="Directional Accuracy by Crop",
            yaxis_title="Directional Accuracy (%)",
            yaxis=dict(range=[20, 70]),
            template='plotly_white', height=430,
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="highlight-box">
        <strong>🔑 Key insight:</strong> The baseline rice model scores
        <strong>34.4%</strong> — substantially below random chance, meaning
        it predicts the wrong direction two out of every three days.
        The dual-stream model improves this to <strong>56.8%</strong> —
        a gain of <strong>+22.4 percentage points</strong>, confirmed highly
        statistically significant (proportions z-test: z = 8.12, p &lt; 0.0001).
        The crop-specific pattern — largest improvement for rice, minimal for
        corn and wheat — aligns with the theoretical prediction that the
        Indian Ocean Dipole most directly affects Asian rice-growing regions.
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 2: Stratified ───────────────────────────────────
    with tab2:
        st.subheader("Stratified Analysis — High-Volatility vs Stable Periods")
        st.markdown("""
        The test set is divided into **high-volatility** days
        (Russia-Ukraine conflict spike + 2023 strong positive IOD event,
        435 days) and **stable** days (151 days). Results show how each
        model performs in different market conditions.
        """)

        img_path = "figure_stratified_analysis.png"
        if os.path.exists(img_path):
            img = Image.open(img_path)
            st.image(img, use_column_width=True)
        else:
            st.warning("figure_stratified_analysis.png not found.")

        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Rice DA — High Volatility",
            "Baseline: 33.9%",
            "Dual-Stream: 58.8% (+24.9pp)"
        )
        col2.metric(
            "Rice DA — Stable Periods",
            "Baseline: 35.3%",
            "Dual-Stream: 51.3% (+16.0pp)"
        )
        col3.metric(
            "Interpretation",
            "Dual-stream strongest",
            "during external shocks"
        )

        st.markdown("""
        <div class="highlight-box">
        <strong>🔑 Key insight:</strong> The dual-stream model performs
        best precisely during the high-volatility structural break period
        (58.8% DA for rice) — the exact conditions the dissertation was
        designed to test. The baseline performs worst during this same period
        (33.9%) — well below random chance. This divergence of
        <strong>24.9 percentage points</strong> during external shocks
        directly validates the theoretical motivation of the research.
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 3: Feature Importance ───────────────────────────
    with tab3:
        st.subheader("XGBoost Feature Importance by Stream")
        st.markdown("""
        Feature importance measures which inputs XGBoost relied on most
        when making predictions. The **price stream** contains 45 days of
        price history. The **exogenous stream** contains 7 lagged
        climate and energy features (4 DMI lags + 3 WTI lags).
        """)

        img_path = "figure_feature_importance.png"
        if os.path.exists(img_path):
            img = Image.open(img_path)
            st.image(img, use_column_width=True)
        else:
            st.warning("figure_feature_importance.png not found.")

        crops = ['Corn', 'Rice', 'Wheat']
        price_imp = [IMPORTANCE[c]['Price Stream']     for c in crops]
        exog_imp  = [IMPORTANCE[c]['Exogenous Stream'] for c in crops]

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            name='Price History Stream', x=crops, y=price_imp,
            marker_color=COLOURS['Baseline'], opacity=0.85,
            text=[f"{v:.1f}%" for v in price_imp],
            textposition='inside', textfont=dict(color='white')
        ))
        fig2.add_trace(go.Bar(
            name='Exogenous Stream (DMI + WTI)', x=crops, y=exog_imp,
            marker_color=COLOURS['Dual-Stream'], opacity=0.85,
            text=[f"{v:.1f}%" for v in exog_imp],
            textposition='inside', textfont=dict(color='white')
        ))
        fig2.add_hline(
            y=50, line_dash="dash", line_color="black",
            annotation_text="Equal contribution (50%)"
        )
        fig2.update_layout(
            barmode='group',
            title="Feature Importance: Price Stream vs Exogenous Stream",
            yaxis_title="% of Total Importance",
            yaxis=dict(range=[35, 65]),
            template='plotly_white', height=400,
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("""
        <div class="finding-box">
        <strong>🔑 Key insight:</strong> The exogenous driver stream
        outweighs the price history stream for <strong>all three crops</strong>.
        Rice shows the highest exogenous contribution at <strong>56.3%</strong>.
        This confirms the model is genuinely using the climate and energy
        signals — not ignoring them. The features are used for directional
        prediction rather than magnitude prediction.
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
        '<div class="sub-header">Understanding the climate and '
        'macroeconomic signals used in the dual-stream model</div>',
        unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🌊 DMI — Climate Signal", "🛢️ WTI — Energy Signal"])

    with tab1:
        st.subheader("Dipole Mode Index (DMI) — Indian Ocean Dipole")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            The **Indian Ocean Dipole (IOD)** is a climate phenomenon
            measured by the Dipole Mode Index. It captures the difference
            in sea surface temperature between the western and eastern
            tropical Indian Ocean.

            **Positive IOD (DMI > 0.5):** Warm western Indian Ocean →
            drought across South and Southeast Asia → disrupted monsoon
            → reduced rice and wheat harvests → price increases.

            **Negative IOD (DMI < -0.5):** Cool western Indian Ocean →
            enhanced rainfall → better harvests → downward price pressure.

            **Why DMI over ONI?** The ONI captures Pacific Ocean conditions
            (El Niño / La Niña). DMI captures Indian Ocean conditions which
            are more geographically proximate to the Asian rice-growing
            regions whose production drives Chinese domestic prices.
            """)
        with col2:
            st.metric("Peak DMI in study", "0.964", "Oct 2019 — strong +IOD")
            st.metric("2023 peak DMI",     "0.946", "Sep 2023 — strong +IOD")
            st.metric("Negative minimum",  "-0.758", "Study period low")
            st.metric("Lag structure",
                      "30/60/90/180 days",
                      "4 lag features used")

        try:
            dmi = load_dmi_data()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dmi['date'], y=dmi['DMI'],
                fill='tozeroy',
                line=dict(color='#2980B9', width=1.5),
                fillcolor='rgba(41,128,185,0.15)',
                name='DMI',
                hovertemplate="DMI: %{y:.3f}<br>Date: %{x}<extra></extra>"
            ))
            fig.add_hline(
                y=0.5, line_dash="dash", line_color="red",
                line_width=1.5,
                annotation_text="Positive IOD threshold (+0.5)"
            )
            fig.add_hline(
                y=-0.5, line_dash="dash", line_color="green",
                line_width=1.5,
                annotation_text="Negative IOD threshold (-0.5)"
            )
            # Annotate key events
            fig.add_annotation(
                x='2019-10-01', y=0.964,
                text="2019 Strong +IOD<br>(DMI=0.964)",
                showarrow=True, arrowhead=2, arrowcolor='red',
                bgcolor='white', bordercolor='red'
            )
            fig.add_annotation(
                x='2023-09-01', y=0.946,
                text="2023 Strong +IOD<br>(DMI=0.946)",
                showarrow=True, arrowhead=2, arrowcolor='red',
                bgcolor='white', bordercolor='red'
            )
            fig.update_layout(
                title="Dipole Mode Index (DMI) — 2012 to 2023",
                yaxis_title="DMI Value",
                xaxis_title="Date",
                template='plotly_white',
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load DMI data for chart: {e}")

    with tab2:
        st.subheader("WTI Crude Oil Spot Price — Cushing, Oklahoma")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            **WTI (West Texas Intermediate)** crude oil is the primary
            US benchmark price for crude oil, published daily by the
            U.S. Energy Information Administration (EIA).

            **Two transmission channels to agricultural prices:**

            🔴 **Input cost channel:** Higher oil prices increase diesel
            fuel costs for farm machinery, nitrogen fertiliser production
            costs, and logistics costs — raising agricultural production costs.

            🔴 **Biofuel demand channel:** Higher oil prices make corn-derived
            ethanol more economically competitive, increasing demand from
            ethanol refineries and pushing corn prices up.

            **Why daily WTI over monthly CPI?** Daily frequency aligns
            naturally with the daily price data, eliminating the need for
            interpolation which would introduce artificial noise.
            """)
        with col2:
            st.metric("Study period max", "$123.64/bbl", "Jun 2022 — RU spike")
            st.metric("Study period min", "$8.91/bbl",   "Apr 2020 — COVID")
            st.metric("Notable fix",
                      "Apr 20 2020",
                      "-$36.98 → replaced with avg")
            st.metric("Lag structure",
                      "7 / 14 / 30 days",
                      "3 lag features used")

        try:
            wti = load_wti_data()
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=wti['date'], y=wti['WTI'],
                line=dict(color='#2C3E50', width=1),
                name='WTI Price',
                hovertemplate="WTI: $%{y:.2f}/bbl<br>Date: %{x}<extra></extra>"
            ))
            # Annotate key events
            fig2.add_annotation(
                x='2020-04-20', y=20,
                text="COVID-19 crash<br>(Apr 2020)",
                showarrow=True, arrowhead=2,
                bgcolor='white', bordercolor='orange'
            )
            fig2.add_annotation(
                x='2022-06-08', y=123.64,
                text="Russia-Ukraine spike<br>($123.64 peak)",
                showarrow=True, arrowhead=2,
                bgcolor='white', bordercolor='red'
            )
            fig2.update_layout(
                title="WTI Crude Oil Spot Price — 2012 to 2023 (USD/barrel)",
                yaxis_title="USD per Barrel",
                xaxis_title="Date",
                template='plotly_white',
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load WTI data for chart: {e}")

        st.markdown("""
        <div class="finding-box">
        <strong>Lag Structure Rationale:</strong> The biofuel demand channel
        (WTI → corn) operates at 7–14 day lags. The input cost channel
        (WTI → fertiliser → production costs) operates at 14–30 day lags.
        Three WTI lag features capture both channels: WTI_lag7, WTI_lag14,
        and WTI_lag30. DMI lags of 30/60/90/180 days capture the longer
        monsoon-harvest-supply chain for rice and wheat.
        </div>
        """, unsafe_allow_html=True)
