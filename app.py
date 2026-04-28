import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "newDataScripts"))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import shap
import warnings

warnings.filterwarnings("ignore")

from modeling import build_preprocessor, time_aware_train_test_split

# ── paths ──────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(__file__)
DATA_PATH = os.path.join(ROOT, "data", "v2", "model_ready.csv")

# ── spending category config ───────────────────────────────────────────────
SPENDING_COLS = [
    "entertainment", "food_dining", "gas_transport",
    "grocery_net", "grocery_pos", "health_fitness",
    "home", "kids_pets", "misc_net", "misc_pos",
    "personal_care", "shopping_net", "shopping_pos", "travel",
]
DISCRETIONARY_COLS = ["entertainment", "shopping_net", "shopping_pos", "travel"]
ESSENTIAL_COLS     = ["food_dining", "grocery_net", "grocery_pos", "home", "health_fitness"]

CATEGORY_LABELS = {
    "entertainment":  "Entertainment",
    "food_dining":    "Food & Dining",
    "gas_transport":  "Gas & Transport",
    "grocery_net":    "Grocery (Online)",
    "grocery_pos":    "Grocery (In-Store)",
    "health_fitness": "Health & Fitness",
    "home":           "Home",
    "kids_pets":      "Kids & Pets",
    "misc_net":       "Misc (Online)",
    "misc_pos":       "Misc (In-Store)",
    "personal_care":  "Personal Care",
    "shopping_net":   "Shopping (Online)",
    "shopping_pos":   "Shopping (In-Store)",
    "travel":         "Travel",
}

CATEGORY_GROUPS = {
    "Essentials": ["food_dining", "grocery_net", "grocery_pos", "health_fitness", "home"],
    "Transportation": ["gas_transport"],
    "Lifestyle": ["entertainment", "personal_care", "kids_pets"],
    "Shopping & Travel": ["shopping_net", "shopping_pos", "travel"],
    "Miscellaneous": ["misc_net", "misc_pos"],
}

CATEGORY_DEFAULTS = {
    "entertainment": 200.0, "food_dining": 400.0, "gas_transport": 150.0,
    "grocery_net": 100.0,   "grocery_pos": 300.0,  "health_fitness": 80.0,
    "home": 1200.0,         "kids_pets": 50.0,     "misc_net": 30.0,
    "misc_pos": 40.0,       "personal_care": 60.0, "shopping_net": 100.0,
    "shopping_pos": 150.0,  "travel": 50.0,
}


# ══════════════════════════════════════════════════════════════════════════
# Model training  (Optuna-tuned GB params from paper: Section 7-E)
# ══════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    df = pd.read_csv(DATA_PATH)

    X_train, _, y_train, _, _, _ = time_aware_train_test_split(df)
    preprocessor = build_preprocessor(X_train)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", GradientBoostingClassifier(
            n_estimators=190, learning_rate=0.084, max_depth=2,
            random_state=42
        ))
    ])
    pipeline.fit(X_train, y_train)

    explainer = shap.TreeExplainer(pipeline.named_steps["model"])

    df_ref = df[SPENDING_COLS + ["median_monthly_income"]].copy()

    return pipeline, X_train, explainer, df_ref


# ══════════════════════════════════════════════════════════════════════════
# Feature engineering — supports optional prior-month totals for time-series
# prior_totals: list of 1 or 2 floats, oldest-first  [T-2, T-1]
# ══════════════════════════════════════════════════════════════════════════
def engineer_row(spend: dict, income: float, train_cols: list,
                 prior_totals: list = None) -> pd.DataFrame:
    row = {col: spend.get(col, 0.0) for col in SPENDING_COLS}
    row["total_amount"]            = sum(row.values())
    row["median_monthly_income"]   = income

    vals  = np.array([row[c] for c in SPENDING_COLS])
    total = vals.sum()
    shares = vals / total if total else np.zeros(len(SPENDING_COLS))

    row["expense_to_income_ratio"] = row["total_amount"] / income if income else np.nan
    row["category_concentration"]  = float((shares ** 2).sum())
    for col in SPENDING_COLS:
        row[f"{col}_income_adj"] = row[col] / income if income else np.nan
    gn, gp = row["grocery_net"], row["grocery_pos"]
    row["grocery_online_ratio"] = gn / (gn + gp) if (gn + gp) > 0 else np.nan
    mn, mp = row["misc_net"], row["misc_pos"]
    row["misc_online_ratio"] = mn / (mn + mp) if (mn + mp) > 0 else np.nan

    # ── time-series features: computed when prior data is available ────
    current = row["total_amount"]
    if prior_totals and len(prior_totals) >= 1:
        last = prior_totals[-1]
        row["expense_mom_change"] = (current - last) / last if last > 0 else np.nan
    else:
        row["expense_mom_change"] = np.nan

    if prior_totals and len(prior_totals) >= 2:
        history = list(prior_totals) + [current]           # [T-2, T-1, T]
        row["rolling_3m_expense"]  = float(np.mean(history))
        mu = np.mean(history)
        row["spending_volatility"] = float(np.std(history) / mu) if mu > 0 else np.nan
    elif prior_totals and len(prior_totals) == 1:
        row["rolling_3m_expense"]  = float(np.mean([prior_totals[0], current]))
        row["spending_volatility"] = np.nan
    else:
        row["rolling_3m_expense"]  = np.nan
        row["spending_volatility"] = np.nan

    for col in SPENDING_COLS:
        row[f"trend_{col}"] = np.nan          # per-category trend needs prior category data

    # frequency / regularity
    row["active_categories"]       = int((vals > 0).sum())
    row["zero_spend_categories"]   = int((vals == 0).sum())
    row["spending_dispersion"]     = float(vals.std())
    row["category_consistency"]    = np.nan

    # habit indicators
    disc  = sum(row.get(c, 0) for c in DISCRETIONARY_COLS)
    essen = sum(row.get(c, 0) for c in ESSENTIAL_COLS)
    row["discretionary_total"]     = disc
    row["essential_total"]         = essen
    row["discretionary_ratio"]     = disc / essen if essen > 0 else np.nan
    row["monthly_savings"]         = income - row["total_amount"]
    row["savings_rate"]            = row["monthly_savings"] / income if income else np.nan
    row["binge_spending_flag"]     = 0
    row["binge_category"]          = None
    dom_idx = int(np.argmax(vals)) if total > 0 else 0
    row["dominant_category"]       = SPENDING_COLS[dom_idx]
    row["dominant_category_share"] = float(max(vals) / total) if total > 0 else 0.0

    df_row = pd.DataFrame([row])
    for c in train_cols:
        if c not in df_row.columns:
            df_row[c] = np.nan
    return df_row[train_cols]


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════
def get_feature_names(preprocessor, X):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if cat_cols:
        ohe = preprocessor.named_transformers_["cat"]["onehot"]
        return num_cols + ohe.get_feature_names_out(cat_cols).tolist()
    return num_cols


def prettify(name: str) -> str:
    for k, v in CATEGORY_LABELS.items():
        name = name.replace(k, v)
    return name.replace("_", " ").title()


def risk_color(prob: float) -> str:
    if prob < 0.35:  return "#27AE60"
    if prob < 0.60:  return "#F39C12"
    return "#E74C3C"


def risk_label(prob: float) -> str:
    if prob < 0.35:  return "Low Risk"
    if prob < 0.60:  return "Moderate Risk"
    return "High Risk"


def health_score(prob: float, spend: dict, income: float) -> tuple[int, str, str, dict]:
    total     = sum(spend.values()) or 1
    exp_ratio = total / income if income else 1.0
    sav_rate  = (income - total) / income if income else 0.0
    disc      = sum(spend.get(c, 0) for c in DISCRETIONARY_COLS)
    essen     = sum(spend.get(c, 0) for c in ESSENTIAL_COLS)
    disc_ratio = disc / essen if essen > 0 else 2.0

    savings_pts = max(0.0, min(35.0, (sav_rate / 0.20) * 35)) if sav_rate > 0 else 0.0
    expense_pts = max(0.0, min(35.0, (1.5 - exp_ratio) / 1.5 * 35))
    risk_pts    = (1 - prob) * 20
    disc_pts    = max(0.0, min(10.0, (2.0 - disc_ratio) / 2.0 * 10))

    score = int(savings_pts + expense_pts + risk_pts + disc_pts)

    if score >= 80:   grade, color = "Excellent", "#27AE60"
    elif score >= 60: grade, color = "Good",      "#2ECC71"
    elif score >= 40: grade, color = "Fair",       "#F39C12"
    elif score >= 20: grade, color = "Poor",       "#E67E22"
    else:             grade, color = "Critical",   "#E74C3C"

    components = {
        "Savings Rate":          round(savings_pts, 1),
        "Expense Control":       round(expense_pts, 1),
        "Low Overspend Risk":    round(risk_pts, 1),
        "Discretionary Balance": round(disc_pts, 1),
    }
    return score, grade, color, components


# ══════════════════════════════════════════════════════════════════════════
# UI sections
# ══════════════════════════════════════════════════════════════════════════
def render_input_form() -> tuple[dict, float, list]:
    st.markdown("### Enter Your Monthly Financials")
    st.markdown(
        "Fill in your estimated spending for the current month along with your income. "
        "We'll analyze your financial behavior and predict your overspending risk for next month."
    )

    income = st.number_input(
        "Monthly Income ($)",
        min_value=500.0, max_value=50000.0,
        value=5000.0, step=100.0,
        help="Your total take-home income this month",
    )

    st.markdown("---")
    st.markdown("**Monthly Spending by Category**")

    spend = {}
    for group_name, cols in CATEGORY_GROUPS.items():
        with st.expander(group_name, expanded=(group_name in ("Essentials", "Lifestyle", "Shopping & Travel"))):
            gcols = st.columns(min(len(cols), 3))
            for i, col in enumerate(cols):
                spend[col] = gcols[i % 3].number_input(
                    CATEGORY_LABELS[col],
                    min_value=0.0, max_value=20000.0,
                    value=CATEGORY_DEFAULTS[col],
                    step=10.0, key=f"input_{col}",
                )

    # ── optional prior-month history ───────────────────────────────────
    st.markdown("---")
    with st.expander("📅 Prior Month History — Optional (improves trend accuracy)"):
        st.caption(
            "Providing prior months lets the analysis compute your spending trend, "
            "month-over-month change, and volatility — making the prediction more accurate."
        )
        pc1, pc2 = st.columns(2)
        last_month = pc1.number_input(
            "Last Month — Total Spending ($)",
            min_value=0.0, max_value=50000.0, value=0.0, step=50.0,
            key="prior_last",
            help="Your total spending across all categories last month",
        )
        two_ago = pc2.number_input(
            "Two Months Ago — Total Spending ($)",
            min_value=0.0, max_value=50000.0, value=0.0, step=50.0,
            key="prior_two",
            help="Your total spending two months ago (requires Last Month to be filled in)",
        )

    prior_totals = []
    if two_ago > 0 and last_month > 0:
        prior_totals = [two_ago, last_month]
    elif last_month > 0:
        prior_totals = [last_month]

    total_spend = sum(spend.values())
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Spending",  f"${total_spend:,.0f}")
    c2.metric("Monthly Income",  f"${income:,.0f}")
    net = income - total_spend
    c3.metric("Net Balance", f"${net:,.0f}",
              delta="Surplus" if net >= 0 else "Deficit",
              delta_color="normal" if net >= 0 else "inverse")

    return spend, income, prior_totals


def render_risk_banner(prob: float, savings: float, exp_ratio: float):
    color = risk_color(prob)
    label = risk_label(prob)

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}18, {color}08);
        border-left: 5px solid {color};
        border-radius: 10px;
        padding: 20px 28px;
        margin-bottom: 8px;
    ">
        <div style="font-size: 1.0rem; color: #555; margin-bottom: 4px;">
            Overspending Risk — Next Month
        </div>
        <div style="font-size: 2.8rem; font-weight: 700; color: {color};">
            {prob:.0%} &nbsp;
            <span style="font-size: 1.1rem; font-weight: 500; color: {color};">
                {label}
            </span>
        </div>
        <div style="font-size: 0.85rem; color: #888; margin-top: 4px;">
            Based on your current spending patterns and behavioral indicators
        </div>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Expense-to-Income Ratio", f"{exp_ratio:.1%}",
              delta="Over budget" if exp_ratio > 1 else "Within budget",
              delta_color="inverse" if exp_ratio > 1 else "normal")
    m2.metric("Estimated Savings", f"${savings:,.0f}",
              delta="Positive" if savings >= 0 else "Negative",
              delta_color="normal" if savings >= 0 else "inverse")
    m3.metric("Risk Threshold", "50%",
              delta=f"You are {'above' if prob >= 0.5 else 'below'} threshold",
              delta_color="inverse" if prob >= 0.5 else "normal")


def render_gauge(prob: float):
    color = risk_color(prob)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 40, "color": color}},
        title={"text": "Overspend Probability", "font": {"size": 15, "color": "#555"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#ccc"},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 35],  "color": "#E8F5E9"},
                {"range": [35, 60], "color": "#FFF8E1"},
                {"range": [60, 100], "color": "#FFEBEE"},
            ],
            "threshold": {
                "line": {"color": "#333", "width": 3},
                "thickness": 0.75,
                "value": 50,
            },
        },
    ))
    fig.update_layout(height=250, margin=dict(t=40, b=0, l=20, r=20),
                      paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)


def render_health_score(prob: float, spend: dict, income: float):
    score, grade, color, components = health_score(prob, spend, income)

    st.markdown("#### Financial Health Score")
    st.markdown(f"""
    <div style="
        text-align: center;
        background: linear-gradient(135deg, {color}15, {color}05);
        border: 2px solid {color}55;
        border-radius: 12px;
        padding: 16px 10px 10px 10px;
        margin-bottom: 10px;
    ">
        <div style="font-size: 3.6rem; font-weight: 800; color: {color}; line-height: 1;">
            {score}
        </div>
        <div style="font-size: 0.75rem; color: #888; margin-top: 2px;">out of 100</div>
        <div style="
            display: inline-block;
            background: {color};
            color: white;
            font-size: 0.85rem;
            font-weight: 600;
            padding: 3px 14px;
            border-radius: 20px;
            margin-top: 6px;
        ">{grade}</div>
    </div>
    """, unsafe_allow_html=True)

    max_pts = {"Savings Rate": 35, "Expense Control": 35,
               "Low Overspend Risk": 20, "Discretionary Balance": 10}
    for label, pts in components.items():
        pct = pts / max_pts[label]
        bar_color = color if pct >= 0.6 else ("#F39C12" if pct >= 0.3 else "#E74C3C")
        st.markdown(f"""
        <div style="margin-bottom: 7px;">
            <div style="display:flex; justify-content:space-between;
                        font-size:0.78rem; color:#555; margin-bottom:2px;">
                <span>{label}</span>
                <span>{pts:.0f} / {max_pts[label]}</span>
            </div>
            <div style="background:#eee; border-radius:4px; height:8px;">
                <div style="background:{bar_color}; width:{pct*100:.0f}%;
                            height:8px; border-radius:4px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_spending_breakdown(spend: dict):
    st.markdown("#### Spending Breakdown")
    nonzero = {CATEGORY_LABELS[k]: v for k, v in spend.items() if v > 0}
    if not nonzero:
        st.info("No spending entered.")
        return

    fig = px.pie(
        names=list(nonzero.keys()),
        values=list(nonzero.values()),
        color_discrete_sequence=px.colors.qualitative.Set3,
        hole=0.4,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(
        showlegend=False, height=300,
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(
            text=f"${sum(spend.values()):,.0f}",
            x=0.5, y=0.5, font_size=16, showarrow=False, font_color="#333"
        )]
    )
    st.plotly_chart(fig, use_container_width=True)

    disc  = sum(spend.get(c, 0) for c in DISCRETIONARY_COLS)
    essen = sum(spend.get(c, 0) for c in ESSENTIAL_COLS)
    t     = sum(spend.values()) or 1
    st.markdown(
        f"**Discretionary:** ${disc:,.0f} ({disc/t:.0%})  "
        f"**Essential:** ${essen:,.0f} ({essen/t:.0%})",
        unsafe_allow_html=True,
    )


def render_shap_explanation(sv: np.ndarray, feature_names: list):
    st.markdown("#### Why This Prediction Was Made")
    st.caption("Each bar shows how much a factor pushed your risk higher (red) or lower (blue).")

    shap_df = pd.DataFrame({"feature": [prettify(f) for f in feature_names], "value": sv})
    shap_df = shap_df.reindex(shap_df["value"].abs().sort_values(ascending=False).index)
    shap_df = shap_df.head(12).reset_index(drop=True)

    colors = ["#E74C3C" if v > 0 else "#3498DB" for v in shap_df["value"]]
    fig = go.Figure(go.Bar(
        x=shap_df["value"], y=shap_df["feature"],
        orientation="h", marker_color=colors,
        text=[f"{'+' if v > 0 else ''}{v:.3f}" for v in shap_df["value"]],
        textposition="outside",
    ))
    fig.update_layout(
        xaxis_title="Impact on prediction",
        yaxis=dict(autorange="reversed"),
        height=420,
        margin=dict(t=10, b=20, l=10, r=60),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(zeroline=True, zerolinecolor="#ccc", zerolinewidth=1.5,
                   gridcolor="#f0f0f0"),
    )
    st.plotly_chart(fig, use_container_width=True)

    risk_factors = shap_df[shap_df["value"] > 0].head(3)
    protective   = shap_df[shap_df["value"] < 0].head(3)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Increasing your risk:**")
        for _, r in risk_factors.iterrows():
            st.markdown(f"- {r.feature}")
    with col2:
        st.markdown("**Reducing your risk:**")
        for _, r in protective.iterrows():
            st.markdown(f"- {r.feature}")


def _peer_averages(df_ref: pd.DataFrame, income: float, window: float = 0.25):
    """Return (averages_dict, peer_count) for people within ±window of income."""
    lo, hi = income * (1 - window), income * (1 + window)
    peers  = df_ref[df_ref["median_monthly_income"].between(lo, hi)]
    if len(peers) < 15:                      # widen bracket if too few peers
        lo, hi = income * 0.5, income * 1.5
        peers  = df_ref[df_ref["median_monthly_income"].between(lo, hi)]
    avgs = {col: float(peers[col].mean()) for col in SPENDING_COLS}
    return avgs, len(peers), lo, hi


def render_benchmarks(spend: dict, income: float, df_ref: pd.DataFrame):
    peer_avgs, n_peers, lo, hi = _peer_averages(df_ref, income)

    st.markdown("#### How You Compare to Similar Earners")
    st.caption(
        f"Compared against **{n_peers:,} individuals** earning "
        f"${lo:,.0f}–${hi:,.0f}/month — people in a similar income bracket to you."
    )

    rows = []
    for col in SPENDING_COLS:
        user_val = spend.get(col, 0.0)
        avg_val  = peer_avgs.get(col, 0.0)
        diff_pct = ((user_val - avg_val) / avg_val * 100) if avg_val else 0
        rows.append({"Category": CATEGORY_LABELS[col], "You ($)": user_val,
                     "Peer Avg ($)": round(avg_val, 0), "vs. Peers": diff_pct})

    bench_df   = pd.DataFrame(rows).sort_values("vs. Peers", ascending=False)
    colors_bar = ["#E74C3C" if v > 0 else "#27AE60" for v in bench_df["vs. Peers"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bench_df["Category"], y=bench_df["vs. Peers"],
        marker_color=colors_bar,
        text=[f"{v:+.0f}%" for v in bench_df["vs. Peers"]],
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "You: $%{customdata[0]:,.0f}<br>"
            "Peer avg: $%{customdata[1]:,.0f}<br>"
            "Difference: %{y:+.1f}%<extra></extra>"
        ),
        customdata=bench_df[["You ($)", "Peer Avg ($)"]].values,
    ))
    fig.add_hline(y=0, line_color="#555", line_width=1)
    fig.update_layout(
        yaxis_title="% vs. Peer Average", xaxis_tickangle=-35, height=360,
        margin=dict(t=10, b=80),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(gridcolor="#f0f0f0"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_insights(spend: dict, income: float, prob: float,
                    sv: np.ndarray, feature_names: list):
    total      = sum(spend.values())
    exp_ratio  = total / income if income else 0
    savings    = income - total
    sav_rate   = savings / income if income else 0
    disc       = sum(spend.get(c, 0) for c in DISCRETIONARY_COLS)
    essen      = sum(spend.get(c, 0) for c in ESSENTIAL_COLS)
    disc_ratio = disc / essen if essen else 0

    st.markdown("#### Personalized Insights")
    insights = []

    if exp_ratio > 1.0:
        insights.append(("🔴", "Spending exceeds income",
            f"Your expenses (${total:,.0f}) are **{(exp_ratio-1)*100:.0f}% above** your income. "
            "Immediate budget cuts are advised."))
    elif exp_ratio > 0.80:
        insights.append(("🟡", "Spending is high relative to income",
            f"You're spending **{exp_ratio:.0%}** of your income, leaving little buffer."))
    else:
        insights.append(("🟢", "Healthy expense-to-income ratio",
            f"Your spending is **{exp_ratio:.0%}** of income — a healthy buffer."))

    if sav_rate < 0:
        insights.append(("🔴", "Negative savings rate",
            f"You're running a **${abs(savings):,.0f} deficit** this month."))
    elif sav_rate < 0.10:
        insights.append(("🟡", "Low savings rate",
            f"Only **{sav_rate:.0%}** of income saved. Advisors recommend 20%+."))
    else:
        insights.append(("🟢", f"Saving {sav_rate:.0%} of income",
            f"You're saving **${savings:,.0f}** this month — solid financial discipline."))

    if disc_ratio > 1.5:
        insights.append(("🟡", "High discretionary spending",
            f"Discretionary spending (${disc:,.0f}) is **{disc_ratio:.1f}× your essentials** "
            f"(${essen:,.0f})."))
    elif disc_ratio < 0.3 and disc > 0:
        insights.append(("🟢", "Minimal discretionary spending",
            "Your discretionary spending is well controlled relative to essentials."))

    shap_df  = pd.DataFrame({"feature": [prettify(f) for f in feature_names], "value": sv})
    top_risk = shap_df[shap_df["value"] > 0].sort_values("value", ascending=False)
    if not top_risk.empty:
        top_name = top_risk.iloc[0]["feature"]
        insights.append(("⚠️", f"Primary risk driver: {top_name}",
            f"**{top_name}** is the single biggest factor increasing your overspend risk. "
            "Use the What-If Simulator below to explore how reducing it would change your score."))

    dom_col = max(SPENDING_COLS, key=lambda c: spend.get(c, 0))
    dom_pct = spend.get(dom_col, 0) / total * 100 if total else 0
    if dom_pct > 30:
        insights.append(("ℹ️", f"Concentrated spending in {CATEGORY_LABELS[dom_col]}",
            f"**{CATEGORY_LABELS[dom_col]}** accounts for **{dom_pct:.0f}%** of total spending."))

    for icon, title, body in insights:
        st.markdown(f"""
        <div style="background:#fafafa; border-radius:8px; padding:14px 18px;
                    margin-bottom:10px; border:1px solid #ececec;">
            <div style="font-weight:600; margin-bottom:4px;">{icon} {title}</div>
            <div style="color:#555; font-size:0.88rem;">{body}</div>
        </div>
        """, unsafe_allow_html=True)


def render_whatif(spend: dict, income: float, pipeline, train_cols: list,
                  current_prob: float, prior_totals: list):
    st.markdown("#### What-If Simulator")
    st.caption(
        "See how adjusting a single spending category would change your overspend probability."
    )

    wi_col, wi_slider_col = st.columns([1, 2])

    with wi_col:
        cat_choice = st.selectbox(
            "Category to adjust",
            options=SPENDING_COLS,
            format_func=lambda k: CATEGORY_LABELS[k],
            key="whatif_cat",
        )

    with wi_slider_col:
        adjustment = st.slider(
            "Adjustment (%)",
            min_value=-80, max_value=100, value=0, step=5,
            key="whatif_slider",
            help="Negative = reduce spending, positive = increase spending",
        )

    # ── precompute probability curve across range ──────────────────────
    steps   = list(range(-80, 105, 5))
    probs   = []
    for pct in steps:
        modified = dict(spend)
        modified[cat_choice] = max(0.0, spend.get(cat_choice, 0) * (1 + pct / 100))
        X_mod = engineer_row(modified, income, train_cols, prior_totals)
        probs.append(float(pipeline.predict_proba(X_mod)[0, 1]))

    # adjusted probability at selected slider value
    modified_spend = dict(spend)
    modified_spend[cat_choice] = max(0.0, spend.get(cat_choice, 0) * (1 + adjustment / 100))
    X_adj      = engineer_row(modified_spend, income, train_cols, prior_totals)
    adj_prob   = float(pipeline.predict_proba(X_adj)[0, 1])
    delta_prob = adj_prob - current_prob

    # ── metrics row ────────────────────────────────────────────────────
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Current Probability",  f"{current_prob:.1%}")
    mc2.metric("Adjusted Probability", f"{adj_prob:.1%}",
               delta=f"{delta_prob:+.1%}",
               delta_color="inverse")
    old_amt = spend.get(cat_choice, 0)
    new_amt = max(0.0, old_amt * (1 + adjustment / 100))
    mc3.metric(f"{CATEGORY_LABELS[cat_choice]} Spend",
               f"${new_amt:,.0f}",
               delta=f"${new_amt - old_amt:+,.0f}",
               delta_color="inverse")

    # ── probability curve chart ────────────────────────────────────────
    fig = go.Figure()

    # risk zone bands
    fig.add_hrect(y0=0, y1=0.35,   fillcolor="#E8F5E9", opacity=0.3, line_width=0)
    fig.add_hrect(y0=0.35, y1=0.60, fillcolor="#FFF8E1", opacity=0.3, line_width=0)
    fig.add_hrect(y0=0.60, y1=1.0,  fillcolor="#FFEBEE", opacity=0.3, line_width=0)

    fig.add_trace(go.Scatter(
        x=steps, y=probs, mode="lines",
        line=dict(color="#4A90D9", width=2.5),
        name="Overspend Probability",
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="#888", line_width=1,
                  annotation_text="50% threshold", annotation_position="right")

    # current point (0%)
    fig.add_trace(go.Scatter(
        x=[0], y=[current_prob], mode="markers",
        marker=dict(size=10, color="#555", symbol="circle"),
        name="Current",
    ))
    # adjusted point
    fig.add_trace(go.Scatter(
        x=[adjustment], y=[adj_prob], mode="markers",
        marker=dict(size=12, color=risk_color(adj_prob), symbol="star"),
        name=f"At {adjustment:+d}%",
    ))

    fig.update_layout(
        xaxis_title=f"Adjustment to {CATEGORY_LABELS[cat_choice]} (%)",
        yaxis_title="Overspend Probability",
        yaxis=dict(range=[0, 1], tickformat=".0%", gridcolor="#f0f0f0"),
        xaxis=dict(gridcolor="#f0f0f0"),
        height=340,
        margin=dict(t=20, b=40, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── plain-language takeaway ────────────────────────────────────────
    if adjustment < 0 and delta_prob < -0.03:
        st.success(
            f"Reducing **{CATEGORY_LABELS[cat_choice]}** by {abs(adjustment)}% "
            f"(saving **${old_amt - new_amt:,.0f}/month**) would lower your overspend "
            f"probability by **{abs(delta_prob):.1%}**."
        )
    elif adjustment > 0 and delta_prob > 0.03:
        st.warning(
            f"Increasing **{CATEGORY_LABELS[cat_choice]}** by {adjustment}% "
            f"(an extra **${new_amt - old_amt:,.0f}/month**) would raise your overspend "
            f"probability by **{delta_prob:.1%}**."
        )
    else:
        st.info(
            f"Adjusting **{CATEGORY_LABELS[cat_choice]}** by {adjustment:+d}% "
            f"has a minimal effect on your overspend probability ({delta_prob:+.1%})."
        )


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(
        page_title="FinancialBehaviorXAI — Financial Health Analyzer",
        page_icon="💳",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown("""
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }
        h1 { font-weight: 700; }
        h3 { font-weight: 600; color: #1a1a2e; }
        h4 { font-weight: 600; color: #333; margin-top: 1.5rem; }
        [data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; }
        .stExpander > summary { font-weight: 600; }
        div[data-testid="stNumberInput"] label { font-size: 0.82rem; }
    </style>
    """, unsafe_allow_html=True)

    col_logo, col_title = st.columns([1, 10])
    col_logo.markdown("<div style='font-size:2.8rem;padding-top:8px'>💳</div>",
                      unsafe_allow_html=True)
    col_title.markdown(
        "<h1 style='margin-bottom:2px'>FinancialBehaviorXAI</h1>"
        "<p style='color:#888;margin-top:0'>AI-powered financial behavior analysis "
        "&amp; overspending prediction</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    with st.spinner("Loading analysis engine…"):
        pipeline, X_train, explainer, df_ref = load_model()

    train_cols = X_train.columns.tolist()

    left, right = st.columns([1, 1.6], gap="large")

    with left:
        spend, income, prior_totals = render_input_form()
        st.markdown("")
        run = st.button("Analyze My Spending", type="primary", use_container_width=True)

    with right:
        if not run and "last_result" not in st.session_state:
            st.markdown(
                "<div style='text-align:center;color:#bbb;padding:80px 0;font-size:1.1rem'>"
                "← Enter your financials and click <strong>Analyze</strong> to see your results"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            if run:
                X_row = engineer_row(spend, income, train_cols, prior_totals)
                prob  = float(pipeline.predict_proba(X_row)[0, 1])

                preprocessor = pipeline.named_steps["preprocessor"]
                X_arr  = preprocessor.transform(X_row)
                sv     = explainer.shap_values(X_arr)[0]
                fnames = get_feature_names(preprocessor, X_row)

                st.session_state["last_result"] = {
                    "prob": prob, "spend": spend, "income": income,
                    "sv": sv, "fnames": fnames, "prior_totals": prior_totals,
                }

            res          = st.session_state["last_result"]
            prob         = res["prob"]
            spend        = res["spend"]
            income       = res["income"]
            sv           = res["sv"]
            fnames       = res["fnames"]
            prior_totals = res["prior_totals"]

            total     = sum(spend.values())
            exp_ratio = total / income if income else 0
            savings   = income - total

            # ── risk banner ────────────────────────────────────────────
            render_risk_banner(prob, savings, exp_ratio)
            st.markdown("")

            # ── gauge | health score | spending pie ───────────────────
            g1, g2, g3 = st.columns(3)
            with g1:
                render_gauge(prob)
            with g2:
                render_health_score(prob, spend, income)
            with g3:
                render_spending_breakdown(spend)

            st.divider()

            # ── SHAP explanation ───────────────────────────────────────
            render_shap_explanation(sv, fnames)

            st.divider()

            # ── benchmark vs. dataset averages ────────────────────────
            render_benchmarks(spend, income, df_ref)

            st.divider()

            # ── personalized insights ──────────────────────────────────
            render_insights(spend, income, prob, sv, fnames)

            st.divider()

            # ── what-if simulator ──────────────────────────────────────
            render_whatif(spend, income, pipeline, train_cols,
                          prob, prior_totals)


if __name__ == "__main__":
    main()
