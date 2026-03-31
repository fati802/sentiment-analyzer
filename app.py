import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from supabase import create_client
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="Review Intelligence", layout="wide", page_icon=None)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&family=Playfair+Display:wght@700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main {
        background-color: #caf0f8;
    }

    .block-container {
        padding: 2rem 3rem 4rem 3rem;
        max-width: 1400px;
    }

    section[data-testid="stSidebar"] {
        background-color: #03045e;
        border-right: none;
        padding-top: 2rem;
    }

    section[data-testid="stSidebar"] * {
        color: #90e0ef !important;
    }

    section[data-testid="stSidebar"] .stRadio > label {
        font-family: 'DM Mono', monospace !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #90e0ef !important;
        margin-bottom: 0.5rem;
    }

    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
        gap: 2px;
    }

    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.95rem !important;
        font-weight: 400 !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        color: #caf0f8 !important;
        padding: 8px 12px;
        border-radius: 4px;
        transition: background 0.15s;
    }

    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
        background: rgba(0,180,216,0.15);
    }

    .sidebar-brand {
        font-family: 'Playfair Display', serif;
        font-size: 1.25rem;
        font-weight: 800;
        color: #caf0f8 !important;
        letter-spacing: -0.01em;
        padding: 0 1rem 1.5rem 1rem;
        border-bottom: 1px solid #0077b6;
        margin-bottom: 1.5rem;
    }

    .sidebar-stat {
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
        color: #0077b6 !important;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 0 1rem;
        line-height: 1.6;
    }

    .sidebar-stat span {
        color: #caf0f8 !important;
        font-weight: 500;
    }

    .page-header {
        margin-bottom: 2.5rem;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid #90e0ef;
    }

    .page-header h1 {
        font-family: 'Playfair Display', serif;
        font-size: 2.6rem;
        font-weight: 800;
        color: #03045e;
        line-height: 1.1;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.02em;
    }

    .page-header p {
        font-family: 'DM Mono', monospace;
        font-size: 0.75rem;
        color: #0077b6;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin: 0;
    }

    .stat-block {
        background: #ffffff;
        border: 1px solid #90e0ef;
        border-radius: 2px;
        padding: 1.5rem 1.75rem;
    }

    .stat-block .label {
        font-family: 'DM Mono', monospace;
        font-size: 0.68rem;
        color: #0077b6;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    .stat-block .value {
        font-family: 'Playfair Display', serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #03045e;
        line-height: 1;
    }

    .stat-block .sub {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.8rem;
        color: #00b4d8;
        margin-top: 0.3rem;
    }

    .stat-block.accent {
        background: #03045e;
        border-color: #03045e;
    }

    .stat-block.accent .label {
        color: #90e0ef;
    }

    .stat-block.accent .value {
        color: #caf0f8;
    }

    .stat-block.accent .sub {
        color: #0077b6;
    }

    .chart-panel {
        background: #ffffff;
        border: 1px solid #90e0ef;
        border-radius: 2px;
        padding: 1.75rem;
        margin-bottom: 1.5rem;
    }

    .chart-panel .panel-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.68rem;
        color: #0077b6;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 1.25rem;
    }

    .predict-box {
        background: #ffffff;
        border: 1px solid #90e0ef;
        border-radius: 2px;
        padding: 2rem;
    }

    .result-strip {
        border-left: 4px solid #03045e;
        padding: 1.25rem 1.5rem;
        background: #caf0f8;
        margin-top: 1.5rem;
    }

    .result-strip .result-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.68rem;
        color: #0077b6;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }

    .result-strip .result-value {
        font-family: 'Playfair Display', serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #03045e;
    }

    .result-strip .result-score {
        font-family: 'DM Mono', monospace;
        font-size: 0.78rem;
        color: #0077b6;
        margin-top: 0.3rem;
    }

    .stTextArea textarea {
        background: #caf0f8 !important;
        border: 1px solid #90e0ef !important;
        border-radius: 2px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.95rem !important;
        color: #03045e !important;
        resize: vertical;
    }

    .stTextArea textarea:focus {
        border-color: #0077b6 !important;
        box-shadow: none !important;
    }

    .stButton > button {
        background: #03045e !important;
        color: #caf0f8 !important;
        border: none !important;
        border-radius: 2px !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        padding: 0.65rem 2rem !important;
        font-weight: 500 !important;
        transition: background 0.15s !important;
    }

    .stButton > button:hover {
        background: #0077b6 !important;
        color: #caf0f8 !important;
    }

    div[data-testid="stMetric"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    .stPlotlyChart {
        border-radius: 0 !important;
    }

    hr {
        border: none;
        border-top: 1px solid #90e0ef;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Data Loading from Supabase ──
SUPABASE_URL = "https://lfsgvmfojkkjkwqwarlm.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imxmc2d2bWZvamtramt3cXdhcmxtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzQ5NDMxODMsImV4cCI6MjA5MDUxOTE4M30.ADtZAEJLJU1qTu4sorzA__DeKP5igTtscP5Y9z_lR04"

@st.cache_data
def load_data():
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    all_rows = []
    batch_size = 1000
    offset = 0
    while True:
        response = supabase.table('reviews').select('*').range(offset, offset + batch_size - 1).execute()
        rows = response.data
        if not rows:
            break
        all_rows.extend(rows)
        offset += batch_size
    df = pd.DataFrame(all_rows)
    df = df.rename(columns={
        'id': 'Id',
        'productid': 'ProductId',
        'score': 'Score',
        'sentiment': 'Sentiment',
        'date': 'Date',
        'summary': 'Summary',
        'text': 'Text'
    })
    return df

df = load_data()

with open('evaluation_results.json', 'r') as f:
    eval_results = json.load(f)

sentiment_counts = df['Sentiment'].value_counts()

CHART_COLORS = {
    'Positive': '#0077b6',
    'Negative': '#03045e',
    'Neutral':  '#90e0ef',
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='DM Sans, sans-serif', color='#03045e', size=12),
    margin=dict(t=10, b=10, l=10, r=10),
    showlegend=True,
    legend=dict(
        bgcolor='rgba(0,0,0,0)',
        bordercolor='rgba(0,0,0,0)',
        font=dict(size=11),
    )
)


# ── Sidebar ──
with st.sidebar:
    st.markdown("<div class='sidebar-brand'>Review<br>Intelligence</div>", unsafe_allow_html=True)
    page = st.radio("", ["Overview", "Model Performance", "Trends", "Predict"])
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='sidebar-stat'>
        Total Reviews<br><span>{len(df):,}</span><br>
        Best Model<br><span>LR + TF-IDF</span><br>
        Best Accuracy<br><span>{eval_results['LR_TF-IDF']*100:.2f}%</span><br>
        Positive Rate<br><span>{sentiment_counts.get('Positive',0)/len(df)*100:.1f}%</span>
    </div>
    """, unsafe_allow_html=True)


# ── PAGE: Overview ──
if page == "Overview":
    st.markdown("""
    <div class='page-header'>
        <h1>Customer Sentiment<br>at a Glance</h1>
        <p>Amazon Fine Food Reviews — Sentiment Analysis Report</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class='stat-block accent'>
            <div class='label'>Total Reviews</div>
            <div class='value'>{len(df)/1000:.0f}K</div>
            <div class='sub'>in dataset</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='stat-block'>
            <div class='label'>Positive</div>
            <div class='value'>{sentiment_counts.get('Positive',0)/1000:.0f}K</div>
            <div class='sub'>{sentiment_counts.get('Positive',0)/len(df)*100:.1f}% of total</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='stat-block'>
            <div class='label'>Negative</div>
            <div class='value'>{sentiment_counts.get('Negative',0)/1000:.0f}K</div>
            <div class='sub'>{sentiment_counts.get('Negative',0)/len(df)*100:.1f}% of total</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class='stat-block'>
            <div class='label'>Neutral</div>
            <div class='value'>{sentiment_counts.get('Neutral',0)/1000:.0f}K</div>
            <div class='sub'>{sentiment_counts.get('Neutral',0)/len(df)*100:.1f}% of total</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown("<div class='chart-panel'><div class='panel-label'>Sentiment Split</div>", unsafe_allow_html=True)
        fig_pie = go.Figure(data=[go.Pie(
            labels=sentiment_counts.index.tolist(),
            values=sentiment_counts.values.tolist(),
            hole=0.55,
            marker_colors=[CHART_COLORS.get(l, '#00b4d8') for l in sentiment_counts.index],
            textfont=dict(family='DM Mono, monospace', size=11),
            hovertemplate='%{label}<br>%{value:,} reviews<br>%{percent}<extra></extra>',
        )])
        fig_pie.update_layout(**PLOTLY_LAYOUT, height=280)
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown("<div class='chart-panel'><div class='panel-label'>Reviews by Star Rating</div>", unsafe_allow_html=True)
        score_counts = df['Score'].value_counts().sort_index().reset_index()
        score_counts.columns = ['Score', 'Count']
        fig_bar = go.Figure(data=[go.Bar(
            x=score_counts['Score'],
            y=score_counts['Count'],
            marker_color=['#03045e', '#0077b6', '#00b4d8', '#90e0ef', '#caf0f8'],
            hovertemplate='%{x} stars: %{y:,}<extra></extra>',
        )])
        fig_bar.update_layout(**PLOTLY_LAYOUT, height=280,
                              xaxis=dict(tickvals=[1,2,3,4,5], ticktext=['1 star','2','3','4','5 stars']),
                              yaxis=dict(showgrid=True, gridcolor='#caf0f8'))
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ── PAGE: Model Performance ──
elif page == "Model Performance":
    st.markdown("""
    <div class='page-header'>
        <h1>Model Performance</h1>
        <p>Accuracy comparison across three classification approaches</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    models_data = [
        ("VADER", eval_results['VADER'], "Rule-based lexicon"),
        ("LR + BoW", eval_results['LR_BoW'], "Bag of Words features"),
        ("LR + TF-IDF", eval_results['LR_TF-IDF'], "TF-IDF features — best"),
    ]
    for col, (name, acc, desc) in zip([c1, c2, c3], models_data):
        is_best = name == "LR + TF-IDF"
        with col:
            st.markdown(f"""
            <div class='stat-block {"accent" if is_best else ""}'>
                <div class='label'>{name}</div>
                <div class='value'>{acc*100:.2f}%</div>
                <div class='sub'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<div class='chart-panel'><div class='panel-label'>Accuracy Comparison</div>", unsafe_allow_html=True)
    names = [m[0] for m in models_data]
    accs  = [m[1]*100 for m in models_data]
    fig_acc = go.Figure(data=[go.Bar(
        x=names,
        y=accs,
        marker_color=['#90e0ef', '#0077b6', '#03045e'],
        text=[f"{a:.2f}%" for a in accs],
        textposition='outside',
        textfont=dict(family='DM Mono, monospace', size=11, color='#03045e'),
        hovertemplate='%{x}: %{y:.2f}%<extra></extra>',
    )])
    fig_acc.update_layout(**PLOTLY_LAYOUT, height=360,
                          yaxis=dict(range=[70, 95], showgrid=True, gridcolor='#caf0f8',
                                     ticksuffix='%'),
                          bargap=0.5)
    st.plotly_chart(fig_acc, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ── PAGE: Trends ──
elif page == "Trends":
    st.markdown("""
    <div class='page-header'>
        <h1>Sentiment Over Time</h1>
        <p>Monthly review volume by sentiment class</p>
    </div>
    """, unsafe_allow_html=True)

    df['Date'] = pd.to_datetime(df['Date'])
    df['YearMonth'] = df['Date'].dt.to_period('M').astype(str)
    monthly = df.groupby(['YearMonth', 'Sentiment']).size().reset_index(name='count')

    st.markdown("<div class='chart-panel'><div class='panel-label'>Monthly Sentiment Volume</div>", unsafe_allow_html=True)
    fig_line = px.line(
        monthly, x='YearMonth', y='count', color='Sentiment',
        color_discrete_map=CHART_COLORS,
    )
    fig_line.update_traces(line_width=2)
    fig_line.update_layout(**PLOTLY_LAYOUT, height=400,
                           xaxis=dict(showgrid=False, tickangle=45, tickfont=dict(size=10)),
                           yaxis=dict(showgrid=True, gridcolor='#caf0f8'))
    st.plotly_chart(fig_line, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ── PAGE: Predict ──
elif page == "Predict":
    st.markdown("""
    <div class='page-header'>
        <h1>Live Prediction</h1>
        <p>VADER sentiment scoring — real-time</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='predict-box'>", unsafe_allow_html=True)
    user_input = st.text_area("", height=160, placeholder="Paste or type a customer review...")

    if st.button("Run Analysis"):
        if not user_input.strip():
            st.warning("Enter a review first.")
        else:
            analyzer = SentimentIntensityAnalyzer()
            score = analyzer.polarity_scores(user_input)['compound']

            if score >= 0.05:
                sentiment = "Positive"
                bar_color = "#0077b6"
            elif score <= -0.05:
                sentiment = "Negative"
                bar_color = "#03045e"
            else:
                sentiment = "Neutral"
                bar_color = "#00b4d8"

            st.markdown(f"""
            <div class='result-strip' style='border-left-color:{bar_color}'>
                <div class='result-label'>Detected Sentiment</div>
                <div class='result-value'>{sentiment}</div>
                <div class='result-score'>VADER compound score: {score:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)