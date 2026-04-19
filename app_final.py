import os
import sys
import streamlit as st
import pandas as pd
import time
from datetime import datetime

from milestone2.agent import run_agent

st.set_page_config(
    page_title="News Credibility Checker",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)
# session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'stats' not in st.session_state:
    st.session_state.stats = {"Real": 0, "Fake": 0, "Uncertain": 0, "Total": 0}


st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Outfit:wght@600;700&display=swap');
    
    :root {
        --bg-color: #060C1A;
        --card-bg: #101C33;
        --card-border: #1E3A5F;
        --text-primary: #F0F6FC;
        --text-secondary: #8B949E;
        --accent-blue: #38BDF8;
        --verdict-credible: #3FB950;
        --verdict-fake: #F85149;
        --verdict-uncertain: #D29922;
    }

    .stApp, [data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background-color: var(--bg-color) !important;
        color: var(--text-primary) !important;
    }

    [data-testid="stSidebar"] {
        background-color: var(--card-bg) !important;
        border-right: 1px solid var(--card-border);
    }

    h1, h2, h3, .verdict-header {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700;
        color: var(--text-primary);
    }

    p, span, div, .stMarkdown {
        font-family: 'Inter', sans-serif;
    }

    .dashboard-card {
        background-color: var(--card-bg);
        border-radius: 16px;
        padding: 28px;
        border: 1px solid var(--card-border);
        margin-bottom: 24px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    }

    .verdict-header {
        font-size: 36px;
        margin-bottom: 24px;
        padding-bottom: 16px;
        border-bottom: 2px solid var(--card-border);
        text-shadow: 0 0 20px rgba(255,255,255,0.05);
    }

    .verdict-CREDIBLE { color: var(--verdict-credible) !important; filter: drop-shadow(0 0 8px rgba(63, 185, 80, 0.3)); }
    .verdict-FAKE { color: var(--verdict-fake) !important; filter: drop-shadow(0 0 8px rgba(248, 81, 73, 0.3)); }
    .verdict-UNCERTAIN { color: var(--verdict-uncertain) !important; filter: drop-shadow(0 0 8px rgba(210, 153, 34, 0.3)); }

    /* Metric Overrides */
    [data-testid="stMetricValue"] {
        font-weight: 700 !important;
        color: var(--text-primary) !important;
    }
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
    }
    
    /* Sidebar Labels Styling */
    [data-testid="stSidebar"] label {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
    }

    /* Source Cards */
    .source-card {
        background-color: #0D1117;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        border: 1px solid var(--card-border);
        transition: all 0.3s ease;
    }

    .source-card:hover {
        border-color: var(--accent-blue);
        transform: translateX(4px);
        background-color: #161B22;
    }

    .source-title {
        color: var(--accent-blue) !important;
        font-weight: 600;
        font-size: 16px;
        text-decoration: none;
    }

    .source-snippet {
        color: var(--text-secondary);
        font-size: 14px;
        margin-top: 8px;
        line-height: 1.6;
    }
    
    /* Input Area */
    .stTextArea textarea, .stTextInput input {
        background-color: #060C1A !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 12px !important;
    }
    .stTextArea textarea::placeholder, .stTextInput input::placeholder {
        color: #FFFFFF !important;
        opacity: 0.7 !important;
    }

    /* Status Widget Dark Mode */
    [data-testid="stStatusWidget"] {
        background-color: #101C33 !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 12px !important;
    }
    [data-testid="stStatusWidget"] div {
        color: var(--text-primary) !important;
    }
    .stStatusWidget {
        background-color: #161B22 !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--card-border) !important;
    }
    /* All Buttons Styling (including Download) */
    div.stButton > button, div.stDownloadButton > button {
        background-color: #101C33 !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 10px !important;
        padding: 10px 20px !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover {
        background-color: #1E3A5F !important;
        border-color: var(--accent-blue) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
    }

    /* Primary Button override */
    div.stButton > button[kind="primary"] {
        background-color: #1E3A5F !important;
        border: 1px solid var(--accent-blue) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #2D5A8F !important;
        box-shadow: 0 4px 15px rgba(56, 189, 248, 0.2) !important;
    }
    </style>
""", unsafe_allow_html=True)



# sidebar 

with st.sidebar:
    st.title("News Credibility Checker")
    st.markdown("Milestone 2 - Agentic AI System")
    
    st.divider()
    if os.getenv("GROQ_API_KEY"):
        st.success("Groq API key loaded")
    else:
        groq_api_key = st.text_input("Groq API Key", type="password", help="Get a free key at console.groq.com")
        if groq_api_key:
            os.environ["GROQ_API_KEY"] = groq_api_key
    
    st.divider()
    st.markdown("### Session Stats")
    st.metric("Total Analyzed", st.session_state.stats["Total"])
    
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Credible", st.session_state.stats["Real"])
        st.metric("Fake", st.session_state.stats["Fake"])
    with c2:
        st.metric("Uncertain", st.session_state.stats["Uncertain"])
    
    st.divider()
    if st.button("Reset Session History"):
        st.session_state.stats = {"Real": 0, "Fake": 0, "Uncertain": 0, "Total": 0}
        st.session_state.history = []
        st.rerun()


st.title("News Credibility Analysis")

st.markdown("Analyze news articles using machine learning and AI-assisted fact checking.")

#  Analysis Trends(charts)
if st.session_state.history:
    with st.expander("View Analysis Trends", expanded=False):
        trend_data = pd.DataFrame(st.session_state.history)
        trend_data['time'] = pd.to_datetime(trend_data['timestamp'])
        chart_data = trend_data.set_index('time').resample('min').count()['verdict'].cumsum()
        st.line_chart(chart_data)

with st.container():
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    article_text = st.text_area(
        "",
        placeholder="Paste article body or social media post content here...",
        height=200,
        label_visibility="collapsed"
    )
    
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        launch_btn = st.button("Run Analysis", type="primary", use_container_width=True)
    with btn_col2:
        if st.button("Clear", use_container_width=True):
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    

if launch_btn:
    if not article_text.strip() or len(article_text.strip()) < 30:
        st.error("Input content must be at least 30 characters.")
    elif not os.getenv("GROQ_API_KEY"):
        st.warning("Groq API Key required for deep reasoning. Enter it in the sidebar.")
    else:
        try:
            with st.status("Running analysis...", expanded=True) as status:
                st.write("Step 1: Running SVM classifier...")
                st.write("Step 2: Retrieving fact-check sources...")
                st.write("Step 3: Generating AI reasoning report...")
                
                result = run_agent(article_text)
                
                if result.get("error"):
                    status.update(label="Analysis Failed", state="error")
                    st.error(f"Reason: {result['error']}")
                else:
                    status.update(label="Analysis Complete", state="complete")
                    
              
                    analysis = result["analysis"]
                    verdict = analysis.get("confidence_assessment", {}).get("overall_verdict", "UNCERTAIN")
                    
                    st.session_state.stats["Total"] += 1
                    if "CREDIBLE" in verdict.upper(): st.session_state.stats["Real"] += 1
                    elif "FAKE" in verdict.upper(): st.session_state.stats["Fake"] += 1
                    else: st.session_state.stats["Uncertain"] += 1
                    
                    st.session_state.history.append({
                        "timestamp": datetime.now().isoformat(),
                        "verdict": verdict,
                        "ml_score": result["prediction"].get("confidence", 0)
                    })
            
            # --- Results Dashboard ---
            st.markdown("---")
            st.markdown(f'<div class="verdict-header verdict-{verdict}">Consolidated Verdict: {verdict}</div>', unsafe_allow_html=True)
            
           
            cols = st.columns(3)
            with cols[0]:
                st.markdown("#### ML Classifier")
                pred = result["prediction"]
                st.metric("SVM Prediction", pred.get("label"))
                st.metric("Base Confidence", f"{pred.get('confidence')}%")
                st.progress(pred.get("confidence", 0) / 100)
            
            with cols[1]:
                st.markdown("#### Retrieved Sources")
                retrieval = result["retrieval"]
                st.metric("Retrieved Sources", len(retrieval.get("sources", [])))
                st.metric("Source Consensus", "HIGH" if len(retrieval.get("sources", [])) > 2 else "LOW")
                with st.expander("View Sources"):
                    for s in retrieval.get("sources", []):
                        st.markdown(f"""
                            <div class="source-card">
                                <a href="{s.get('url', '#')}" target="_blank" class="source-title">{s.get('title')}</a>
                                <p class="source-snippet">{s.get('snippet')}</p>
                            </div>
                        """, unsafe_allow_html=True)

            with cols[2]:
                st.markdown("#### Agentic Reasoning")
                ca = analysis.get("confidence_assessment", {})
                st.metric("Final Confidence", ca.get("confidence_level"))
                st.write(f"**Final Rationale:** {ca.get('reasoning')}")
                if result.get("pdf_report"):
                    st.download_button(
                        label="Download Credibility Report",
                        data=result["pdf_report"],
                        file_name=f"credibility_report_{int(time.time())}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

            # Detailed Breakdown
            st.divider()
            d_col1, d_col2 = st.columns([2, 1])
            
            with d_col1:
                st.subheader("Article Summary")
                st.info(analysis.get("article_summary"))
                
                st.subheader("Deep Verification Findings")
                st.write(analysis.get("cross_source_verification"))
                
                st.caption(f"**Ethical Disclaimer:** {analysis.get('ethical_disclaimer')}")

            with d_col2:
                st.subheader("Risk & Credibility")
                
                st.markdown("**Risk Factors**")
                for r in analysis.get("risk_factors", []):
                    st.warning(f"{r}")
                
                st.markdown("**Positive Indicators**")
                for p in analysis.get("credibility_indicators", {}).get("positive", []):
                    st.success(f"{p}")
                
                warn = analysis.get("misinformation_warning")
                if warn:
                    st.error(f"{warn}")

        except Exception as e:
            st.error(f"Critical System Error: {str(e)}")
            st.exception(e)

else:
    if not st.session_state.history:
        st.info("Paste a news article or social media post above and click Run Analysis to get a credibility assessment.")
        st.markdown("""
        **How it works:**
        1. **Classification** - A trained SVM model predicts whether the article is real or fake.
        2. **Source Retrieval** - The system searches fact-checking sites like Snopes, AP, and PolitiFact.
        3. **AI Reasoning** - A Groq LLaMA 3 model reviews the ML result and retrieved sources to produce a structured report.
        """)
