import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "SIMPLE Q&A Chatbot with Gemini"
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gemini Q&A",
    page_icon="✦",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f !important;
    color: #e8e6f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 60% at 50% -10%, rgba(99,60,180,0.18) 0%, transparent 70%),
        radial-gradient(ellipse 60% 40% at 90% 80%, rgba(30,180,140,0.08) 0%, transparent 60%),
        #0a0a0f !important;
}

/* hide default header */
[data-testid="stHeader"] { display: none !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #10101a !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] * { font-family: 'DM Sans', sans-serif !important; }

.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 0 24px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 24px;
}
.sidebar-brand .gem { font-size: 22px; }
.sidebar-brand .brand-text {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700;
    font-size: 15px;
    color: #c4b5fd;
    letter-spacing: 0.03em;
}
.sidebar-brand .brand-sub {
    font-size: 11px;
    color: rgba(255,255,255,0.35);
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.sidebar-section-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.3);
    margin-bottom: 8px;
}

/* selectbox */
[data-testid="stSelectbox"] > div > div {
    background: #1a1a28 !important;
    border: 1px solid rgba(196,181,253,0.2) !important;
    border-radius: 10px !important;
    color: #e8e6f0 !important;
    font-size: 13px !important;
}
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: rgba(196,181,253,0.5) !important;
    box-shadow: 0 0 0 3px rgba(196,181,253,0.08) !important;
}

/* slider */
[data-testid="stSlider"] > div > div > div {
    background: rgba(196,181,253,0.15) !important;
}
[data-testid="stSlider"] [data-testid="stSliderThumb"] {
    background: #c4b5fd !important;
    border: none !important;
    box-shadow: 0 0 12px rgba(196,181,253,0.4) !important;
}

.sidebar-info {
    margin-top: 32px;
    padding: 14px;
    background: rgba(196,181,253,0.05);
    border: 1px solid rgba(196,181,253,0.1);
    border-radius: 10px;
    font-size: 12px;
    color: rgba(255,255,255,0.4);
    line-height: 1.7;
}
.sidebar-info strong { color: rgba(196,181,253,0.7); }

/* ── Main hero ── */
.hero {
    text-align: center;
    padding: 48px 0 36px;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(196,181,253,0.08);
    border: 1px solid rgba(196,181,253,0.2);
    border-radius: 100px;
    padding: 5px 14px;
    font-size: 11px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #c4b5fd;
    margin-bottom: 20px;
}
.hero-title {
    font-family: 'Syne', sans-serif !important;
    font-size: clamp(32px, 5vw, 52px) !important;
    font-weight: 800 !important;
    line-height: 1.1 !important;
    color: #ffffff !important;
    letter-spacing: -0.02em !important;
    margin-bottom: 14px !important;
}
.hero-title span {
    background: linear-gradient(135deg, #c4b5fd 0%, #7dd3fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 15px;
    color: rgba(255,255,255,0.4);
    letter-spacing: 0.01em;
    max-width: 420px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Input area ── */
.input-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.35);
    margin-bottom: 8px;
}

[data-testid="stTextInput"] > div > div > input {
    background: #13131f !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 14px !important;
    color: #e8e6f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    padding: 16px 20px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stTextInput"] > div > div > input:focus {
    border-color: rgba(196,181,253,0.45) !important;
    box-shadow: 0 0 0 4px rgba(196,181,253,0.07) !important;
    outline: none !important;
}
[data-testid="stTextInput"] > div > div > input::placeholder {
    color: rgba(255,255,255,0.2) !important;
}

/* ── Response card ── */
.response-card {
    background: linear-gradient(135deg, #13131f 0%, #161624 100%);
    border: 1px solid rgba(196,181,253,0.15);
    border-radius: 16px;
    padding: 24px 28px;
    margin-top: 20px;
    position: relative;
    overflow: hidden;
}
.response-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #c4b5fd, #7dd3fc, #6ee7b7);
    border-radius: 16px 16px 0 0;
}
.response-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 16px;
}
.response-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #6ee7b7;
    box-shadow: 0 0 8px rgba(110,231,183,0.6);
    flex-shrink: 0;
}
.response-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.35);
}
.response-text {
    font-size: 15px;
    line-height: 1.75;
    color: #ddd8f0;
    white-space: pre-wrap;
}

/* ── Error / info cards ── */
.error-card {
    background: rgba(239,68,68,0.06);
    border: 1px solid rgba(239,68,68,0.2);
    border-radius: 12px;
    padding: 16px 20px;
    margin-top: 16px;
    color: #fca5a5;
    font-size: 13.5px;
    line-height: 1.6;
}
.idle-card {
    background: rgba(196,181,253,0.04);
    border: 1px solid rgba(196,181,253,0.1);
    border-radius: 12px;
    padding: 16px 20px;
    margin-top: 16px;
    color: rgba(255,255,255,0.3);
    font-size: 13px;
    text-align: center;
    letter-spacing: 0.02em;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #c4b5fd !important; }

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 40px 0 20px;
    font-size: 12px;
    color: rgba(255,255,255,0.2);
    letter-spacing: 0.04em;
}
.footer span { color: rgba(196,181,253,0.4); }

/* hide streamlit default elements */
#MainMenu, footer, [data-testid="stDecoration"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── Prompt template (unchanged) ───────────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Answer user questions clearly and accurately."),
    ("user", "Question: {question}"),
])

# ── Core function (unchanged) ─────────────────────────────────────────────────
def generate_response(question, llm_model, temperature):
    llm = ChatGoogleGenerativeAI(model=llm_model, temperature=temperature)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    return chain.invoke({"question": question})

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="gem">✦</div>
        <div>
            <div class="brand-text">Gemini Q&A</div>
            <div class="brand-sub">Powered by LangChain</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-label">Model</div>', unsafe_allow_html=True)
    llm_model = st.selectbox(
        label="model",
        options=[
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-2.5-pro-preview-05-06",
        ],
        format_func=lambda x: {
            "gemini-2.5-flash":            "✦ Gemini 2.5 Flash",
            "gemini-2.0-flash":            "⚡ Gemini 2.0 Flash",
            "gemini-2.0-flash-lite":       "🪶 Gemini 2.0 Flash Lite",
            "gemini-2.5-pro-preview-05-06":"🧠 Gemini 2.5 Pro",
        }.get(x, x),
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-section-label" style="margin-top:20px;">Temperature</div>', unsafe_allow_html=True)
    temperature = st.slider(
        label="temp",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        label_visibility="collapsed",
    )
    temp_desc = (
        "🧊 Precise & deterministic" if temperature < 0.3 else
        "⚖️ Balanced responses"     if temperature < 0.7 else
        "🔥 Creative & varied"
    )
    st.markdown(f'<div style="font-size:11px;color:rgba(255,255,255,0.3);margin-top:4px;">{temp_desc}</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-info">
        <strong>Tips</strong><br>
        Ask anything — code, concepts, analysis, writing.<br><br>
        Lower temperature → more focused answers.<br>
        Higher temperature → more creative output.
    </div>
    """, unsafe_allow_html=True)

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">✦ AI Assistant</div>
    <div class="hero-title">Ask anything,<br><span>get clarity</span></div>
    <div class="hero-sub">Powered by Google Gemini & LangChain. Ask a question and get a precise, thoughtful answer.</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="input-label">Your question</div>', unsafe_allow_html=True)
user_input = st.text_input(
    label="question",
    placeholder="e.g. Explain transformers in simple terms...",
    label_visibility="collapsed",
)

if user_input:
    try:
        with st.spinner("Thinking..."):
            response = generate_response(user_input, llm_model, temperature)
        st.markdown(f"""
        <div class="response-card">
            <div class="response-header">
                <div class="response-dot"></div>
                <div class="response-label">Response · {llm_model}</div>
            </div>
            <div class="response-text">{response}</div>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f'<div class="error-card">⚠️ {e}</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="idle-card">✦ &nbsp; Type a question above to get started</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">Built with <span>Streamlit · LangChain · Google Gemini</span></div>', unsafe_allow_html=True)