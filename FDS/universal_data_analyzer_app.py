"""
╔══════════════════════════════════════════════════════════════════╗
║ SmartStream: An End-to-End Interactive Data Science Playground   ║
║         Project: SmartStream using Python                        ║
║         Tools  : Streamlit, Pandas, NumPy, Matplotlib,           ║
║                  Seaborn, Scikit-learn                           ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import sys
import subprocess
warnings.filterwarnings("ignore")

from streamlit.runtime.scriptrunner import get_script_run_ctx

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
import io

if __name__ == "__main__" and get_script_run_ctx() is None:
    print("Starting with Streamlit...", flush=True)
    cmd = [sys.executable, "-m", "streamlit", "run", __file__]
    subprocess.run(cmd, check=False)
    sys.exit(0)

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartStream: An End-to-End Interactive Data Science Playground",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# ENHANCED CUSTOM CSS — Professional White Theme
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Font imports ── */
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── CSS Variables ── */
:root {
    --white: #ffffff;
    --off-white: #fafbfc;
    --surface: #f4f6f9;
    --surface-2: #eef1f6;
    --border: #e2e8f0;
    --border-light: #f0f4fa;
    --ink: #0a0f1e;
    --ink-2: #1e293b;
    --ink-3: #475569;
    --ink-muted: #94a3b8;
    --blue: #2563eb;
    --blue-light: #3b82f6;
    --blue-pale: #eff6ff;
    --blue-glow: rgba(37, 99, 235, 0.12);
    --green: #059669;
    --green-pale: #ecfdf5;
    --amber: #d97706;
    --amber-pale: #fffbeb;
    --red: #dc2626;
    --red-pale: #fef2f2;
    --glass: rgba(255,255,255,0.82);
    --glass-border: rgba(255,255,255,0.9);
    --shadow-sm: 0 1px 3px rgba(10,15,30,0.06), 0 1px 2px rgba(10,15,30,0.04);
    --shadow-md: 0 4px 16px rgba(10,15,30,0.08), 0 2px 6px rgba(10,15,30,0.05);
    --shadow-lg: 0 12px 40px rgba(10,15,30,0.10), 0 4px 16px rgba(10,15,30,0.06);
    --shadow-xl: 0 24px 64px rgba(10,15,30,0.12), 0 8px 24px rgba(10,15,30,0.08);
    --shadow-glow: 0 0 0 3px rgba(37,99,235,0.15), 0 4px 20px rgba(37,99,235,0.12);
    --radius: 14px;
    --radius-sm: 8px;
    --radius-lg: 20px;
    --radius-xl: 28px;
    --transition: all 0.28s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
    background: var(--off-white);
    color: var(--ink) !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, .stDeployButton { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

.block-container {
    padding-top: 0 !important;
    padding-bottom: 3rem;
    max-width: 1280px;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: var(--ink-muted); }

/* ══════════════════════════════════════════
   PARALLAX HERO HEADER
═══════════════════════════════════════════ */
.parallax-hero {
    position: relative;
    background: var(--white);
    border-radius: 0 0 var(--radius-xl) var(--radius-xl);
    padding: 3rem 3rem 2.5rem;
    margin: -1rem -1rem 2rem -1rem;
    overflow: hidden;
    border-bottom: 1px solid var(--border);
    box-shadow: var(--shadow-md);
}

/* Layered parallax grid background */
.parallax-hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(var(--border-light) 1px, transparent 1px),
        linear-gradient(90deg, var(--border-light) 1px, transparent 1px);
    background-size: 40px 40px;
    animation: gridDrift 20s linear infinite;
    opacity: 0.6;
    z-index: 0;
}

/* Glossy orb layer 1 */
.parallax-hero::after {
    content: '';
    position: absolute;
    top: -80px;
    right: -60px;
    width: 420px;
    height: 420px;
    background: radial-gradient(circle at 40% 40%,
        rgba(37,99,235,0.10) 0%,
        rgba(99,102,241,0.06) 40%,
        transparent 70%);
    border-radius: 50%;
    animation: orbFloat 8s ease-in-out infinite;
    z-index: 0;
}

.hero-orb-2 {
    position: absolute;
    bottom: -100px;
    left: -80px;
    width: 360px;
    height: 360px;
    background: radial-gradient(circle at 60% 60%,
        rgba(16,185,129,0.07) 0%,
        rgba(6,182,212,0.04) 40%,
        transparent 70%);
    border-radius: 50%;
    animation: orbFloat 11s ease-in-out infinite reverse;
    z-index: 0;
}

.hero-orb-3 {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 500px;
    height: 200px;
    background: radial-gradient(ellipse,
        rgba(37,99,235,0.04) 0%,
        transparent 70%);
    animation: pulse 6s ease-in-out infinite;
    z-index: 0;
}

@keyframes gridDrift {
    from { transform: translate(0, 0); }
    to   { transform: translate(40px, 40px); }
}

@keyframes orbFloat {
    0%, 100% { transform: translateY(0) scale(1); }
    50%       { transform: translateY(-20px) scale(1.04); }
}

@keyframes pulse {
    0%, 100% { opacity: 0.5; transform: translate(-50%, -50%) scale(1); }
    50%       { opacity: 1; transform: translate(-50%, -50%) scale(1.08); }
}

.hero-content {
    position: relative;
    z-index: 1;
}

.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--blue-pale);
    border: 1px solid rgba(37,99,235,0.2);
    border-radius: 99px;
    padding: 4px 14px;
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--blue);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    animation: slideDown 0.6s cubic-bezier(0.4,0,0.2,1) both;
}

.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    color: var(--ink);
    letter-spacing: -0.03em;
    line-height: 1.15;
    margin: 0 0 0.6rem 0;
    animation: slideDown 0.6s 0.08s cubic-bezier(0.4,0,0.2,1) both;
}

.hero-title span {
    background: linear-gradient(135deg, var(--blue) 0%, #6366f1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-subtitle {
    font-size: 0.95rem;
    color: var(--ink-3);
    margin: 0 0 1.5rem 0;
    font-family: 'JetBrains Mono', monospace;
    animation: slideDown 0.6s 0.14s cubic-bezier(0.4,0,0.2,1) both;
}

.badge-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    animation: slideDown 0.6s 0.2s cubic-bezier(0.4,0,0.2,1) both;
}

.badge {
    background: var(--glass);
    color: var(--ink-2);
    border: 1px solid var(--border);
    border-radius: 99px;
    padding: 5px 14px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    font-family: 'JetBrains Mono', monospace;
    backdrop-filter: blur(8px);
    box-shadow: var(--shadow-sm),
                inset 0 1px 0 rgba(255,255,255,0.9);
    transition: var(--transition);
}

.badge:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-md),
                inset 0 1px 0 rgba(255,255,255,0.9);
    border-color: rgba(37,99,235,0.3);
    color: var(--blue);
}

@keyframes slideDown {
    from { opacity: 0; transform: translateY(-12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ══════════════════════════════════════════
   SECTION HEADERS
═══════════════════════════════════════════ */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0;
    margin: 2rem 0 1.25rem 0;
    animation: fadeIn 0.4s ease both;
}

.section-icon {
    width: 42px;
    height: 42px;
    background: var(--blue-pale);
    border: 1px solid rgba(37,99,235,0.2);
    border-radius: var(--radius-sm);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    box-shadow: var(--shadow-sm),
                inset 0 1px 0 rgba(255,255,255,0.8);
    flex-shrink: 0;
}

.section-header h2 {
    color: var(--ink) !important;
    font-size: 1.3rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.02em;
}

.section-divider {
    height: 1px;
    background: linear-gradient(90deg, var(--border) 0%, transparent 100%);
    margin: 0 0 1.25rem 0;
}

/* ══════════════════════════════════════════
   GLASS CARDS
═══════════════════════════════════════════ */
.glass-card {
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius);
    box-shadow: var(--shadow-md),
                inset 0 1px 0 rgba(255,255,255,0.95);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    transition: var(--transition);
    overflow: hidden;
}

.glass-card:hover {
    box-shadow: var(--shadow-lg),
                inset 0 1px 0 rgba(255,255,255,0.95);
    transform: translateY(-2px);
}

/* ══════════════════════════════════════════
   METRIC CARDS — Glossy
═══════════════════════════════════════════ */
.metric-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 12px;
    margin-bottom: 1.25rem;
}

.metric-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.1rem 1rem;
    text-align: center;
    box-shadow: var(--shadow-sm),
                inset 0 1px 0 rgba(255,255,255,1);
    position: relative;
    overflow: hidden;
    transition: var(--transition);
    animation: fadeInUp 0.5s ease both;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--blue), #6366f1);
    opacity: 0;
    transition: var(--transition);
}

.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: var(--shadow-lg),
                inset 0 1px 0 rgba(255,255,255,1);
    border-color: rgba(37,99,235,0.2);
}

.metric-card:hover::before {
    opacity: 1;
}

.metric-card .val {
    font-size: 1.65rem;
    font-weight: 800;
    color: var(--ink);
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -0.02em;
    line-height: 1.1;
}

.metric-card .lbl {
    font-size: 0.7rem;
    color: var(--ink-muted);
    margin-top: 4px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ══════════════════════════════════════════
   STEP CARDS
═══════════════════════════════════════════ */
.step-card {
    background: var(--blue-pale);
    border: 1px solid rgba(37,99,235,0.15);
    border-radius: var(--radius);
    padding: 1.1rem 1.4rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    animation: fadeIn 0.4s ease both;
}

.step-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: linear-gradient(180deg, var(--blue), #6366f1);
    border-radius: 0 2px 2px 0;
}

.step-card h4 {
    color: var(--ink) !important;
    font-size: 0.9rem;
    font-weight: 700;
    margin: 0 0 0.35rem 0;
    letter-spacing: -0.01em;
}

.step-card p {
    color: var(--ink-3) !important;
    font-size: 0.84rem;
    margin: 0;
    line-height: 1.7;
}

/* ══════════════════════════════════════════
   INSIGHT BOXES
═══════════════════════════════════════════ */
.insight-box {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.1rem 1.4rem;
    margin-bottom: 0.75rem;
    box-shadow: var(--shadow-sm),
                inset 0 1px 0 rgba(255,255,255,0.9);
    transition: var(--transition);
    animation: fadeInUp 0.4s ease both;
}

.insight-box:hover {
    box-shadow: var(--shadow-md),
                inset 0 1px 0 rgba(255,255,255,0.9);
    border-color: rgba(37,99,235,0.2);
}

.insight-box .ins-title {
    font-weight: 700;
    color: var(--ink) !important;
    font-size: 0.88rem;
    margin-bottom: 0.35rem;
    letter-spacing: -0.01em;
}

.insight-box .ins-body {
    color: var(--ink-3) !important;
    font-size: 0.84rem;
    line-height: 1.7;
}

/* ══════════════════════════════════════════
   VIVA CARDS
═══════════════════════════════════════════ */
.viva-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.1rem 1.4rem;
    margin-bottom: 0.75rem;
    box-shadow: var(--shadow-sm),
                inset 0 1px 0 rgba(255,255,255,0.9);
    position: relative;
    overflow: hidden;
    transition: var(--transition);
    animation: fadeInUp 0.4s ease both;
}

.viva-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: linear-gradient(180deg, #f59e0b, #ef4444);
}

.viva-card:hover {
    box-shadow: var(--shadow-md),
                inset 0 1px 0 rgba(255,255,255,0.9);
    transform: translateX(3px);
}

.viva-card .viva-q {
    font-weight: 700;
    color: var(--blue) !important;
    font-size: 0.88rem;
    margin-bottom: 0.4rem;
}

.viva-card .viva-a {
    color: var(--ink-3) !important;
    font-size: 0.84rem;
    line-height: 1.7;
}

/* ══════════════════════════════════════════
   CODE BOX
═══════════════════════════════════════════ */
.code-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 1rem 1.25rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--ink-2);
    margin: 0.5rem 0;
    white-space: pre;
    overflow-x: auto;
    box-shadow: inset 0 2px 8px rgba(10,15,30,0.04);
}

/* ══════════════════════════════════════════
   DATASET CARDS
═══════════════════════════════════════════ */
.dataset-card {
    background: var(--white);
    border: 1.5px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: var(--transition);
    box-shadow: var(--shadow-sm),
                inset 0 1px 0 rgba(255,255,255,1);
    animation: fadeInUp 0.5s ease both;
}

.dataset-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-xl),
                inset 0 1px 0 rgba(255,255,255,1);
    border-color: rgba(37,99,235,0.3);
}

.dataset-card::after {
    content: '';
    position: absolute;
    top: -60px; right: -40px;
    width: 120px; height: 120px;
    background: radial-gradient(circle,
        rgba(37,99,235,0.08) 0%, transparent 70%);
    border-radius: 50%;
    transition: var(--transition);
}

.dataset-card:hover::after {
    transform: scale(1.5);
    opacity: 0.6;
}

/* ══════════════════════════════════════════
   TABS
═══════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    background: var(--surface);
    border-radius: var(--radius-sm);
    padding: 4px;
    border: 1px solid var(--border);
    box-shadow: inset 0 2px 4px rgba(10,15,30,0.04);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 6px;
    padding: 7px 18px;
    font-weight: 500;
    font-size: 0.83rem;
    color: var(--ink-3) !important;
    background: transparent;
    border: none !important;
    transition: var(--transition);
}

.stTabs [data-baseweb="tab"]:hover {
    background: var(--white) !important;
    color: var(--ink) !important;
}

.stTabs [aria-selected="true"] {
    background: var(--white) !important;
    color: var(--blue) !important;
    box-shadow: var(--shadow-sm) !important;
    font-weight: 600 !important;
}

/* ══════════════════════════════════════════
   SIDEBAR
═══════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: var(--white);
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] > div {
    background: var(--white);
}

section[data-testid="stSidebar"] * {
    color: var(--ink) !important;
}

.sidebar-brand {
    padding: 1.25rem 0 1rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.25rem;
}

.sidebar-brand .brand-icon {
    width: 40px; height: 40px;
    background: linear-gradient(135deg, var(--blue), #6366f1);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 4px 12px rgba(37,99,235,0.25);
}

.sidebar-brand h3 {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--ink) !important;
    margin: 0;
    letter-spacing: -0.01em;
}

.sidebar-brand p {
    font-size: 0.72rem;
    color: var(--ink-muted) !important;
    margin: 2px 0 0 0;
    font-family: 'JetBrains Mono', monospace;
}

/* ══════════════════════════════════════════
   FORM ELEMENTS
═══════════════════════════════════════════ */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: var(--white) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--ink) !important;
    box-shadow: var(--shadow-sm) !important;
    transition: var(--transition) !important;
    font-family: 'Sora', sans-serif !important;
}

.stSelectbox > div > div:focus-within,
.stMultiSelect > div > div:focus-within {
    border-color: var(--blue) !important;
    box-shadow: var(--shadow-glow) !important;
}

.stTextInput input, .stNumberInput input {
    background: var(--white) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--ink) !important;
    font-family: 'Sora', sans-serif !important;
    transition: var(--transition) !important;
    box-shadow: var(--shadow-sm) !important;
}

.stTextInput input:focus, .stNumberInput input:focus {
    border-color: var(--blue) !important;
    box-shadow: var(--shadow-glow) !important;
    outline: none !important;
}

/* File uploader */
.stFileUploader > div {
    background: var(--white) !important;
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    transition: var(--transition) !important;
}

.stFileUploader > div:hover {
    border-color: var(--blue) !important;
    background: var(--blue-pale) !important;
    box-shadow: var(--shadow-glow) !important;
}

/* ══════════════════════════════════════════
   BUTTONS — Glossy
═══════════════════════════════════════════ */
.stButton > button {
    background: linear-gradient(160deg, #2563eb 0%, #1d4ed8 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.55rem 1.4rem !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.3),
                inset 0 1px 0 rgba(255,255,255,0.2) !important;
    transition: var(--transition) !important;
    letter-spacing: 0.01em !important;
    position: relative !important;
    overflow: hidden !important;
}

.stButton > button::before {
    content: '' !important;
    position: absolute !important;
    top: 0 !important; left: 0 !important; right: 0 !important;
    height: 50% !important;
    background: linear-gradient(180deg, rgba(255,255,255,0.15) 0%, transparent 100%) !important;
    pointer-events: none !important;
}

.stButton > button:hover {
    background: linear-gradient(160deg, #1d4ed8 0%, #1e40af 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(37,99,235,0.35),
                inset 0 1px 0 rgba(255,255,255,0.2) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
    box-shadow: 0 2px 6px rgba(37,99,235,0.25) !important;
}

/* Download buttons */
.stDownloadButton > button {
    background: var(--white) !important;
    color: var(--blue) !important;
    border: 1.5px solid rgba(37,99,235,0.3) !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
    box-shadow: var(--shadow-sm),
                inset 0 1px 0 rgba(255,255,255,0.9) !important;
    transition: var(--transition) !important;
}

.stDownloadButton > button:hover {
    background: var(--blue-pale) !important;
    border-color: var(--blue) !important;
    transform: translateY(-2px) !important;
    box-shadow: var(--shadow-md),
                0 0 0 3px rgba(37,99,235,0.1) !important;
}

/* ══════════════════════════════════════════
   RADIO BUTTONS
═══════════════════════════════════════════ */
.stRadio > div {
    gap: 6px !important;
}

.stRadio [data-testid="stMarkdownContainer"] p {
    color: var(--ink) !important;
    font-size: 0.88rem !important;
    font-family: 'Sora', sans-serif !important;
}

/* ══════════════════════════════════════════
   SLIDERS
═══════════════════════════════════════════ */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--blue) !important;
    border: 3px solid white !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.3) !important;
    transition: var(--transition) !important;
}

.stSlider [data-baseweb="slider"] div[role="slider"]:hover {
    transform: scale(1.15) !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.4) !important;
}

/* ══════════════════════════════════════════
   DATAFRAMES
═══════════════════════════════════════════ */
.stDataFrame {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ══════════════════════════════════════════
   ALERTS
═══════════════════════════════════════════ */
div[data-testid="stInfo"] {
    background: var(--blue-pale) !important;
    border: 1px solid rgba(37,99,235,0.2) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--ink-2) !important;
}

div[data-testid="stSuccess"] {
    background: var(--green-pale) !important;
    border: 1px solid rgba(5,150,105,0.2) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--ink-2) !important;
}

div[data-testid="stWarning"] {
    background: var(--amber-pale) !important;
    border: 1px solid rgba(217,119,6,0.2) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--ink-2) !important;
}

div[data-testid="stError"] {
    background: var(--red-pale) !important;
    border: 1px solid rgba(220,38,38,0.2) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--ink-2) !important;
}

/* ══════════════════════════════════════════
   PROGRESS BAR
═══════════════════════════════════════════ */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--blue), #6366f1) !important;
    border-radius: 99px !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.3) !important;
}

/* ══════════════════════════════════════════
   CHARTS & PLOTS
═══════════════════════════════════════════ */
.stPyplot, [data-testid="stImage"] {
    border-radius: var(--radius) !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-md) !important;
    border: 1px solid var(--border) !important;
}

/* ══════════════════════════════════════════
   CLEANING LOG ITEMS
═══════════════════════════════════════════ */
.clean-log-item {
    background: var(--green-pale);
    border: 1px solid rgba(5,150,105,0.2);
    border-radius: var(--radius-sm);
    padding: 8px 14px;
    margin-bottom: 6px;
    font-size: 0.82rem;
    animation: slideIn 0.3s ease both;
}

.clean-log-item b { color: var(--green) !important; }
.clean-log-item span { color: var(--ink-3) !important; }

@keyframes slideIn {
    from { opacity: 0; transform: translateX(-8px); }
    to   { opacity: 1; transform: translateX(0); }
}

/* ══════════════════════════════════════════
   ANIMATIONS
═══════════════════════════════════════════ */
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ══════════════════════════════════════════
   CONCLUSION BLOCK
═══════════════════════════════════════════ */
.conclusion-block {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: var(--radius-xl);
    padding: 2.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-lg),
                inset 0 1px 0 rgba(255,255,255,0.9);
}

.conclusion-block::before {
    content: '';
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(var(--border-light) 1px, transparent 1px),
        linear-gradient(90deg, var(--border-light) 1px, transparent 1px);
    background-size: 30px 30px;
    opacity: 0.5;
}

.conclusion-block > * { position: relative; z-index: 1; }

.conclusion-block h2 {
    font-size: 1.4rem;
    font-weight: 800;
    color: var(--ink);
    margin: 0 0 1rem;
    letter-spacing: -0.02em;
}

.conclusion-block p {
    color: var(--ink-3);
    font-size: 0.95rem;
    line-height: 1.8;
    max-width: 600px;
    margin: 0 auto;
}

/* ══════════════════════════════════════════
   SIDEBAR WORKFLOW ITEMS
═══════════════════════════════════════════ */
.stRadio label {
    transition: var(--transition) !important;
    border-radius: 6px !important;
    padding: 2px 4px !important;
}

.stRadio label:hover {
    background: var(--blue-pale) !important;
    color: var(--blue) !important;
}

/* ══════════════════════════════════════════
   SPINNER
═══════════════════════════════════════════ */
.stSpinner > div {
    border-top-color: var(--blue) !important;
}

/* Smooth page transitions */
.main .block-container {
    animation: fadeIn 0.4s ease both;
}

</style>
""", unsafe_allow_html=True)

# Parallax JS for depth effect
st.markdown("""
<script>
document.addEventListener('mousemove', function(e) {
    const hero = document.querySelector('.parallax-hero');
    if (!hero) return;
    const rect = hero.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    const dx = (e.clientX - cx) / rect.width;
    const dy = (e.clientY - cy) / rect.height;
    const orbs = hero.querySelectorAll('.hero-orb-2, .hero-orb-3');
    orbs.forEach((orb, i) => {
        const depth = (i + 1) * 8;
        orb.style.transform = `translate(${dx * depth}px, ${dy * depth}px)`;
    });
});
</script>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# HELPER UTILITIES
# ─────────────────────────────────────────────────────────────────
def section(icon, title):
    st.markdown(f"""
    <div class="section-header">
        <div class="section-icon">{icon}</div>
        <h2>{title}</h2>
    </div>
    <div class="section-divider"></div>
    """, unsafe_allow_html=True)

def metric_cards(items):
    cols = st.columns(len(items))
    for idx, (col, (val, lbl)) in enumerate(zip(cols, items)):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="animation-delay: {idx * 0.06}s">
                <div class="val">{val}</div>
                <div class="lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

def insight(icon, title, body):
    st.markdown(f"""
    <div class="insight-box">
        <div class="ins-title">{icon} {title}</div>
        <div class="ins-body">{body}</div>
    </div>""", unsafe_allow_html=True)

def viva(q, a):
    st.markdown(f"""
    <div class="viva-card">
        <div class="viva-q">🎤 Q: {q}</div>
        <div class="viva-a">💬 {a}</div>
    </div>""", unsafe_allow_html=True)

def fig_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130,
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf

def detect_cols(df):
    id_kw  = ['id','no','number','index','sl','serial','code']
    nm_kw  = ['name','id','code','number','email','phone']
    num = [c for c in df.select_dtypes(include=[np.number]).columns
           if not any(k in c.lower() for k in id_kw)]
    cat = [c for c in df.select_dtypes(include='object').columns
           if not any(k in c.lower() for k in nm_kw)]
    return num, cat

# Matplotlib theme
def apply_plot_style(fig, *axes):
    fig.patch.set_facecolor('#ffffff')
    for ax in axes:
        ax.set_facecolor('#fafbfc')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#e2e8f0')
        ax.spines['bottom'].set_color('#e2e8f0')
        ax.tick_params(colors='#475569', labelsize=9)
        ax.xaxis.label.set_color('#475569')
        ax.yaxis.label.set_color('#475569')
        ax.title.set_color('#0a0f1e')
        ax.grid(True, color='#f0f4fa', linewidth=0.8, linestyle='-')
        ax.set_axisbelow(True)

PALETTE = ['#2563eb','#059669','#dc2626','#d97706',
           '#7c3aed','#0891b2','#db2777','#ea580c']


# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="brand-icon">🔬</div>
        <h3>SmartStream</h3>
        <p>Universal · Automated · Smart</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Workflow**")
    workflow_stage = st.radio(
        "",
        [
            "Data Upload",
            "Data Understanding",
            "Data Cleaning",
            "Data Filtering",
            "EDA & Analysis",
            "Visualization",
            "ML Prediction",
            "Result Summary",
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:0.75rem;color:#94a3b8;line-height:1.8;padding:0.5rem 0'>
        <div style='color:#0a0f1e;font-weight:600;margin-bottom:6px;font-size:0.8rem'>Project Info</div>
        <div><span style='color:#94a3b8'>Type</span>&nbsp;&nbsp; End-to-End DS</div>
        <div><span style='color:#94a3b8'>Stack</span>&nbsp; Python · Sklearn</div>
        <div><span style='color:#94a3b8'>Input</span>&nbsp;&nbsp; Any CSV</div>
        <div style='margin-top:10px;padding:8px 10px;background:#f4f6f9;border-radius:8px;border:1px solid #e2e8f0'>
            <span style='color:#2563eb;font-weight:600'>▸ {workflow_stage}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# PARALLAX HERO HEADER
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="parallax-hero">
    <div class="hero-orb-2"></div>
    <div class="hero-orb-3"></div>
    <div class="hero-content">
        <div class="hero-eyebrow">
            <span>⚡</span> End-to-End Data Science Platform
        </div>
        <h1 class="hero-title">SmartStream: <span>An End-to-End Interactive Data Science Playground</span></h1>
        <p class="hero-subtitle">Upload CSV → Auto Clean → Analyze → Visualize → Predict</p>
        <div class="badge-row">
            <span class="badge">Pandas</span>
            <span class="badge">NumPy</span>
            <span class="badge">Matplotlib</span>
            <span class="badge">Seaborn</span>
            <span class="badge">Scikit-learn</span>
            <span class="badge">Streamlit</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# STEP 1 — DATA UPLOAD
# ─────────────────────────────────────────────────────────────────
if workflow_stage == "Data Upload":
    section("📂", "Data Upload")

@st.cache_data(show_spinner=False)
def load_data(file_bytes, file_name):
    return pd.read_csv(io.BytesIO(file_bytes))

@st.cache_data(show_spinner=False)
def sample_student_data():
    np.random.seed(42)
    n = 150
    df = pd.DataFrame({
        'StudentID'  : range(1, n+1),
        'Name'       : [f'Student_{i:03d}' for i in range(1, n+1)],
        'Math'       : np.random.randint(30, 100, n).astype(float),
        'Science'    : np.random.randint(25, 100, n).astype(float),
        'English'    : np.random.randint(20, 100, n).astype(float),
        'History'    : np.random.randint(35, 100, n).astype(float),
        'Attendance' : np.random.uniform(50, 100, n).round(1),
        'Grade'      : np.random.choice(['A','B','C','D','F'], n, p=[0.2,0.3,0.25,0.15,0.1])
    })
    for col in ['Math','Science','English','History']:
        df.loc[np.random.choice(n,10,replace=False), col] = np.nan
    df = pd.concat([df, df.sample(8, random_state=1)], ignore_index=True)
    return df

@st.cache_data(show_spinner=False)
def sample_retail_data():
    np.random.seed(42)
    n = 220
    df = pd.DataFrame({
        'InvoiceID': range(1, n+1),
        'StoreID'  : np.random.choice(['A','B','C','D'], n),
        'Product'  : np.random.choice(['Widget','Gadget','Doohickey','Thingamabob'], n),
        'Quantity' : np.random.randint(1, 12, n),
        'Price'    : np.round(np.random.uniform(5, 120, n), 2),
    })
    df['Revenue'] = df['Quantity'] * df['Price']
    return df

@st.cache_data(show_spinner=False)
def sample_heart_data():
    np.random.seed(42)
    n = 300
    df = pd.DataFrame({
        'Age'         : np.random.randint(29, 78, n),
        'Sex'         : np.random.choice(['M','F'], n),
        'Cholesterol' : np.random.randint(150, 310, n),
        'RestBP'      : np.random.randint(90, 180, n),
        'MaxHR'       : np.random.randint(90, 210, n),
        'Oldpeak'     : np.round(np.random.uniform(0, 6, n), 1),
        'HeartDisease': np.random.choice([0,1], n, p=[0.65,0.35])
    })
    return df

# Dataset selection UI
col_up, col_hint = st.columns([1.6, 1])

dataset_preset_options = [
    "📘 student_marks.csv",
    "📗 retail_store.csv",
    "📕 heart_disease.csv",
    "📙 Custom CSV Upload"
]

with col_up:
    st.markdown("### Choose Dataset")
    dataset_choice = st.selectbox(
        "Select a dataset to analyze:",
        dataset_preset_options,
        index=0,
        help="Built-in samples or upload your own CSV"
    )
    uploaded = st.file_uploader(
        "Upload your CSV (optional)",
        type=["csv"],
        help="Required only for Custom CSV Upload"
    )

with col_hint:
    cards = {
        "📘 student_marks.csv": {
            "emoji": "📘", "title": "Student Marks",
            "color": "#2563eb", "bg": "#eff6ff",
            "desc": "150 students · Math, Science, English, History, Attendance & Grade columns",
            "use": "Academic analysis & grade prediction"
        },
        "📗 retail_store.csv": {
            "emoji": "📗", "title": "Retail Store",
            "color": "#059669", "bg": "#ecfdf5",
            "desc": "220 invoices · Product, Quantity, Price & Revenue across 4 stores",
            "use": "Sales forecasting & store performance"
        },
        "📕 heart_disease.csv": {
            "emoji": "📕", "title": "Heart Disease",
            "color": "#dc2626", "bg": "#fef2f2",
            "desc": "300 patients · Age, Cholesterol, BP, MaxHR & diagnosis",
            "use": "Medical prediction & risk assessment"
        },
        "📙 Custom CSV Upload": {
            "emoji": "📙", "title": "Custom Dataset",
            "color": "#7c3aed", "bg": "#f5f3ff",
            "desc": "Any size · Auto column detection · Flexible analysis",
            "use": "Custom research & specialized datasets"
        },
    }
    info = cards.get(dataset_choice, cards["📘 student_marks.csv"])
    st.markdown(f"""
    <div style="background:{info['bg']};border:1.5px solid {info['color']}22;border-radius:16px;
                padding:1.5rem;position:relative;overflow:hidden;
                box-shadow:0 4px 20px {info['color']}15,inset 0 1px 0 rgba(255,255,255,0.9)">
        <div style="position:absolute;top:-30px;right:-20px;width:80px;height:80px;
                    background:radial-gradient(circle,{info['color']}15 0%,transparent 70%);border-radius:50%"></div>
        <div style="font-size:2rem;margin-bottom:0.75rem">{info['emoji']}</div>
        <div style="font-weight:700;color:{info['color']};font-size:1rem;margin-bottom:0.5rem">{info['title']}</div>
        <div style="color:#475569;font-size:0.82rem;line-height:1.6;margin-bottom:0.75rem">{info['desc']}</div>
        <div style="background:rgba(255,255,255,0.7);border:1px solid {info['color']}20;border-radius:8px;
                    padding:6px 12px;font-size:0.75rem;color:{info['color']};font-weight:600">
            🎯 {info['use']}
        </div>
    </div>
    """, unsafe_allow_html=True)

# Load dataset
if uploaded is not None:
    with st.spinner("Processing your CSV..."):
        prog = st.progress(0)
        for i in range(100):
            prog.progress(i + 1)
        df_raw = load_data(uploaded.read(), uploaded.name)
        prog.empty()
    st.success(f"✅ **{uploaded.name}** loaded — {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
else:
    loader_map = {
        "📘 student_marks.csv": (sample_student_data, "📘"),
        "📗 retail_store.csv":  (sample_retail_data,  "📗"),
        "📕 heart_disease.csv": (sample_heart_data,   "📕"),
    }
    if dataset_choice == "📙 Custom CSV Upload":
        st.warning("Please upload a CSV file to use Custom mode.")
        st.stop()
    elif dataset_choice in loader_map:
        fn, icon = loader_map[dataset_choice]
        with st.spinner(f"Loading {dataset_choice}..."):
            prog = st.progress(0)
            for i in range(100):
                prog.progress(i + 1)
            df_raw = fn()
            prog.empty()
        st.success(f"✅ Loaded **{dataset_choice}** — {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
    else:
        st.warning("Please select a dataset.")
        st.stop()


# ─────────────────────────────────────────────────────────────────
# STEP 2 — DATA UNDERSTANDING
# ─────────────────────────────────────────────────────────────────
null_total     = int(df_raw.isnull().sum().sum())
dup_total      = int(df_raw.duplicated().sum())
num_cols_raw, cat_cols_raw = detect_cols(df_raw)

if workflow_stage == "Data Understanding":
    section("🔍", "Data Understanding")

    metric_cards([
        (f"{df_raw.shape[0]:,}", "Total Rows"),
        (df_raw.shape[1],        "Columns"),
        (null_total,             "Missing Values"),
        (dup_total,              "Duplicate Rows"),
        (len(num_cols_raw),      "Numeric Cols"),
        (len(cat_cols_raw),      "Categorical Cols"),
    ])

    tab_head, tab_info, tab_desc, tab_null = st.tabs(
        ["📋 Preview", "🗂 Info", "📊 Statistics", "⚠️ Missing"])

    with tab_head:
        n_preview = st.slider("Rows to preview", 5, min(50, len(df_raw)), 10)
        st.dataframe(df_raw.head(n_preview), use_container_width=True)

    with tab_info:
        info_df = pd.DataFrame({
            'Column'   : df_raw.columns,
            'Dtype'    : df_raw.dtypes.values,
            'Non-Null' : df_raw.count().values,
            'Null'     : df_raw.isnull().sum().values,
            'Null %'   : (df_raw.isnull().mean()*100).round(2).values,
            'Sample'   : [str(df_raw[c].dropna().iloc[0]) if df_raw[c].notna().any() else 'N/A'
                          for c in df_raw.columns]
        })
        st.dataframe(info_df, use_container_width=True)

    with tab_desc:
        st.dataframe(df_raw.describe().round(3), use_container_width=True)

    with tab_null:
        null_df = df_raw.isnull().sum().reset_index()
        null_df.columns = ['Column', 'Missing Count']
        null_df['Missing %'] = (null_df['Missing Count'] / len(df_raw) * 100).round(2)
        null_df = null_df[null_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        if null_df.empty:
            st.success("✅ No missing values found!")
        else:
            st.dataframe(null_df, use_container_width=True)
            fig, ax = plt.subplots(figsize=(8, 3))
            apply_plot_style(fig, ax)
            ax.barh(null_df['Column'], null_df['Missing %'],
                    color='#dc2626', alpha=0.8, edgecolor='white', height=0.55)
            ax.set_xlabel('Missing %')
            ax.set_title('Missing Values per Column', fontweight='bold', pad=10)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

    insight("🔍", "Data Understanding Complete",
            f"Dataset has <b>{df_raw.shape[0]:,} rows</b> and <b>{df_raw.shape[1]} columns</b>. "
            f"Found <b>{null_total} missing values</b> and <b>{dup_total} duplicates</b> requiring cleanup. "
            f"Detected <b>{len(num_cols_raw)} numeric</b> and <b>{len(cat_cols_raw)} categorical</b> columns.")

    viva("What did you check in Data Understanding?",
         "I checked shape (rows × columns), data types, missing value counts, and duplicate rows. "
         "I also used describe() for statistical summary: mean, std, min, max for each numeric column.")


# ─────────────────────────────────────────────────────────────────
# STEP 3 — DATA CLEANING
# ─────────────────────────────────────────────────────────────────
df = df_raw.copy()
cleaning_log = []

before = len(df)
df.drop_duplicates(inplace=True)
dupes_removed = before - len(df)
cleaning_log.append(("Duplicates Removed", f"{dupes_removed} duplicate rows dropped"))

num_fills = {}
for col in df.select_dtypes(include=[np.number]).columns:
    n_null = df[col].isnull().sum()
    if n_null > 0:
        m = df[col].mean()
        df[col].fillna(m, inplace=True)
        num_fills[col] = (n_null, round(m, 2))
        cleaning_log.append(("Numeric Null Filled", f"[{col}] — {n_null} nulls → mean = {round(m,2)}"))

cat_fills = {}
for col in df.select_dtypes(include='object').columns:
    n_null = df[col].isnull().sum()
    if n_null > 0:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        cat_fills[col] = (n_null, mode_val)
        cleaning_log.append(("Categorical Null Filled", f"[{col}] — {n_null} nulls → mode = '{mode_val}'"))

df.reset_index(drop=True, inplace=True)

if workflow_stage == "Data Cleaning":
    section("🧹", "Data Cleaning")

    st.markdown("""
    <div class="step-card">
        <h4>Why Data Cleaning? → Garbage In = Garbage Out</h4>
        <p>Raw data contains errors. We fix: Missing values (fill with mean/mode) ·
        Duplicate rows (remove) · Invalid entries — ensuring the model learns from quality data.</p>
    </div>""", unsafe_allow_html=True)

    col_log, col_after = st.columns([1.2, 1])
    with col_log:
        st.markdown("**Cleaning Log**")
        for status, detail in cleaning_log:
            st.markdown(f"""
            <div class="clean-log-item">
                <b>✅ {status}</b><br>
                <span>{detail}</span>
            </div>""", unsafe_allow_html=True)

    with col_after:
        st.markdown("**After Cleaning**")
        metric_cards([
            (f"{len(df):,}", "Clean Rows"),
            (int(df.isnull().sum().sum()), "Nulls Left"),
            (int(df.duplicated().sum()), "Dupes Left"),
        ])
        st.dataframe(df.head(8), use_container_width=True)

    st.download_button("⬇️  Download Cleaned Dataset",
                       df.to_csv(index=False).encode(),
                       "cleaned_dataset.csv", "text/csv")

    insight("🧹", "Cleaning Complete",
            f"Removed <b>{dupes_removed} duplicate rows</b>. "
            f"Filled <b>{len(num_fills)} numeric columns</b> with column mean. "
            f"Dataset is now clean with <b>{len(df):,} rows</b> ready for analysis.")

    viva("Why use mean to fill missing values?",
         "Mean represents the central tendency for normally distributed data without distorting "
         "the distribution. For categorical columns, mode (most frequent value) is used instead.")


# ─────────────────────────────────────────────────────────────────
# STEP 4 — DATA FILTERING
# ─────────────────────────────────────────────────────────────────
num_cols, cat_cols = detect_cols(df)
df_filtered = df.copy()

if workflow_stage == "Data Filtering":
    section("🔎", "Data Filtering")

    st.markdown("""
    <div class="step-card">
        <h4>Purpose: Focus on Useful Data Only</h4>
        <p>Filtering extracts rows meeting a condition — e.g. Marks > 60, Revenue > Average.
        This narrows analysis to only the relevant subset of data.</p>
    </div>""", unsafe_allow_html=True)

    if num_cols:
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            filter_col = st.selectbox("Column to filter", num_cols)
        with col_f2:
            filter_op = st.selectbox("Condition", [">", ">=", "<", "<=", "==", "!="])
        with col_f3:
            filter_val = st.number_input("Value",
                value=round(float(df[filter_col].mean()), 2),
                min_value=float(df[filter_col].min()),
                max_value=float(df[filter_col].max()))

        ops = {'>': df[filter_col] > filter_val, '>=': df[filter_col] >= filter_val,
               '<': df[filter_col] < filter_val,  '<=': df[filter_col] <= filter_val,
               '==': df[filter_col] == filter_val, '!=': df[filter_col] != filter_val}
        df_filtered = df[ops[filter_op]].reset_index(drop=True)

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.info(f"Filter: **{filter_col} {filter_op} {filter_val}**")
            metric_cards([
                (f"{len(df):,}", "Original"),
                (f"{len(df_filtered):,}", "Filtered"),
                (f"{len(df)-len(df_filtered):,}", "Removed"),
            ])
        with col_r2:
            pct = len(df_filtered)/len(df)*100
            fig, ax = plt.subplots(figsize=(5, 2.5))
            apply_plot_style(fig, ax)
            ax.barh(['Kept', 'Removed'],
                    [len(df_filtered), len(df)-len(df_filtered)],
                    color=[PALETTE[0], PALETTE[2]],
                    alpha=0.85, edgecolor='white', height=0.5)
            ax.set_xlabel('Rows')
            ax.set_title(f'{pct:.1f}% of rows match', fontweight='bold', pad=8)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        st.dataframe(df_filtered.head(10), use_container_width=True)
        st.download_button("⬇️  Download Filtered Dataset",
                           df_filtered.to_csv(index=False).encode(),
                           "filtered_dataset.csv", "text/csv")
    else:
        st.warning("No numeric columns found for filtering.")

    viva("What is Data Filtering?",
         "Selecting only rows satisfying a specific condition. Example: students with marks > 60. "
         "It focuses analysis on meaningful data subsets.")


# ─────────────────────────────────────────────────────────────────
# STEP 5 — EDA & ANALYSIS
# ─────────────────────────────────────────────────────────────────
if workflow_stage == "EDA & Analysis":
    section("📊", "Exploratory Data Analysis (EDA)")

    st.markdown("""
    <div class="step-card">
        <h4>Purpose: Understand Patterns in Data</h4>
        <p>EDA computes mean, sum, std, and correlations to reveal hidden patterns
        before building any predictive model.</p>
    </div>""", unsafe_allow_html=True)

    if num_cols:
        tab_stat, tab_corr, tab_cat = st.tabs(["📐 Statistics", "🔗 Correlation", "🏷 Category Counts"])

        with tab_stat:
            stats_df = pd.DataFrame({
                'Mean'   : df[num_cols].mean().round(3),
                'Median' : df[num_cols].median().round(3),
                'Std Dev': df[num_cols].std().round(3),
                'Min'    : df[num_cols].min().round(3),
                'Max'    : df[num_cols].max().round(3),
                'Sum'    : df[num_cols].sum().round(2),
            })
            st.dataframe(stats_df, use_container_width=True)
            for col in num_cols[:3]:
                m = df[col].mean(); s = df[col].std()
                cv = s/m*100 if m != 0 else 0
                insight("📐", f"Column: {col}",
                        f"Mean = <b>{m:.2f}</b> | Std = <b>{s:.2f}</b> | CV = <b>{cv:.1f}%</b> — "
                        f"{'Low variance (stable)' if cv < 20 else 'High variance (spread out)'}")

        with tab_corr:
            if len(num_cols) >= 2:
                corr_matrix = df[num_cols].corr()
                st.dataframe(corr_matrix.round(3), use_container_width=True)
                corr_pairs = [(num_cols[i], num_cols[j], round(corr_matrix.iloc[i,j],3))
                              for i in range(len(num_cols)) for j in range(i+1, len(num_cols))]
                corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                st.markdown("**Top Correlated Pairs:**")
                for a, b, r in corr_pairs[:5]:
                    color = "#059669" if abs(r)>0.7 else "#d97706" if abs(r)>0.4 else "#94a3b8"
                    direction = "Strong positive" if r>0.7 else "Strong negative" if r<-0.7 else \
                                "Moderate" if abs(r)>0.4 else "Weak"
                    st.markdown(f"""
                    <div style='padding:7px 14px;margin:4px 0;background:#fafbfc;border:1px solid #e2e8f0;
                                border-radius:8px;font-size:0.83rem;transition:all 0.2s'>
                        <b style='color:{color}'>{a} ↔ {b}</b>:
                        <span style='font-family:JetBrains Mono,monospace'>r = {r}</span>
                        &nbsp;·&nbsp; <span style='color:{color}'>{direction}</span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.info("Need ≥ 2 numeric columns for correlation.")

        with tab_cat:
            if cat_cols:
                for col in cat_cols[:3]:
                    vc = df[col].value_counts().reset_index()
                    vc.columns = [col, 'Count']
                    vc['%'] = (vc['Count']/len(df)*100).round(1)
                    st.markdown(f"**{col}**")
                    st.dataframe(vc, use_container_width=True)
            else:
                st.info("No categorical columns detected.")

    viva("What is EDA?",
         "EDA — Exploratory Data Analysis — summarizes dataset characteristics via statistics "
         "and charts before modelling. It reveals patterns, outliers, and relationships between variables.")


# ─────────────────────────────────────────────────────────────────
# STEP 6 — VISUALIZATION
# ─────────────────────────────────────────────────────────────────
if workflow_stage == "Visualization":
    section("📈", "Visualization")

    st.markdown("""
    <div class="step-card">
        <h4>Purpose: See Patterns Visually</h4>
        <p>Charts make patterns obvious. Auto-generated: Correlation Heatmap · Histogram ·
        Scatter Plot · Line Chart · Box Plot · Bar Chart.</p>
    </div>""", unsafe_allow_html=True)

    chart_tabs = st.tabs(["🔥 Heatmap", "📦 Histogram", "🔵 Scatter",
                           "📉 Line Chart", "📊 Bar Chart", "🎁 Box Plot"])

    with chart_tabs[0]:
        if len(num_cols) >= 2:
            fig, ax = plt.subplots(figsize=(min(12, len(num_cols)*1.5+3),
                                           min(9, len(num_cols)*1.2+2)))
            apply_plot_style(fig, ax)
            mask = np.triu(np.ones_like(df[num_cols].corr(), dtype=bool))
            cmap = sns.diverging_palette(230, 15, as_cmap=True)
            sns.heatmap(df[num_cols].corr(), annot=True, fmt='.2f',
                        cmap=cmap, center=0, mask=mask,
                        linewidths=0.8, linecolor='white', ax=ax,
                        annot_kws={'size':9,'weight':'600'})
            ax.set_title('Correlation Heatmap', fontsize=13, fontweight='bold', pad=12)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
            st.info("💡 **+1** = strong positive · **−1** = strong negative · **0** = no relation")
        else:
            st.warning("Need ≥ 2 numeric columns.")

    with chart_tabs[1]:
        hist_col = st.selectbox("Select column", num_cols, key="hist_col")
        bins_n   = st.slider("Bins", 5, 50, 15, key="hist_bins")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        apply_plot_style(fig, axes[0], axes[1])

        axes[0].hist(df[hist_col].dropna(), bins=bins_n,
                     color=PALETTE[0], edgecolor='white', alpha=0.8, linewidth=0.5)
        axes[0].axvline(df[hist_col].mean(), color=PALETTE[2], linewidth=2,
                        linestyle='--', label=f'Mean = {df[hist_col].mean():.2f}')
        axes[0].axvline(df[hist_col].median(), color=PALETTE[1], linewidth=2,
                        linestyle='--', label=f'Median = {df[hist_col].median():.2f}')
        axes[0].set_title(f'Distribution — {hist_col}', fontweight='bold')
        axes[0].legend(fontsize=8)

        vals = df[hist_col].dropna()
        axes[1].hist(vals, bins=bins_n, color=PALETTE[0], edgecolor='white', alpha=0.3, density=True)
        vals.plot.kde(ax=axes[1], color=PALETTE[0], linewidth=2.5)
        axes[1].set_title(f'KDE — {hist_col}', fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with chart_tabs[2]:
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1: sc_x = st.selectbox("X-axis", num_cols, index=0, key="sc_x")
        with col_s2: sc_y = st.selectbox("Y-axis", num_cols, index=min(1, len(num_cols)-1), key="sc_y")
        with col_s3: sc_hue = st.selectbox("Color by", ["None"] + cat_cols, key="sc_hue")

        fig, ax = plt.subplots(figsize=(9, 5))
        apply_plot_style(fig, ax)

        if sc_hue != "None" and sc_hue in df.columns:
            for i, cat in enumerate(df[sc_hue].unique()):
                sub = df[df[sc_hue] == cat]
                ax.scatter(sub[sc_x], sub[sc_y], label=str(cat),
                           color=PALETTE[i % len(PALETTE)], alpha=0.72, s=55,
                           edgecolors='white', linewidth=0.8)
            ax.legend(title=sc_hue, fontsize=8)
        else:
            ax.scatter(df[sc_x], df[sc_y], color=PALETTE[0], alpha=0.72,
                       s=55, edgecolors='white', linewidth=0.8)

        valid = df[[sc_x, sc_y]].dropna()
        if len(valid) > 2:
            m, b = np.polyfit(valid[sc_x], valid[sc_y], 1)
            x_line = np.linspace(valid[sc_x].min(), valid[sc_x].max(), 100)
            ax.plot(x_line, m*x_line+b, color=PALETTE[2], linewidth=1.8,
                    linestyle='--', alpha=0.8, label='Trend')

        r_val = df[[sc_x, sc_y]].dropna().corr().iloc[0,1]
        ax.set_title(f'{sc_x} vs {sc_y}  (r = {r_val:.3f})', fontweight='bold')
        ax.set_xlabel(sc_x); ax.set_ylabel(sc_y)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with chart_tabs[3]:
        line_cols = st.multiselect("Columns", num_cols, default=num_cols[:min(3,len(num_cols))])
        line_max  = st.slider("Max rows", 20, min(200, len(df)), min(100, len(df)))
        if line_cols:
            fig, ax = plt.subplots(figsize=(11, 4))
            apply_plot_style(fig, ax)
            for i, col in enumerate(line_cols):
                ax.plot(df[col].head(line_max).values,
                        color=PALETTE[i % len(PALETTE)], linewidth=2,
                        label=col, alpha=0.9)
            ax.set_title('Line Chart — Trends', fontweight='bold')
            ax.set_xlabel('Record Index')
            ax.legend(fontsize=9)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

    with chart_tabs[4]:
        if cat_cols and num_cols:
            col_b1, col_b2 = st.columns(2)
            with col_b1: bar_cat = st.selectbox("Category (X)", cat_cols)
            with col_b2: bar_num = st.selectbox("Value (Y)", num_cols)
            bar_agg = st.radio("Aggregate by", ["Mean","Sum","Count"], horizontal=True)
            agg_map = {"Mean":"mean","Sum":"sum","Count":"count"}
            grouped = df.groupby(bar_cat)[bar_num].agg(agg_map[bar_agg]).sort_values(ascending=False)

            fig, ax = plt.subplots(figsize=(max(7, len(grouped)*0.9), 4))
            apply_plot_style(fig, ax)
            bars = ax.bar(grouped.index.astype(str), grouped.values,
                          color=[PALETTE[i % len(PALETTE)] for i in range(len(grouped))],
                          edgecolor='white', width=0.65, alpha=0.88)
            for bar in bars:
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+grouped.max()*0.01,
                        f'{bar.get_height():.1f}',
                        ha='center', va='bottom', fontsize=8, color='#475569', fontweight='600')
            ax.set_title(f'{bar_agg} of {bar_num} by {bar_cat}', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

    with chart_tabs[5]:
        box_cols = st.multiselect("Columns", num_cols, default=num_cols[:min(4,len(num_cols))])
        if box_cols:
            fig, ax = plt.subplots(figsize=(max(7, len(box_cols)*1.5), 5))
            apply_plot_style(fig, ax)
            bp = ax.boxplot([df[c].dropna() for c in box_cols],
                            labels=box_cols, patch_artist=True,
                            medianprops=dict(color=PALETTE[2], linewidth=2.5),
                            whiskerprops=dict(color='#94a3b8'),
                            capprops=dict(color='#94a3b8'),
                            flierprops=dict(marker='o', markerfacecolor='#dc2626',
                                           markersize=4, alpha=0.5, markeredgewidth=0))
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(PALETTE[i % len(PALETTE)] + '28')
                patch.set_edgecolor(PALETTE[i % len(PALETTE)])
                patch.set_linewidth(1.5)
            ax.set_title('Box Plot — Outlier Detection', fontweight='bold')
            ax.tick_params(axis='x', rotation=30)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
            st.info("💡 Dots beyond whiskers = **Outliers** · Middle line = **Median**")

    viva("What charts did you create and why?",
         "Heatmap (correlations), Histogram (distribution + KDE), Scatter with trend line "
         "(two-variable relationship), Line chart (trends), Bar chart (category comparison), "
         "Box plot (outlier detection). Each serves a specific analytical purpose.")


# ─────────────────────────────────────────────────────────────────
# STEP 7 — ML PREDICTION
# ─────────────────────────────────────────────────────────────────
if workflow_stage == "ML Prediction":
    section("🤖", "ML Prediction")

    st.markdown("""
    <div class="step-card">
        <h4>ML Models Applied Automatically</h4>
        <p><b>Linear Regression</b> → Predicts a numeric value (e.g., predict Science from Math)<br>
        <b>Logistic Regression</b> → Classifies into categories (e.g., predict Grade A/B/C/D/F)</p>
    </div>""", unsafe_allow_html=True)

    ml_tabs = st.tabs(["📈 Linear Regression", "🏷 Logistic Regression"])

    default_lr_x, default_lr_y = None, None
    default_lo_target, default_lo_feats = None, None

    if dataset_choice == "📘 student_marks.csv":
        default_lr_x, default_lr_y = "Math", "Science"
        default_lo_target = "Grade"
        default_lo_feats  = [c for c in num_cols if c not in ['StudentID']]
    elif dataset_choice == "📗 retail_store.csv":
        default_lr_x, default_lr_y = "Quantity", "Revenue"
        default_lo_target = "StoreID" if 'StoreID' in cat_cols else None
        default_lo_feats  = [c for c in num_cols if c not in ['Revenue','InvoiceID']]
    elif dataset_choice == "📕 heart_disease.csv":
        default_lr_x, default_lr_y = "Age", "Cholesterol"
        if 'HeartDisease' in num_cols and 'HeartDisease' not in cat_cols:
            cat_cols.append('HeartDisease')
        default_lo_target = "HeartDisease"
        default_lo_feats  = [c for c in num_cols if c != 'HeartDisease']
    else:
        default_lr_x = num_cols[0] if num_cols else None
        default_lr_y = num_cols[1] if len(num_cols) > 1 else None
        default_lo_target = cat_cols[0] if cat_cols else None
        default_lo_feats  = num_cols[:min(3, len(num_cols))]

    # ── 7.1 LINEAR REGRESSION ──
    with ml_tabs[0]:
        if len(num_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_i  = num_cols.index(default_lr_x) if default_lr_x in num_cols else 0
                lr_x = st.selectbox("Feature (X)", num_cols, index=x_i)
            with col2:
                lr_y_opts = [c for c in num_cols if c != lr_x]
                y_i  = lr_y_opts.index(default_lr_y) if default_lr_y in lr_y_opts else 0
                lr_y = st.selectbox("Target (Y)", lr_y_opts, index=y_i)
            test_size = st.slider("Test set size (%)", 10, 40, 20) / 100

            if st.button("▶  Run Linear Regression", type="primary"):
                with st.spinner("Training model..."):
                    valid_df = df[[lr_x, lr_y]].dropna()
                    X = valid_df[[lr_x]].values; y = valid_df[lr_y].values
                    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42)
                    model = LinearRegression()
                    model.fit(Xtr, ytr)
                    yp = model.predict(Xte)
                    mse = mean_squared_error(yte, yp)
                    r2  = r2_score(yte, yp)

                metric_cards([
                    (f"{r2:.4f}", "R² Score"),
                    (f"{mse:.3f}", "MSE"),
                    (f"{np.sqrt(mse):.3f}", "RMSE"),
                    (f"{model.coef_[0]:.4f}", "Coefficient"),
                    (f"{model.intercept_:.4f}", "Intercept"),
                ])

                if r2 >= 0.8:   st.success("🟢 Excellent model fit!")
                elif r2 >= 0.6: st.success("🟡 Good model fit.")
                elif r2 >= 0.4: st.warning("🟠 Moderate fit — try more features.")
                else:           st.error("🔴 Weak fit — variables may not be linear.")

                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                apply_plot_style(fig, axes[0], axes[1])

                axes[0].scatter(Xte, yte, color=PALETTE[0], s=55, alpha=0.75,
                                edgecolors='white', linewidth=0.8, label='Actual')
                axes[0].scatter(Xte, yp, color=PALETTE[2], s=55, alpha=0.75,
                                edgecolors='white', linewidth=0.8, marker='^', label='Predicted')
                xl = np.linspace(X.min(), X.max(), 200)
                axes[0].plot(xl, model.predict(xl.reshape(-1,1)),
                             color=PALETTE[3], linewidth=2, linestyle='--', label='Regression Line')
                axes[0].set_title(f'Linear Regression: {lr_x} → {lr_y}', fontweight='bold')
                axes[0].legend(fontsize=8)

                res = yte - yp
                axes[1].scatter(yp, res, color=PALETTE[4], alpha=0.7, s=55,
                                edgecolors='white', linewidth=0.8)
                axes[1].axhline(0, color=PALETTE[2], linestyle='--', linewidth=1.8)
                axes[1].set_title('Residual Plot', fontweight='bold')
                axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Residuals')

                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

                st.markdown(f"""
                <div class="code-box">Regression Equation:
{lr_y}  =  {model.coef_[0]:.4f} × {lr_x}  +  {model.intercept_:.4f}

Train samples : {len(Xtr)}
Test  samples : {len(Xte)}
R² Score      : {r2:.4f}  →  Model explains {r2*100:.1f}% of variance</div>""",
                unsafe_allow_html=True)

                st.markdown("**🔮 Make a Prediction**")
                user_val = st.number_input(f"Enter {lr_x} value", value=float(df[lr_x].mean()))
                pred_out = model.predict([[user_val]])[0]
                st.success(f"If **{lr_x} = {user_val}** → Predicted **{lr_y} = {pred_out:.2f}**")
        else:
            st.warning("Need ≥ 2 numeric columns for Linear Regression.")

    # ── 7.2 LOGISTIC REGRESSION ──
    with ml_tabs[1]:
        if len(num_cols) >= 1:
            col1, col2 = st.columns(2)
            with col1:
                lo_opts = list(cat_cols)
                if default_lo_target and default_lo_target in df.columns and default_lo_target not in lo_opts:
                    lo_opts.append(default_lo_target)
                if not lo_opts:
                    st.warning("No categorical target available.")
                    lo_target = None
                else:
                    ti = lo_opts.index(default_lo_target) if default_lo_target in lo_opts else 0
                    lo_target = st.selectbox("Target (categorical)", lo_opts, index=ti)
            with col2:
                dlo = [c for c in (default_lo_feats or []) if c in num_cols] or num_cols[:min(3,len(num_cols))]
                lo_feats = st.multiselect("Feature columns (numeric)", num_cols, default=dlo)
            lo_test = st.slider("Test set size (%)", 10, 40, 20, key="lo_test") / 100

            if lo_target and lo_feats and st.button("▶  Run Logistic Regression", type="primary"):
                with st.spinner("Training classifier..."):
                    sub = df[lo_feats + [lo_target]].dropna()
                    le  = LabelEncoder()
                    y   = le.fit_transform(sub[lo_target])
                    X   = sub[lo_feats].values
                    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=lo_test, random_state=42)
                    clf = LogisticRegression(max_iter=1000)
                    clf.fit(Xtr, ytr)
                    yp  = clf.predict(Xte)
                    acc = accuracy_score(yte, yp)
                    rep = classification_report(yte, yp, target_names=le.classes_, zero_division=0)
                    cm  = confusion_matrix(yte, yp)

                metric_cards([
                    (f"{acc*100:.1f}%", "Accuracy"),
                    (len(Xtr), "Train Samples"),
                    (len(Xte), "Test Samples"),
                    (len(le.classes_), "Classes"),
                ])

                if acc >= 0.85:   st.success(f"🟢 High accuracy — {acc*100:.1f}%!")
                elif acc >= 0.65: st.warning(f"🟡 Moderate — {acc*100:.1f}%. Acceptable.")
                else:             st.error(f"🔴 Low accuracy — {acc*100:.1f}%. Add more features.")

                col_cm, col_rep = st.columns(2)
                with col_cm:
                    st.markdown("**Confusion Matrix**")
                    fig, ax = plt.subplots(figsize=(5, 4))
                    apply_plot_style(fig, ax)
                    cmap2 = sns.light_palette("#2563eb", as_cmap=True)
                    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap2,
                                xticklabels=le.classes_, yticklabels=le.classes_,
                                ax=ax, linewidths=1, linecolor='white')
                    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
                    ax.set_title('Confusion Matrix', fontweight='bold')
                    st.pyplot(fig, use_container_width=True)
                    plt.close()
                    st.caption("Diagonal = correct · Off-diagonal = errors")
                with col_rep:
                    st.markdown("**Classification Report**")
                    st.code(rep, language=None)
        else:
            st.warning("Need at least one categorical and one numeric column.")

    viva("What ML models did you use and what is the difference?",
         "Linear Regression predicts continuous values (e.g., marks) evaluated with R². "
         "Logistic Regression classifies into categories (e.g., Grade) evaluated with Accuracy "
         "and Confusion Matrix. Both use 80/20 train-test split with random_state=42.")


# ─────────────────────────────────────────────────────────────────
# STEP 8 — RESULT SUMMARY
# ─────────────────────────────────────────────────────────────────
if workflow_stage == "Result Summary":
    section("🧾", "Result Summary & Conclusion")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Dataset Summary**")
        summary = pd.DataFrame({
            'Metric': ['Original Rows','Columns','Missing (raw)',
                       'Duplicates Removed','Clean Rows',
                       'Numeric Cols','Categorical Cols'],
            'Value':  [f"{df_raw.shape[0]:,}", df_raw.shape[1], null_total,
                       dupes_removed, f"{len(df):,}",
                       len(num_cols), len(cat_cols)]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**Downloads**")
        st.download_button("⬇️  Cleaned Dataset (CSV)",
                           df.to_csv(index=False).encode(),
                           "cleaned_dataset.csv", "text/csv")
        if num_cols:
            st.download_button("⬇️  Statistics Summary (CSV)",
                               df[num_cols].describe().round(3).to_csv().encode(),
                               "statistics_summary.csv", "text/csv")
        if 'df_filtered' in locals() and len(df_filtered) > 0:
            st.download_button("⬇️  Filtered Dataset (CSV)",
                               df_filtered.to_csv(index=False).encode(),
                               "filtered_dataset.csv", "text/csv")

    st.markdown("---")
    st.markdown("**Automatic Insights**")
    if num_cols:
        for col in num_cols[:4]:
            vals = df[col].dropna()
            m, s = vals.mean(), vals.std()
            cv = s/m*100 if m != 0 else 0
            insight("📌", f"{col} — Key Findings",
                    f"Range: <b>{vals.min():.2f} – {vals.max():.2f}</b> | "
                    f"Mean: <b>{m:.2f}</b> | Std: <b>{s:.2f}</b> | Variation: <b>{cv:.1f}%</b> — "
                    f"{'Stable / consistent' if cv < 20 else 'High variation detected'}")

    st.markdown("---")
    st.markdown("""
    <div class="conclusion-block">
        <h2>🎯 Project Conclusion</h2>
        <p>
            <b>SmartStream: An End-to-End Interactive Data Science Playground</b> automates the complete data science pipeline.<br>
            It accepts any CSV and performs<br><br>
            <b>Data Cleaning → Filtering → EDA → Visualization → ML Prediction</b><br><br>
            without writing dataset-specific code.<br><br>
            <i>"A smart Python-based system that converts raw data into meaningful insights automatically."</i>
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Viva Preparation**")
    viva("What is your project?",
         "SmartStream: An End-to-End Interactive Data Science Playground built with Python and Streamlit. Any CSV can be uploaded and "
         "the system auto-performs cleaning, filtering, EDA, visualization (Matplotlib/Seaborn), "
         "and ML prediction (Linear & Logistic Regression).")
    viva("Why did you use Streamlit?",
         "Streamlit converts Python scripts into interactive web apps. It enabled a user-friendly "
         "interface for file upload, options, and results — no web development knowledge needed.")
    viva("What makes your project universal?",
         "Dynamic column detection — it auto-identifies numeric and categorical columns from "
         "any dataset. No column names are hardcoded. Works for student, sales, medical, or any CSV.")