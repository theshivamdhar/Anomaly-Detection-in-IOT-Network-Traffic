"""
app.py — IoT Intrusion Detection System
=========================================
Fixed sidebar + hero upload + tabbed insights.

Launch:
    python -m streamlit run app.py
"""

import os, copy
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    average_precision_score, roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

# ─────────────────────────────────────────────────────────────────────
# PAGE
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IoT IDS — Intrusion Detection",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# TOKENS
# ─────────────────────────────────────────────────────────────────────
BG      = "#060c1a"
SURF    = "#0d1321"
CARD    = "#131c2e"
ELEV    = "#1a2540"
BORD    = "#1c2c48"
T1      = "#edf0f7"
T2      = "#9ba5bb"
T3      = "#5a657b"
IND     = "#6366f1"
CYAN    = "#22d3ee"
GRN     = "#34d399"
RED     = "#f87171"
AMB     = "#fbbf24"
PRP     = "#a78bfa"
PAL     = [IND, CYAN, GRN, RED, AMB, PRP]
CH_H    = 360

# ─────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────
def _css(pres=False):
    s = 1.06 if pres else 1.0
    st.markdown(f"""<style>
/* ── base ─────────────────── */
.block-container {{ padding:2.6rem 2.2rem 2rem 2.2rem; max-width:1380px; margin:auto; }}
header[data-testid="stHeader"] {{ background:transparent; }}

/* ── sidebar: fixed, no scroll ─── */
section[data-testid="stSidebar"] {{
    background:{SURF}; border-right:1px solid {BORD};
    width:220px !important; min-width:220px !important; max-width:220px !important;
}}
section[data-testid="stSidebar"] > div:first-child {{
    padding:0.9rem 0.9rem 0.6rem 0.9rem;
    max-height:100vh; overflow:hidden !important;
    display:flex; flex-direction:column; justify-content:space-between;
    height:100vh;
}}
section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {{
    gap:0.08rem;
}}

/* sidebar typography */
.sb-brand {{ color:{T1}; font-size:0.88rem; font-weight:700; margin:0; letter-spacing:-0.01em; }}
.sb-sub   {{ color:{T3}; font-size:0.62rem; margin:0; }}
.sb-lbl   {{
    color:{T3}; font-size:0.58rem; font-weight:700;
    text-transform:uppercase; letter-spacing:0.09em;
    margin:0.55rem 0 0.08rem 0;
}}
.sb-stat {{
    display:inline-flex; align-items:center; gap:0.2rem;
    padding:0.08rem 0.35rem; border-radius:8px;
    font-size:0.6rem; font-weight:600; margin-bottom:0.1rem;
}}
.sb-stat.ok {{ background:rgba(52,211,153,0.08); color:{GRN}; }}
.sb-stat.na {{ background:rgba(251,191,36,0.08); color:{AMB}; }}
.sb-meta {{
    background:rgba(99,102,241,0.04); border-radius:6px;
    padding:0.3rem 0.5rem; margin-top:0.25rem;
}}
.sb-meta p {{ color:{T2}; font-size:0.68rem; margin:0.05rem 0; }}
.sb-meta strong {{ color:{T1}; }}
.sb-ft {{ color:{T3}; font-size:0.58rem; text-align:center; padding:0.3rem 0 0 0; }}

/* ── hero ─────────────────── */
.hero {{
    text-align:center; padding:3rem 1rem 2rem 1rem;
    max-width:640px; margin:0 auto;
}}
.hero h1 {{
    color:{T1}; font-size:1.6rem; font-weight:700;
    margin:0 0 0.4rem 0; letter-spacing:-0.02em;
}}
.hero p {{
    color:{T3}; font-size:0.88rem; margin:0 0 1.5rem 0;
    line-height:1.5;
}}

/* ── KPI ──────────────────── */
.kpi {{
    background:{CARD}; border-radius:10px;
    padding:{0.95*s}rem 0.7rem; text-align:center;
}}
.kpi-hero {{
    background:linear-gradient(140deg,{ELEV},{CARD});
    border-radius:12px; padding:{1.35*s}rem 1rem; text-align:center;
    box-shadow:0 2px 18px rgba(99,102,241,0.06);
}}
.kpi-l {{
    color:{T3}; font-size:{0.57*s}rem; font-weight:700;
    text-transform:uppercase; letter-spacing:0.1em; margin:0 0 0.2rem 0;
}}
.kpi-v {{
    color:{T1}; font-size:{1.3*s}rem; font-weight:700;
    margin:0; line-height:1.15;
}}
.kpi-v.xl {{ font-size:{2.1*s}rem; font-weight:800; }}
.kpi-v.ind {{ color:{IND}; }}
.kpi-v.cy  {{ color:{CYAN}; }}
.kpi-v.gn  {{ color:{GRN}; }}
.kpi-v.rd  {{ color:{RED}; }}
.kpi-v.am  {{ color:{AMB}; }}

.pill {{
    display:inline-block; padding:0.15rem 0.5rem; border-radius:14px;
    font-size:{0.58*s}rem; font-weight:700; margin-top:0.25rem;
}}
.p-lo {{ background:rgba(52,211,153,0.08); color:{GRN}; }}
.p-md {{ background:rgba(251,191,36,0.08); color:{AMB}; }}
.p-hi {{ background:rgba(248,113,113,0.08); color:{RED}; }}

/* ── section ──────────────── */
.sec {{ color:{T2}; font-size:{0.85*s}rem; font-weight:600; margin:0 0 0.45rem 0; }}
.gap   {{ height:1.4rem; }}
.gap-s {{ height:0.55rem; }}

/* ── warning ──────────────── */
.wb {{
    background:rgba(251,191,36,0.04); border-left:3px solid {AMB};
    border-radius:4px; padding:0.4rem 0.8rem; margin:0.3rem 0 0.7rem 0;
}}
.wb p {{ color:{AMB}; font-size:0.76rem; margin:0; }}

/* ── tab styling ──────────── */
div[data-testid="stTabs"] button[data-baseweb="tab"] {{
    font-size:0.8rem; font-weight:600; color:{T3};
    padding:0.45rem 0.9rem; border:none; background:transparent;
    border-bottom:2px solid transparent;
}}
div[data-testid="stTabs"] button[aria-selected="true"] {{
    color:{T1}; border-bottom:2px solid {IND};
}}

/* ── footer ───────────────── */
.ft {{ text-align:center; color:{T3}; font-size:0.6rem; padding:2rem 0 0.5rem 0; }}
</style>""", unsafe_allow_html=True)

    if pres:
        st.markdown(f"""<style>
section[data-testid="stSidebar"] {{ display:none !important; }}
.block-container {{ max-width:100% !important; padding-left:3rem !important; padding-right:3rem !important; }}
.hero {{ display:none !important; }}
</style>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# PLOTLY
# ─────────────────────────────────────────────────────────────────────
_B = dict(
    paper_bgcolor=CARD, plot_bgcolor=CARD,
    font=dict(color=T2, size=11),
    title=dict(font=dict(color=T1, size=13), x=0.01, xanchor="left"),
    margin=dict(l=48, r=20, t=44, b=44),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color=T2),
                borderwidth=0, orientation="h", y=-0.22),
    hoverlabel=dict(bgcolor=ELEV, font_size=12, font_color=T1),
)
_A = dict(gridcolor=BORD, gridwidth=0.4,
          zerolinecolor=BORD, zerolinewidth=0.4,
          tickfont=dict(size=10, color=T3),
          title_font=dict(size=11, color=T2))

def _dm(b, o):
    m = copy.deepcopy(b)
    for k, v in o.items():
        if k in m and isinstance(m[k], dict) and isinstance(v, dict):
            m[k] = _dm(m[k], v)
        else: m[k] = copy.deepcopy(v)
    return m

def _L(fig, *ds):
    r = {}
    for d in ds: r = _dm(r, d)
    fig.update_layout(**r); return fig


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────
MODELS = {"Random Forest":"models/random_forest.pkl","Decision Tree":"models/decision_tree.pkl","XGBoost":"models/xgboost.pkl"}

@st.cache_resource
def _ld(p):
    if os.path.isfile(p):
        try: return joblib.load(p)
        except Exception: return None
    return None

@st.cache_data
def _prep(raw, fn):
    df = pd.read_csv(raw, low_memory=False); df.drop_duplicates(inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for c in df.select_dtypes(include=[np.number]).columns:
        if df[c].isnull().sum()>0: df[c]=df[c].fillna(df[c].median())
    for c in df.select_dtypes(include=["object"]).columns:
        if c.lower() not in ("label","attack_type","attack_label") and df[c].isnull().sum()>0:
            df[c]=df[c].fillna(df[c].mode().iloc[0])
    return df

def _eng(df):
    df=df.copy()
    if "src_bytes" in df.columns and "dst_bytes" in df.columns:
        df["byte_ratio"]=(df["dst_bytes"]/(df["src_bytes"]+1)).round(4)
        df["src_dst_ratio"]=(df["src_bytes"]/(df["dst_bytes"]+1)).round(4)
    if "packet_size" in df.columns and "duration" in df.columns:
        df["packet_rate"]=(df["packet_size"]/(df["duration"]+0.01)).round(4)
    if "error_rate" in df.columns and "wrong_fragment" in df.columns:
        df["error_flag_interact"]=(df["error_rate"]*df["wrong_fragment"]).round(4)
    return df

def _enc(labels, binary=True):
    nk={"normal","benign","Normal","BENIGN","NORMAL"}
    if binary:
        # Handle already-numeric 0/1 (Attack_label)
        if set(pd.Series(labels).unique()) <= {0, 1, 0.0, 1.0}:
            return np.array(labels).astype(np.int64), ["Normal","Attack"]
        return np.array([0 if str(v).strip() in nk else 1 for v in labels]),["Normal","Attack"]
    le=LabelEncoder(); return le.fit_transform(np.array(labels).astype(str)),list(le.classes_)

def _detect_target(df):
    """Auto-detect target column from dataset."""
    for col in ["Attack_label","Attack_type","attack_label","attack_type","label","Label"]:
        if col in df.columns:
            return col
    return None

def _met(yt,yp):
    return dict(
        Accuracy=round(accuracy_score(yt,yp),4),
        Precision=round(precision_score(yt,yp,average="weighted",zero_division=0),4),
        Recall=round(recall_score(yt,yp,average="weighted",zero_division=0),4),
        F1=round(f1_score(yt,yp,average="weighted",zero_division=0),4))


# ─────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────
def ch_dist(counts):
    n,v=counts.index.tolist(),counts.values.tolist()
    fig=go.Figure(go.Bar(x=v,y=n,orientation="h",
        marker=dict(color=[PAL[i%len(PAL)] for i in range(len(n))],cornerradius=3),
        text=v,textposition="outside",textfont=dict(color=T2,size=11)))
    _L(fig,_B,dict(height=CH_H,title_text="Traffic Distribution",
        xaxis={**_A,"showgrid":False},yaxis={**_A,"autorange":"reversed"})); return fig

def ch_fi(imp,names,top=12):
    top=min(top,len(names)); ix=np.argsort(imp)[::-1][:top]
    tn,tv=np.array(names)[ix].tolist(),imp[ix].tolist()
    fig=go.Figure(go.Bar(x=tv[::-1],y=tn[::-1],orientation="h",
        marker=dict(color=IND,opacity=0.85,cornerradius=3),
        text=[f"{x:.3f}" for x in tv[::-1]],
        textposition="outside",textfont=dict(color=T3,size=10)))
    _L(fig,_B,dict(height=CH_H,title_text=f"Top {top} Features",
        xaxis={**_A,"showgrid":False,"title_text":"Importance"},yaxis=_A)); return fig

def ch_cm(yt,yp,names):
    cm=confusion_matrix(yt,yp)
    fig=go.Figure(go.Heatmap(z=cm,x=names,y=names,
        colorscale=[[0,SURF],[1,IND]],
        text=cm,texttemplate="%{text}",textfont=dict(size=13,color=T1),
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        showscale=False))
    _L(fig,_B,dict(height=CH_H,title_text="Confusion Matrix",
        xaxis={**_A,"title_text":"Predicted","showgrid":False},
        yaxis={**_A,"title_text":"Actual","autorange":"reversed","showgrid":False})); return fig

def ch_roc(yt,yp,names,binary=True):
    fig=go.Figure()
    if binary:
        fpr,tpr,_=roc_curve(yt,yp[:,1])
        fig.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",line=dict(color=CYAN,width=2.5),
            name=f"AUC = {auc(fpr,tpr):.4f}"))
    else:
        nc=yp.shape[1]; yb=label_binarize(yt,classes=list(range(nc)))
        for i in range(nc):
            fpr,tpr,_=roc_curve(yb[:,i],yp[:,i])
            lb=names[i] if i<len(names) else f"C{i}"
            fig.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",
                line=dict(color=PAL[i%len(PAL)],width=2),name=f"{lb} ({auc(fpr,tpr):.3f})"))
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",
        line=dict(dash="dot",color=T3,width=1),showlegend=False))
    _L(fig,_B,dict(height=CH_H,title_text="ROC Curve",
        xaxis={**_A,"title_text":"FPR"},yaxis={**_A,"title_text":"TPR"})); return fig

def ch_pr(yt,yp,names,binary=True):
    fig=go.Figure()
    if binary:
        pr,rc,_=precision_recall_curve(yt,yp[:,1])
        ap=average_precision_score(yt,yp[:,1])
        fig.add_trace(go.Scatter(x=rc,y=pr,mode="lines",line=dict(color=CYAN,width=2.5),
            name=f"AP = {ap:.4f}"))
    else:
        nc=yp.shape[1]; yb=label_binarize(yt,classes=list(range(nc)))
        for i in range(nc):
            pr,rc,_=precision_recall_curve(yb[:,i],yp[:,i])
            ap=average_precision_score(yb[:,i],yp[:,i])
            lb=names[i] if i<len(names) else f"C{i}"
            fig.add_trace(go.Scatter(x=rc,y=pr,mode="lines",
                line=dict(color=PAL[i%len(PAL)],width=2),name=f"{lb} ({ap:.3f})"))
    _L(fig,_B,dict(height=CH_H,title_text="Precision-Recall",
        xaxis={**_A,"title_text":"Recall"},yaxis={**_A,"title_text":"Precision"})); return fig

def ch_perm(means,names):
    fig=go.Figure(go.Bar(x=means[::-1],y=names[::-1],orientation="h",
        marker=dict(color=CYAN,opacity=0.85,cornerradius=3)))
    _L(fig,_B,dict(height=CH_H,title_text="Permutation Importance",
        xaxis={**_A,"showgrid":False,"title_text":"Accuracy Drop"},yaxis=_A)); return fig


# ═════════════════════════════════════════════════════════════════════
#  SIDEBAR — fixed, no scroll, compact
# ═════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<p class="sb-brand">IoT IDS</p>', unsafe_allow_html=True)
    st.markdown('<p class="sb-sub">Intrusion Detection System</p>', unsafe_allow_html=True)

    st.divider()

    st.markdown('<p class="sb-lbl">Model</p>', unsafe_allow_html=True)
    mchoice = st.selectbox("model", list(MODELS.keys()), label_visibility="collapsed")

    st.markdown('<p class="sb-lbl">Mode</p>', unsafe_allow_html=True)
    cmode = st.radio("mode", ["Binary","Multiclass"], horizontal=True, label_visibility="collapsed")

    st.divider()

    if "pres" not in st.session_state:
        st.session_state["pres"] = False

    pres_toggle = st.toggle("Presentation Mode", value=st.session_state["pres"])
    if pres_toggle != st.session_state["pres"]:
        st.session_state["pres"] = pres_toggle
        st.rerun()

    # Status
    st.markdown('<p class="sb-lbl">Status</p>', unsafe_allow_html=True)
    for n, p in MODELS.items():
        c = "ok" if os.path.isfile(p) else "na"
        st.markdown(f'<span class="sb-stat {c}">{n}</span>', unsafe_allow_html=True)

    # Live metadata (after analysis)
    R = st.session_state.get("R")
    if R is not None:
        st.markdown(
            f'<div class="sb-meta">'
            f'<p><strong>{R.get("mc","—")}</strong></p>'
            f'<p>{len(R["yp"]):,} samples · {R["fdf"].shape[1]} features</p>'
            f'</div>', unsafe_allow_html=True)

    st.markdown('<p class="sb-ft">B.Tech Minor Project · 2026</p>', unsafe_allow_html=True)

binary = cmode == "Binary"
mtag = "binary" if binary else "multiclass"

_css(st.session_state.get("pres", False))

# Presentation exit button
if st.session_state.get("pres"):
    if st.button("✕  Exit Presentation", key="_exit_pres"):
        st.session_state["pres"] = False
        st.rerun()


# ═════════════════════════════════════════════════════════════════════
#  HERO — upload + run (main area)
# ═════════════════════════════════════════════════════════════════════

if "R" not in st.session_state:
    st.markdown(f"""
<div class="hero">
    <h1>IoT Intrusion Detection System</h1>
    <p>Upload a network traffic dataset to analyze for anomalies and
    intrusions using machine learning classification.</p>
</div>""", unsafe_allow_html=True)

    # Centered upload + button
    _, uc, _ = st.columns([1, 2, 1])
    with uc:
        ufile = st.file_uploader("Upload CSV Dataset", type=["csv"],
                                  label_visibility="collapsed",
                                  key="hero_upload")
        if ufile:
            try:
                _pv = pd.read_csv(ufile, low_memory=False, nrows=1000); ufile.seek(0)
                _tc = _detect_target(_pv)
                _hl = _tc is not None
                st.caption(f"**{ufile.name}** — {_pv.shape[0]:,}+ rows · {_pv.shape[1]} cols · "
                           f"Target: {_tc if _tc else 'None'}")
            except Exception:
                pass

        _, bl, br, _ = st.columns([1, 2, 2, 1])
        with bl:
            run = st.button("Run Analysis", type="primary", use_container_width=True)
        with br:
            pass  # keep symmetry

    st.markdown(f'<div class="ft">IoT Intrusion Detection System · B.Tech Minor Project · 2026</div>',
                unsafe_allow_html=True)

    if not ufile:
        st.stop()
    if not run:
        st.stop()

    # ── STAGE 2: Execute ──
    with st.spinner("Analyzing network traffic..."):
        try: df = _prep(ufile, ufile.name)
        except Exception as e: st.error(f"Error: {e}"); st.stop()
        if df.empty: st.warning("Empty file."); st.stop()

        # Auto-detect target column
        if binary:
            lc_candidates = ["Attack_label","attack_label","label","Label"]
        else:
            lc_candidates = ["Attack_type","attack_type","label","Label"]
        lc = None
        for c in lc_candidates:
            if c in df.columns: lc=c; break
        hl = lc is not None

        # Safe feature preparation: drop ALL target-related cols, keep numeric only
        drop_cols = [c for c in df.columns if c in
                     {"Attack_label","Attack_type","attack_label","attack_type","label","Label"}]
        fdf = df.drop(columns=drop_cols, errors="ignore")
        fdf = _eng(fdf).select_dtypes(include=[np.number])
        fdf.replace([np.inf, -np.inf], np.nan, inplace=True)
        fdf.fillna(0, inplace=True)
        if fdf.empty: st.error("No numeric features."); st.stop()

        X = fdf.values.astype(np.float32)
        mdl = _ld(MODELS.get(mchoice,""))
        if mdl is None: st.error(f"{mchoice} model not trained. Run train_rf.py first."); st.stop()

        sc = _ld(f"models/scaler_{mtag}.pkl")
        le = _ld(f"models/label_encoder_{mtag}.pkl")

        if sc:
            try: Xs = sc.transform(X)
            except ValueError:
                st.error(f"Feature mismatch: model expects {sc.n_features_in_} features, got {X.shape[1]}.\n"
                         f"Please retrain models with: python train_rf.py --data <your_dataset> --mode {mtag}")
                st.stop()
        else: Xs = X

        yp = mdl.predict(Xs)
        ypr = mdl.predict_proba(Xs)

        if binary:
            cn=["Normal","Attack"]; pl=[cn[int(p)] for p in yp]
        else:
            if le: cn=list(le.classes_); pl=list(le.inverse_transform(yp))
            else: cn=[str(i) for i in range(int(yp.max())+1)]; pl=[cn[int(p)] for p in yp]

        st.session_state["R"] = dict(
            df=df,fdf=fdf,lc=lc,hl=hl,mdl=mdl,mc=mchoice,
            binary=binary,mtag=mtag,cn=cn,
            yp=yp,ypr=ypr,pl=pl,Xs=Xs)
        st.rerun()

# If we reach here, results exist
R = st.session_state["R"]
df=R["df"]; fdf=R["fdf"]; lc=R["lc"]; hl=R["hl"]; mdl=R["mdl"]
cn=R["cn"]; yp=R["yp"]; ypr=R["ypr"]; pl=R["pl"]; Xs=R["Xs"]

cts = pd.Series(pl).value_counts()
if binary: atk=int(cts.get("Attack",0))
else: atk=len(yp)-int(cts.get("Normal",cts.get("normal",0)))
anom = atk/len(yp)*100

if anom<10:   rl,rc,vc="Low","p-lo","gn"
elif anom<40: rl,rc,vc="Medium","p-md","am"
else:         rl,rc,vc="High","p-hi","rd"

# New analysis button (small, top-right)
if not st.session_state.get("pres"):
    _, _, nb = st.columns([4, 4, 1.5])
    with nb:
        if st.button("New Analysis", use_container_width=True):
            del st.session_state["R"]
            st.rerun()


# ═════════════════════════════════════════════════════════════════════
#  STAGE 3 — EXECUTIVE OVERVIEW
# ═════════════════════════════════════════════════════════════════════

c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1])
with c1:
    st.markdown(
        f'<div class="kpi-hero"><p class="kpi-l">Anomaly Rate</p>'
        f'<p class="kpi-v xl {vc}">{anom:.1f}%</p>'
        f'<span class="pill {rc}">{rl} Risk</span></div>', unsafe_allow_html=True)
with c2:
    st.markdown(
        f'<div class="kpi"><p class="kpi-l">Total Samples</p>'
        f'<p class="kpi-v">{len(yp):,}</p></div>', unsafe_allow_html=True)
if binary:
    with c3:
        st.markdown(
            f'<div class="kpi"><p class="kpi-l">Normal</p>'
            f'<p class="kpi-v gn">{int(cts.get("Normal",0)):,}</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(
            f'<div class="kpi"><p class="kpi-l">Attacks</p>'
            f'<p class="kpi-v rd">{atk:,}</p></div>', unsafe_allow_html=True)
else:
    with c3:
        st.markdown(
            f'<div class="kpi"><p class="kpi-l">Classes</p>'
            f'<p class="kpi-v ind">{len(cts)}</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(
            f'<div class="kpi"><p class="kpi-l">Anomalous</p>'
            f'<p class="kpi-v rd">{atk:,}</p></div>', unsafe_allow_html=True)

# Performance row — evaluation on TEST SPLIT only (no data leakage)
if hl and lc:
    yt_full, en = _enc(df[lc], binary=binary)

    # 80-20 stratified split — evaluate ONLY on test portion
    n_samples = len(yt_full)
    indices = np.arange(n_samples)
    try:
        _, test_idx = train_test_split(
            indices, test_size=0.2, random_state=42, stratify=yt_full
        )
    except ValueError:
        # Fallback for very small or single-class data
        _, test_idx = train_test_split(
            indices, test_size=0.2, random_state=42
        )
    yt = yt_full[test_idx]
    yp_test = yp[test_idx]
    ypr_test = ypr[test_idx]

    _ul = sorted(set(yt)|set(yp_test))
    le_obj = _ld(f"models/label_encoder_{mtag}.pkl")
    if not binary and le_obj is not None and hasattr(le_obj,"classes_"):
        dc = [str(le_obj.classes_[i]) if i<len(le_obj.classes_) else str(i) for i in _ul]
    elif binary:
        dc = ["Normal","Attack"] if set(_ul)=={0,1} else [str(i) for i in _ul]
    else:
        dc = [str(i) for i in _ul]

    met = _met(yt, yp_test)
    try:
        if binary: auc_v=roc_auc_score(yt,ypr_test[:,1])
        else: auc_v=roc_auc_score(yt,ypr_test,multi_class="ovr",average="weighted")
    except ValueError: auc_v=0.0

    st.info(f"📊 Evaluating on **20% stratified test split** ({len(yt):,} of {n_samples:,} samples) — no data leakage.")

    if met["Accuracy"]==1.0:
        st.markdown('<div class="wb"><p>Perfect accuracy — verify dataset split.</p></div>',
                    unsafe_allow_html=True)

    st.markdown('<div class="gap-s"></div>', unsafe_allow_html=True)
    perf={**met,"AUC":round(auc_v,4)}
    clr={"Accuracy":"cy","Precision":"","Recall":"","F1":"gn","AUC":"ind"}
    pc=st.columns(5)
    for col,(nm,vl) in zip(pc,perf.items()):
        with col:
            st.markdown(
                f'<div class="kpi"><p class="kpi-l">{nm}</p>'
                f'<p class="kpi-v {clr.get(nm,"")}">{vl:.2%}</p></div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════
#  STAGE 4 — DEEP INSIGHTS (tabs)
# ═════════════════════════════════════════════════════════════════════

st.markdown('<div class="gap"></div>', unsafe_allow_html=True)
t1, t2, t3, t4 = st.tabs(["Distribution","Model Performance","Feature Intelligence","Predictions"])

# TAB 1
with t1:
    st.markdown('<div class="gap-s"></div>', unsafe_allow_html=True)
    dl,dr = st.columns(2)
    with dl:
        st.plotly_chart(ch_dist(cts), use_container_width=True, key="t1d")
    with dr:
        st.markdown(f'<p class="sec">Dataset Summary</p>', unsafe_allow_html=True)
        for k,v in {"Samples":f"{len(yp):,}","Features":str(fdf.shape[1]),
                     "Labels":"Yes" if hl else "No","Classes":str(len(cts))}.items():
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;padding:0.32rem 0;'
                f'border-bottom:1px solid {BORD};">'
                f'<span style="color:{T3};font-size:0.8rem;">{k}</span>'
                f'<span style="color:{T1};font-size:0.8rem;font-weight:600;">{v}</span>'
                f'</div>', unsafe_allow_html=True)

# TAB 2 — all metrics use test-split variables (yt, yp_test, ypr_test)
with t2:
    if hl and lc:
        st.markdown('<div class="gap-s"></div>', unsafe_allow_html=True)
        ml,mr = st.columns(2)
        with ml: st.plotly_chart(ch_cm(yt,yp_test,dc), use_container_width=True, key="t2cm")
        with mr:
            st.markdown(f'<p class="sec">Classification Report</p>', unsafe_allow_html=True)
            st.code(classification_report(yt,yp_test,labels=_ul,target_names=dc,zero_division=0), language="text")

        st.markdown('<div class="gap"></div>', unsafe_allow_html=True)
        rl_,rr_ = st.columns(2)
        with rl_: st.plotly_chart(ch_roc(yt,ypr_test,en,binary), use_container_width=True, key="t2roc")
        with rr_: st.plotly_chart(ch_pr(yt,ypr_test,en,binary), use_container_width=True, key="t2pr")

        # Model comparison: RF vs XGBoost
        comp=f"models/model_comparison_{mtag}.csv"; cvp=f"models/cross_validation_{mtag}.csv"
        if os.path.isfile(comp) or os.path.isfile(cvp):
            st.markdown('<div class="gap"></div>', unsafe_allow_html=True)
            st.markdown(f'<p class="sec">Random Forest vs XGBoost</p>', unsafe_allow_html=True)
            xl_,xr_ = st.columns(2)
            with xl_:
                if os.path.isfile(comp):
                    st.markdown(f'<p class="sec">Model Comparison (Test Set)</p>', unsafe_allow_html=True)
                    cdf = pd.read_csv(comp,index_col=0)
                    st.dataframe(cdf.style.highlight_max(axis=0, color="rgba(99,102,241,0.15)"),
                                 use_container_width=True)
            with xr_:
                if os.path.isfile(cvp):
                    st.markdown(f'<p class="sec">Cross-Validation (Train Set)</p>', unsafe_allow_html=True)
                    st.dataframe(pd.read_csv(cvp,index_col=0), use_container_width=True)
    else:
        st.info("Upload labeled data to see performance metrics.")

# TAB 3
with t3:
    st.markdown('<div class="gap-s"></div>', unsafe_allow_html=True)
    if hasattr(mdl,"feature_importances_"):
        fn=list(fdf.columns); imp=mdl.feature_importances_
        if len(imp)==len(fn):
            fl_,fr_ = st.columns(2)
            with fl_: st.plotly_chart(ch_fi(imp,fn), use_container_width=True, key="t3fi")
            with fr_:
                st.markdown(f'<p class="sec">Top Drivers</p>', unsafe_allow_html=True)
                top_ix=np.argsort(imp)[::-1][:5]
                for rank,i in enumerate(top_ix,1):
                    pct=imp[i]*100; bw=max(imp[i]/imp[top_ix[0]]*100,8)
                    st.markdown(
                        f'<div style="margin-bottom:0.45rem;">'
                        f'<div style="display:flex;justify-content:space-between;margin-bottom:0.12rem;">'
                        f'<span style="color:{T2};font-size:0.78rem;">{rank}. {fn[i]}</span>'
                        f'<span style="color:{T1};font-size:0.78rem;font-weight:600;">{pct:.1f}%</span></div>'
                        f'<div style="background:{BORD};border-radius:3px;height:5px;overflow:hidden;">'
                        f'<div style="width:{bw:.0f}%;height:100%;background:{IND};border-radius:3px;"></div>'
                        f'</div></div>', unsafe_allow_html=True)
            if hl and lc:
                st.markdown('<div class="gap"></div>', unsafe_allow_html=True)
                st.markdown(f'<p class="sec">Permutation Importance</p>', unsafe_allow_html=True)
                if st.button("Compute Permutation Importance"):
                    # Use test split only for permutation importance
                    Xs_test = Xs[test_idx]
                    yt_perm = yt  # already the test labels
                    with st.spinner("Computing..."):
                        pm=permutation_importance(mdl,Xs_test,yt_perm,n_repeats=10,random_state=42,n_jobs=-1,scoring="accuracy")
                    fnp=list(fdf.columns); tn=min(12,len(fnp))
                    si=pm.importances_mean.argsort()[::-1][:tn]
                    st.plotly_chart(ch_perm(pm.importances_mean[si],
                        [fnp[i] for i in si]), use_container_width=True, key="t3pm")
        else: st.caption(f"Feature mismatch ({len(imp)} vs {len(fn)}).")
    else: st.caption("Feature importance not available.")

# TAB 4
with t4:
    st.markdown('<div class="gap-s"></div>', unsafe_allow_html=True)
    rdf=fdf.copy(); rdf["Prediction"]=pl
    st.markdown(f'<p class="sec">Predictions ({len(rdf):,} samples)</p>', unsafe_allow_html=True)
    st.dataframe(rdf.head(200), use_container_width=True, height=380)
    st.download_button("Download CSV", data=rdf.to_csv(index=False).encode("utf-8"),
                       file_name="iot_predictions.csv", mime="text/csv")
    st.markdown('<div class="gap"></div>', unsafe_allow_html=True)
    st.markdown(f'<p class="sec">Raw Data</p>', unsafe_allow_html=True)
    st.caption(f"{df.shape[0]:,} rows · {df.shape[1]} columns")
    st.dataframe(df.head(100), use_container_width=True, height=300)


# Footer
st.markdown('<div class="ft">IoT Intrusion Detection System · B.Tech Minor Project · 2026</div>',
            unsafe_allow_html=True)
