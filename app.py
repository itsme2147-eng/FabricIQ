"""
app.py — FabricIQ Streamlit Web Application
=============================================
PhD Research Committee Demonstration
Integrated Fabric Quality Intelligence System

Deploy:  streamlit run app.py
Cloud:   https://streamlit.io/cloud  (free, GitHub-based)
"""

import streamlit as st
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from PIL import Image
import io, time

import os
from model_loader import (download_models, get_cached_models,
                          clear_cache, extract_gdrive_id, load_all_models)
from fabriciq_models import (
    extract_fft_features, preprocess_for_weave,
    detect_yarn_peaks, build_binary_matrix, compute_weave_features,
    classify_weave_grammar, detect_faults_classical,
    M01_METHODS, M01_BENCHMARK, CLASS_MU_SIGMA, DEMO_IMAGES,
    M02_METHODS, M02_BENCHMARK,
    load_deep_models, score_image_deep,
)

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="FabricIQ — Fabric Quality Intelligence",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
# THEME & CUSTOM CSS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

  /* Global */
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  .stApp { background: #0a0d14; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #111622 !important;
    border-right: 1px solid #1e2b45;
  }
  section[data-testid="stSidebar"] * { color: #94a3b8 !important; }
  section[data-testid="stSidebar"] h1,
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3 { color: #00d4ff !important; }

  /* Main content */
  .main .block-container { padding: 1.5rem 2rem; }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: #111622;
    border: 1px solid #1e2b45;
    border-radius: 10px;
    padding: 14px 18px;
    border-top: 2px solid #00d4ff;
  }
  [data-testid="metric-container"] label { color: #64748b !important; font-family: 'Space Mono', monospace !important; font-size: 0.65rem !important; letter-spacing: 0.1em; }
  [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #e2e8f0 !important; font-family: 'Space Mono', monospace !important; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { background: #111622; border-bottom: 1px solid #1e2b45; gap: 4px; }
  .stTabs [data-baseweb="tab"] { color: #64748b !important; font-family: 'Space Mono', monospace; font-size: 0.75rem; letter-spacing: 0.05em; padding: 10px 20px; }
  .stTabs [aria-selected="true"] { color: #00d4ff !important; border-bottom: 2px solid #00d4ff !important; }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #00d4ff22, #7c3aed22) !important;
    border: 1px solid #00d4ff66 !important;
    color: #00d4ff !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.05em !important;
    border-radius: 8px !important;
    transition: all 0.2s !important;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #00d4ff33, #7c3aed33) !important;
    border-color: #00d4ff !important;
  }

  /* Selectbox / multiselect */
  .stMultiSelect [data-baseweb="select"],
  .stSelectbox [data-baseweb="select"] {
    background: #111622 !important;
    border-color: #1e2b45 !important;
  }

  /* Info / success boxes */
  .stSuccess { background: #10b98122 !important; border-color: #10b981 !important; }
  .stWarning { background: #f59e0b22 !important; border-color: #f59e0b !important; }
  .stError   { background: #ef444422 !important; border-color: #ef4444 !important; }

  /* Header bar */
  .fabriciq-header {
    background: linear-gradient(135deg, #111622, #161d2e);
    border: 1px solid #1e2b45;
    border-left: 4px solid #00d4ff;
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 20px;
  }
  .fabriciq-header h1 {
    font-family: 'Space Mono', monospace;
    color: #00d4ff;
    font-size: 1.8rem;
    margin: 0;
  }
  .fabriciq-header p { color: #94a3b8; margin: 6px 0 0; font-size: 0.9rem; }

  /* Section banners */
  .module-banner {
    background: #111622;
    border-left: 3px solid #00d4ff;
    border-radius: 0 8px 8px 0;
    padding: 10px 16px;
    margin: 16px 0 12px;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #00d4ff;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }

  /* Result verdict */
  .verdict-pass   { color: #10b981; font-weight: 700; font-size: 1.4rem; font-family: 'Space Mono', monospace; }
  .verdict-review { color: #f59e0b; font-weight: 700; font-size: 1.4rem; font-family: 'Space Mono', monospace; }
  .verdict-fault  { color: #ef4444; font-weight: 700; font-size: 1.4rem; font-family: 'Space Mono', monospace; }

  /* Data table */
  .stDataFrame { background: #111622 !important; }
  [data-testid="stDataFrame"] { border: 1px solid #1e2b45; border-radius: 8px; }

  /* Divider */
  hr { border-color: #1e2b45 !important; }

  /* Code font */
  code, .stCode { font-family: 'Space Mono', monospace !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PLOTLY THEME
# ═══════════════════════════════════════════════════════════════
PLOTLY_LAYOUT = dict(
    paper_bgcolor='#0a0d14', plot_bgcolor='#111622',
    font=dict(family='Space Mono, monospace', color='#94a3b8', size=11),
)
PLOTLY_MARGIN = dict(l=40, r=20, t=40, b=40)   # applied separately to avoid duplicate-key
# Grid style applied separately via update_xaxes/update_yaxes to avoid duplicate-key error
AXIS_STYLE = dict(gridcolor='#1e2b45', linecolor='#1e2b45', zerolinecolor='#1e2b45')
ACCENT = ['#00d4ff', '#7c3aed', '#10b981', '#f59e0b', '#ef4444', '#a78bfa']

CMAP_FAULT = LinearSegmentedColormap.from_list('fault', ['#0a0d14', '#f59e0b', '#ef4444'])
WEAVE_COLORS = {'Plain Weave': '#00d4ff', '2/1 Twill': '#7c3aed', '3/1 Twill': '#10b981'}

def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """Convert #RRGGBB hex to rgba() string safe for Plotly fillcolor/marker_color."""
    h = hex_color.lstrip('#')
    if len(h) == 6:
        r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        return f'rgba({r},{g},{b},{alpha})'
    return hex_color  # fallback — return unchanged if unexpected format


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════
def load_image(source) -> np.ndarray:
    """PIL Image or bytes → 512×512 grayscale ndarray."""
    if isinstance(source, np.ndarray):
        img = source
    elif hasattr(source, 'read'):
        arr = np.frombuffer(source.read(), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    else:
        pil = Image.fromarray(source).convert('L')
        img = np.array(pil)
    img = cv2.resize(img, (512, 512))
    return img


def mpl_to_streamlit(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧵 FabricIQ")
    st.markdown("**Integrated Fabric Quality Intelligence**")
    st.markdown("---")

    st.markdown("### 📁 Image Source")
    img_source = st.radio("Image source", ["Upload Fabric Image", "Use Demo Image"], label_visibility="collapsed")

    img_gray = None

    if img_source == "Upload Fabric Image":
        uploaded = st.file_uploader("Choose image", type=['jpg','jpeg','png','bmp'],
                                     label_visibility="collapsed")
        if uploaded:
            pil_img = Image.open(uploaded).convert('L')
            img_gray = cv2.resize(np.array(pil_img), (512, 512))
            st.success(f"✅ {uploaded.name} loaded")
    else:
        demo_choice = st.selectbox("Demo fabric:", list(DEMO_IMAGES.keys()))
        img_gray = DEMO_IMAGES[demo_choice]()
        st.success(f"✅ Synthetic: {demo_choice}")

    st.markdown("---")
    st.markdown("### ⚙️ Module Settings")

    with st.expander("📊 Module 01 — Warp/Weft", expanded=True):
        m01_selected = st.multiselect(
            "Fusion methods to compare:",
            list(M01_METHODS.keys()),
            default=['Ratio-Consistency ⭐', 'Weighted Physical', 'Xu (1996) — FFT'],
            help="Hold Ctrl to select multiple. ⭐ = best recommended.")

    with st.expander("🧬 Module 02 — Weave", expanded=True):
        m02_selected = st.multiselect(
            "Classification methods:",
            list(M02_METHODS.keys()),
            default=list(M02_METHODS.keys()),
            help="Compare Grammar v5.1 against simpler baselines")
        st.caption("⭐ Grammar v5.1: mean_float primary (d'=1.86) → ≥75% accuracy")
        st.caption("Nearest-centroid & threshold shown for academic comparison")

    with st.expander("🔍 Module 03 — Fault Detection", expanded=False):
        m03_selected = st.multiselect(
            "Detectors:",
            ['Isolation Forest', 'Local Outlier Factor', 'One-Class SVM',
             'Classical Ensemble', 'Spatial Anomaly'],
            default=['Isolation Forest', 'Local Outlier Factor', 'One-Class SVM', 'Classical Ensemble'])
        fault_threshold = st.slider("Alert threshold", 0.20, 0.80, 0.50, 0.05,
                                     help="Ensemble score above this = FAULT")

    st.markdown("---")
    run_all = st.button("▶  Run Full Analysis", use_container_width=True)

    st.markdown("---")
    st.markdown("### 🤖 Deep Models (Module 03)")
    # Auto-load IDs from Streamlit secrets if configured
    _secrets_gdrive = {}
    try:
        _secrets_gdrive = dict(st.secrets.get("gdrive", {}))
    except Exception:
        pass

    with st.expander("📥 Load from Google Drive", expanded=False):
        st.markdown("""
<div style="font-family:monospace;font-size:11px;background:#111622;
border:1px solid #1e2b45;border-radius:6px;padding:10px;margin-bottom:8px">
<b style="color:#00d4ff">How to get File IDs:</b><br>
1. Open Google Drive → <code>output/train_models/</code><br>
2. Right-click any .pt file → <b>Share</b> → <b>Anyone with the link</b><br>
3. Copy the link — looks like:<br>
   <code>https://drive.google.com/file/d/<b>FILE_ID</b>/view</code><br>
4. Paste the full URL or just the FILE_ID below
</div>""", unsafe_allow_html=True)

        st.markdown("**Paste Google Drive share URLs or File IDs:**")
        col_l, col_r = st.columns(2)

        with col_l:
            fae_id     = st.text_input("fae_best.pt",                    key="id_fae",     value=_secrets_gdrive.get("fae_best", ""), placeholder="URL or File ID — or set in Streamlit Secrets")
            fae_mu_id  = st.text_input("fae_mu.npy",                     key="id_fae_mu",  value=_secrets_gdrive.get("fae_mu", ""), placeholder="URL or File ID — or set in Streamlit Secrets")
            fae_sig_id = st.text_input("fae_sig.npy",                    key="id_fae_sig", value=_secrets_gdrive.get("fae_sig", ""), placeholder="URL or File ID — or set in Streamlit Secrets")
            mob_r_id   = st.text_input("mobilenet_v3_small_recon.pt",    key="id_mob_r",   value=_secrets_gdrive.get("mobilenet_v3_small_recon", ""), placeholder="URL or File ID — or set in Streamlit Secrets")
        with col_r:
            mob_s_id   = st.text_input("mobilenet_v3_small_selfsup.pt",  key="id_mob_s",   value=_secrets_gdrive.get("mobilenet_v3_small_selfsup", ""), placeholder="URL or File ID — or set in Streamlit Secrets")
            eff_r_id   = st.text_input("efficientnet_b0_recon.pt",       key="id_eff_r",   value=_secrets_gdrive.get("efficientnet_b0_recon", ""), placeholder="URL or File ID — or set in Streamlit Secrets")
            eff_s_id   = st.text_input("efficientnet_b0_selfsup.pt",     key="id_eff_s",   value=_secrets_gdrive.get("efficientnet_b0_selfsup", ""), placeholder="URL or File ID — or set in Streamlit Secrets")

        # Build ID dict from inputs
        gdrive_ids = {
            'fae_best.pt':                   extract_gdrive_id(fae_id)    if fae_id    else '',
            'fae_mu.npy':                    extract_gdrive_id(fae_mu_id) if fae_mu_id else '',
            'fae_sig.npy':                   extract_gdrive_id(fae_sig_id)if fae_sig_id else '',
            'mobilenet_v3_small_recon.pt':   extract_gdrive_id(mob_r_id)  if mob_r_id  else '',
            'mobilenet_v3_small_selfsup.pt': extract_gdrive_id(mob_s_id)  if mob_s_id  else '',
            'efficientnet_b0_recon.pt':      extract_gdrive_id(eff_r_id)  if eff_r_id  else '',
            'efficientnet_b0_selfsup.pt':    extract_gdrive_id(eff_s_id)  if eff_s_id  else '',
        }
        configured = [k for k, v in gdrive_ids.items() if v and not v.startswith('YOUR')]

        # Show already cached
        cached = get_cached_models()
        if cached:
            st.success(f"✅ Cached: {', '.join(cached)}")

        col_dl, col_clr = st.columns(2)
        dl_btn  = col_dl.button("⬇️ Download Models", key="dl_btn",
                                 disabled=len(configured) == 0,
                                 use_container_width=True)
        clr_btn = col_clr.button("🗑️ Clear Cache", key="clr_btn",
                                  use_container_width=True)

        if clr_btn:
            clear_cache()
            if 'deep_models' in st.session_state:
                del st.session_state['deep_models']
            st.success("Cache cleared")

        if dl_btn and configured:
            progress_bar = st.progress(0, text="Starting downloads...")
            status_area  = st.empty()
            results_ok, results_err = [], []
            total = len(configured)

            def on_progress(fname, success, err):
                if success:
                    results_ok.append(fname)
                    status_area.success(f"✅ {fname}")
                else:
                    results_err.append(fname)
                    status_area.error(f"❌ {fname}: {err}")
                progress_bar.progress(
                    (len(results_ok) + len(results_err)) / total,
                    text=f"Downloaded {len(results_ok)}/{total}")

            with st.spinner("Downloading from Google Drive..."):
                dl_paths, dl_errors = download_models(
                    {k: v for k, v in gdrive_ids.items() if v},
                    progress_callback=on_progress)

            if dl_paths:
                with st.spinner("Loading model weights..."):
                    loaded = load_all_models(dl_paths)
                st.session_state['deep_models'] = loaded
                ok  = [k for k in loaded if not k.startswith('_') and 'err' not in k]
                st.success(f"✅ Loaded {len(ok)} deep model(s): {', '.join(ok)}")
                if dl_errors:
                    st.warning(f"⚠️ {len(dl_errors)} file(s) failed: {list(dl_errors.keys())}")
            elif dl_errors:
                st.error("All downloads failed. Check that files are shared as 'Anyone with the link'.")

        if 'deep_models' in st.session_state:
            ok = [k for k in st.session_state['deep_models']
                  if not k.startswith('_') and 'err' not in k]
            if ok:
                st.info(f"🤖 Active models: {', '.join(ok)}")

    st.markdown("---")
    st.markdown("### 📖 Research Context")
    st.caption("Dataset: 616 fabric images")
    st.caption("Best M01: Ratio-Consistency (MAE=3.48) ⭐")
    st.caption("Best M02: Grammar v5.1 (≥75% on 616 real scans)")
    st.caption("Best M03: Isolation Forest (Silhouette=0.543) ⭐")
    st.markdown("---")
    st.caption("FabricIQ · PhD Research Dashboard")
    st.caption("Streamlit · Open Source · No install")


# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="fabriciq-header">
  <h1>🧵 FabricIQ</h1>
  <p>Integrated Fabric Quality Intelligence System &nbsp;·&nbsp;
     PhD Research Demonstration &nbsp;·&nbsp;
     Three ML Modules: Thread Count · Weave Pattern · Fault Detection</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# GUARD: no image
# ═══════════════════════════════════════════════════════════════
# Auto-download models from secrets if not yet cached and secrets are configured
if 'deep_models' not in st.session_state:
    try:
        _sec = dict(st.secrets.get("gdrive", {}))
        _configured = {k: v for k, v in {
            'fae_best.pt':                   _sec.get('fae_best', ''),
            'fae_mu.npy':                    _sec.get('fae_mu', ''),
            'fae_sig.npy':                   _sec.get('fae_sig', ''),
            'mobilenet_v3_small_recon.pt':   _sec.get('mobilenet_v3_small_recon', ''),
            'mobilenet_v3_small_selfsup.pt': _sec.get('mobilenet_v3_small_selfsup', ''),
            'efficientnet_b0_recon.pt':      _sec.get('efficientnet_b0_recon', ''),
            'efficientnet_b0_selfsup.pt':    _sec.get('efficientnet_b0_selfsup', ''),
        }.items() if v and not v.startswith('PASTE')}
        _cached = get_cached_models()
        _need   = [k for k in _configured if k not in _cached]
        if _need:
            with st.sidebar:
                with st.spinner(f"Auto-downloading {len(_need)} model(s) from Drive..."):
                    _paths, _errs = download_models(
                        {k: v for k, v in _configured.items() if k in _need})
                if _paths:
                    _loaded = load_all_models(_paths)
                    st.session_state['deep_models'] = _loaded
        elif _cached and _configured:
            # All cached — just load
            _loaded = load_all_models({})
            if any(not k.startswith('_') for k in _loaded):
                st.session_state['deep_models'] = _loaded
    except Exception:
        pass  # Secrets not set — skip auto-download

if img_gray is None:
    st.info("👈 **Select an image source from the sidebar** to begin analysis.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Module 01 — Warp/Weft Count**")
        st.caption("6 fusion methods: FFT, Gabor, Wavelet integration. Benchmarked against Xu(1996) and Shi(2014).")
    with col2:
        st.markdown("**Module 02 — Weave Pattern**")
        st.caption("Probabilistic Grammar v5. Classifies Plain, 2/1 Twill, 3/1 Twill using structural features.")
    with col3:
        st.markdown("**Module 03 — Fault Detection**")
        st.caption("Isolation Forest + LOF + OC-SVM ensemble. Spatial anomaly heatmap with 8×8 patch grid.")
    st.stop()


# ═══════════════════════════════════════════════════════════════
# COMPUTE (cached per image)
# ═══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def run_pipeline(img_bytes: bytes, h: int = 512, w: int = 512):
    img = np.frombuffer(img_bytes, np.uint8).reshape(h, w)

    # Preprocessing
    t0 = time.time()
    enh   = preprocess_for_weave(img)
    feats = extract_fft_features(img)
    t_pre = time.time() - t0

    # M01 — all 6 methods
    t1 = time.time()
    ww = {}
    for m, fn in M01_METHODS.items():
        try: w, f, note = fn(feats); ww[m] = {'warp':w,'weft':f,'note':note}
        except: ww[m] = {'warp':0,'weft':0,'note':'error'}
    t_m01 = time.time() - t1

    # M02 — run all classifiers
    t2 = time.time()
    hp, vp, h_proj, v_proj = detect_yarn_peaks(enh)
    B  = build_binary_matrix(enh, hp, vp)
    wf = compute_weave_features(enh, B, hp, vp)
    pred, conf, probs = classify_weave_grammar(wf)   # primary result
    # All M02 method results
    m02_results = {}
    for m02_name, m02_fn in M02_METHODS.items():
        try:
            p, c, pr = m02_fn(wf)
            m02_results[m02_name] = {'pred': p, 'conf': c, 'probs': pr}
        except Exception:
            m02_results[m02_name] = {'pred': 'Error', 'conf': 0.0, 'probs': {}}
    t_m02 = time.time() - t2

    # M03
    t3 = time.time()
    scores, hmap, verdict = detect_faults_classical(img)
    t_m03 = time.time() - t3

    return dict(
        img_gray_raw=img_gray,  # stored for deep model scoring
        enh=enh, feats=feats,
        hp=hp, vp=vp, h_proj=h_proj, v_proj=v_proj, B=B, wf=wf,
        pred=pred, conf=conf, probs=probs,
        m02_results=m02_results,
        scores=scores, hmap=hmap, verdict=verdict,
        ww=ww,
        times=dict(pre=t_pre, m01=t_m01, m02=t_m02, m03=t_m03)
    )

with st.spinner("⚙️ Running FabricIQ analysis pipeline..."):
    result = run_pipeline(img_gray.tobytes(), img_gray.shape[0], img_gray.shape[1])

R = result


# ═══════════════════════════════════════════════════════════════
# TOP KPI ROW
# ═══════════════════════════════════════════════════════════════
cg_w = R['ww'].get('Confidence-Gated', {}).get('warp', 0)
cg_f = R['ww'].get('Confidence-Gated', {}).get('weft', 0)
ens_score = R['scores'].get('Ensemble', 0.5)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("🔵 Warp (CG)", f"{cg_w:.1f} px", help="Confidence-Gated fusion")
c2.metric("🟣 Weft (CG)", f"{cg_f:.1f} px", help="Confidence-Gated fusion")
c3.metric("🧬 Weave Type", R['pred'], help=f"Grammar v5 confidence: {R['conf']:.0%}")
c4.metric("📊 Weave Conf", f"{R['conf']:.0%}", delta="Grammar v5")
verdict_emoji = "✅ PASS" if R['verdict']=='PASS' else "⚠️ REVIEW" if R['verdict']=='REVIEW' else "❌ FAULT"
c5.metric("🔍 QC Decision", verdict_emoji, delta=f"Ensemble={ens_score:.3f}")


# ═══════════════════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════════════
tabs = st.tabs([
    "🖼️ Image Preview",
    "📊 Module 01 — Warp/Weft",
    "🧬 Module 02 — Weave Pattern",
    "🔍 Module 03 — Fault Detection",
    "📋 Integrated Report",
    "📐 Method Comparison",
])

# ───────────────────────────────────────────────────────────────
# TAB 0: IMAGE PREVIEW
# ───────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown('<div class="module-banner">📁 Input Image · Preprocessing · Yarn Detection</div>', unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)

    fig_orig, ax_orig = plt.subplots(figsize=(5,5), facecolor='#0a0d14')
    ax_orig.set_facecolor('#111622')
    ax_orig.imshow(img_gray, cmap='gray', aspect='auto')
    ax_orig.axis('off'); ax_orig.set_title('Original Grayscale', color='#e2e8f0', fontsize=10)
    col_a.pyplot(fig_orig, use_container_width=True); plt.close(fig_orig)

    fig_enh, ax_enh = plt.subplots(figsize=(5,5), facecolor='#0a0d14')
    ax_enh.set_facecolor('#111622')
    ax_enh.imshow(R['enh'], cmap='gray', aspect='auto')
    ax_enh.axis('off'); ax_enh.set_title('Enhanced (Homomorphic + Bilateral + CLAHE)', color='#e2e8f0', fontsize=10)
    col_b.pyplot(fig_enh, use_container_width=True); plt.close(fig_enh)

    fig_yarn, ax_yarn = plt.subplots(figsize=(5,5), facecolor='#0a0d14')
    ax_yarn.set_facecolor('#111622')
    ax_yarn.imshow(R['enh'], cmap='gray', aspect='auto')
    for yp in R['hp'][:40]: ax_yarn.axhline(y=yp, color='#00d4ff', alpha=0.5, linewidth=0.8)
    for xp in R['vp'][:40]: ax_yarn.axvline(x=xp, color='#10b981', alpha=0.5, linewidth=0.8)
    ax_yarn.axis('off')
    ax_yarn.set_title(f'Yarn Detection  Weft:{len(R["hp"])} | Warp:{len(R["vp"])}', color='#e2e8f0', fontsize=10)
    col_c.pyplot(fig_yarn, use_container_width=True); plt.close(fig_yarn)

    st.divider()
    # FFT spectrum
    col_d, col_e = st.columns(2)
    with col_d:
        st.markdown("**2D FFT Power Spectrum** — thread periodicity")
        mag = np.log(np.abs(np.fft.fftshift(np.fft.fft2(img_gray.astype(float)))) + 1)
        fig_fft = px.imshow(mag, color_continuous_scale='plasma',
                             labels=dict(color='log|FFT|'))
        fig_fft.update_layout(**PLOTLY_LAYOUT, height=320, margin=PLOTLY_MARGIN,
                               coloraxis_showscale=False, title='')
        fig_fft.update_xaxes(showticklabels=False, **AXIS_STYLE)
        fig_fft.update_yaxes(showticklabels=False, **AXIS_STYLE)
        st.plotly_chart(fig_fft, use_container_width=True)

    with col_e:
        st.markdown("**Yarn Projection Profiles** — weft (blue) & warp (green)")
        fig_proj = go.Figure()
        fig_proj.add_trace(go.Scatter(y=R['h_proj'], name='Weft projection (rows)',
                                       line=dict(color='#00d4ff', width=1.5)))
        fig_proj.add_trace(go.Scatter(y=R['v_proj'], name='Warp projection (cols)',
                                       line=dict(color='#10b981', width=1.5)))
        for yp in R['hp'][:15]:
            fig_proj.add_vline(x=yp, line_color='#00d4ff', line_width=0.5, opacity=0.4)
        fig_proj.update_layout(**PLOTLY_LAYOUT, height=320, margin=PLOTLY_MARGIN, showlegend=True,
                                legend=dict(bgcolor='#111622', bordercolor='#1e2b45'))
        fig_proj.update_xaxes(**AXIS_STYLE)
        fig_proj.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig_proj, use_container_width=True)

    st.info(f"⏱️ Preprocessing: {R['times']['pre']*1000:.0f}ms  |  "
            f"M01: {R['times']['m01']*1000:.0f}ms  |  "
            f"M02: {R['times']['m02']*1000:.0f}ms  |  "
            f"M03: {R['times']['m03']*1000:.0f}ms")


# ───────────────────────────────────────────────────────────────
# TAB 1: MODULE 01 — WARP/WEFT
# ───────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<div class="module-banner">📊 Module 01 — Warp/Weft Thread Count · 6 Fusion Methods</div>', unsafe_allow_html=True)

    sel_results = {m: R['ww'][m] for m in m01_selected if m in R['ww']}

    if not sel_results:
        st.warning("Select at least one method in the sidebar.")
    else:
        # Grouped bar chart
        methods = list(sel_results.keys())
        warps   = [sel_results[m]['warp'] for m in methods]
        wefts   = [sel_results[m]['weft'] for m in methods]
        short   = [m.split('—')[0].strip()[:18] for m in methods]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(name='Warp', x=short, y=warps,
                                  marker_color='#00d4ff', opacity=0.85))
        fig_bar.add_trace(go.Bar(name='Weft', x=short, y=wefts,
                                  marker_color='#7c3aed', opacity=0.85))
        fig_bar.update_layout(**PLOTLY_LAYOUT, barmode='group', height=380, margin=PLOTLY_MARGIN,
                               title='Thread Spacing (px) — Method Comparison',
                               legend=dict(bgcolor='#111622', bordercolor='#1e2b45'))
        fig_bar.update_xaxes(**AXIS_STYLE)
        fig_bar.update_yaxes(title='Thread spacing (px)', **AXIS_STYLE)
        st.plotly_chart(fig_bar, use_container_width=True)

        # Benchmark note
        st.info(
            "**⭐ Ratio-Consistency = Best method** confirmed by your 616-image study "
            "(warp_MAE=3.48 vs Xu FFT MAE=13.52). "
            "Shi(2014) Wavelet overestimates ×3 (mean 61.8px vs actual ~22px).")
        # Results table
        st.markdown("**Method Results Table**")
        rows = []
        for m in methods:
            r = sel_results[m]
            ratio = r['warp'] / (r['weft']+1e-8)
            rows.append({
                'Method': m,
                'Warp (px)': f"{r['warp']:.1f}",
                'Weft (px)': f"{r['weft']:.1f}",
                'W/F Ratio': f"{ratio:.3f}",
                'Source/Note': r['note'],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.divider()
        col_x, col_y = st.columns(2)

        with col_x:
            # Scatter: warp vs weft across methods
            st.markdown("**Warp vs Weft — Method Scatter**")
            fig_sc = go.Figure()
            for i, m in enumerate(methods):
                fig_sc.add_trace(go.Scatter(
                    x=[sel_results[m]['warp']], y=[sel_results[m]['weft']],
                    mode='markers+text', name=m[:18],
                    marker=dict(size=16, color=ACCENT[i % len(ACCENT)]),
                    text=[m.split('—')[0].strip()[:12]], textposition='top center',
                    textfont=dict(size=9, color='#e2e8f0')))
                fig_sc.add_shape(type='line', x0=0,y0=0,x1=150,y1=150,
                                  line=dict(color='#1e2b45',dash='dash',width=1))
            fig_sc.update_layout(**PLOTLY_LAYOUT, height=340, margin=PLOTLY_MARGIN, showlegend=False)
            fig_sc.update_xaxes(title='Warp spacing (px)', **AXIS_STYLE)
            fig_sc.update_yaxes(title='Weft spacing (px)', **AXIS_STYLE)
            st.plotly_chart(fig_sc, use_container_width=True)

        with col_y:
            # Feature breakdown
            st.markdown("**Feature Signal Strengths**")
            f = R['feats']
            feat_names = ['FFT Peak', 'Gabor E0°', 'Gabor E45°', 'Gabor E90°', 'Gabor E135°', 'Wavelet E']
            feat_vals  = [
                f['fft_peak_value'],
                f['gabor_orient_0_energy'],
                f['gabor_orient_45_energy'],
                f['gabor_orient_90_energy'],
                f['gabor_orient_135_energy'],
                f['wavelet_total_energy'],
            ]
            fmax = max(feat_vals) + 1e-8
            feat_norm = [v/fmax for v in feat_vals]
            fig_feat = go.Figure(go.Bar(
                x=feat_norm, y=feat_names, orientation='h',
                marker=dict(color=ACCENT[:len(feat_names)], opacity=0.85)))
            fig_feat.update_layout(**PLOTLY_LAYOUT, height=340, margin=PLOTLY_MARGIN, title='Feature Energy Breakdown')
            fig_feat.update_xaxes(title='Normalised energy', **AXIS_STYLE)
            fig_feat.update_yaxes(**AXIS_STYLE)
            st.plotly_chart(fig_feat, use_container_width=True)


# ───────────────────────────────────────────────────────────────
# TAB 2: MODULE 02 — WEAVE PATTERN
# ───────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown('<div class="module-banner">🧬 Module 02 — Weave Pattern Recognition · Grammar v5</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.info("**Grammar v5.1 (fixed)** — Stage 1 primary: `mean_float` (d\'=1.86). "
            "Confirmed ≥75% accuracy on your 616 real fabric scans. "
            "Lower confidence on synthetic demo images is expected.")
    # Classification result
        pred_color = WEAVE_COLORS.get(R['pred'], '#f59e0b')
        conf_color = '#10b981' if R['conf'] >= 0.70 else '#f59e0b' if R['conf'] >= 0.50 else '#ef4444'
        st.markdown(f"""
        <div style="background:#111622;border:1px solid #1e2b45;border-radius:12px;
                    padding:24px;text-align:center;margin-bottom:16px">
          <div style="font-family:Space Mono,monospace;font-size:1.5rem;
                      font-weight:700;color:{pred_color}">{R['pred']}</div>
          <div style="font-family:Space Mono,monospace;font-size:1.1rem;
                      color:{conf_color};margin-top:8px">{R['conf']:.0%} confidence</div>
          <div style="color:#64748b;font-size:0.8rem;margin-top:8px">
              Probabilistic Grammar v5 · 2-stage log-likelihood</div>
        </div>
        """, unsafe_allow_html=True)

        # Probability bars
        probs_df = pd.DataFrame({
            'Class': list(R['probs'].keys()),
            'Probability': list(R['probs'].values()),
        })
        fig_prob = go.Figure(go.Bar(
            x=probs_df['Probability'], y=probs_df['Class'],
            orientation='h',
            marker_color=[WEAVE_COLORS.get(c,'#f59e0b') for c in probs_df['Class']],
            opacity=0.85, text=[f"{v:.1%}" for v in probs_df['Probability']],
            textposition='outside', textfont=dict(color='#e2e8f0', size=11)))
        fig_prob.add_vline(x=0.5, line_color='#ef4444', line_dash='dash', line_width=1.5)
        fig_prob.update_layout(**PLOTLY_LAYOUT, height=240, margin=PLOTLY_MARGIN,
                                showlegend=False, title='Class Posterior Probabilities')
        fig_prob.update_xaxes(range=[0, 1.15], title='Posterior probability', **AXIS_STYLE)
        fig_prob.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig_prob, use_container_width=True)

    with col_b:
        # Structural features radar
        wf = R['wf']
        radar_features = ['YIDF', 'CSI', 'Coherence', 'Mean Float (norm)', 'Local Var (norm)']
        radar_values   = [
            float(np.clip(wf.get('yidf',0.5), 0, 1)),
            float(np.clip(wf.get('csi',0.35), 0, 1)),
            float(np.clip(wf.get('coherence',0.3), 0, 1)),
            float(np.clip(wf.get('mean_float',2)/6, 0, 1)),
            float(np.clip(wf.get('local_var',0.03)*20, 0, 1)),
        ]
        # Add reference profiles
        plain_ref  = [CLASS_MU_SIGMA['Plain Weave'][k][0]/norm
                      for k,norm in [('yidf',1),('csi',1),('coher',1),('mf',6),('lv',0.05)]]
        twill21_ref= [CLASS_MU_SIGMA['2/1 Twill'][k][0]/norm
                      for k,norm in [('yidf',1),('csi',1),('coher',1),('mf',6),('lv',0.05)]]
        twill31_ref= [CLASS_MU_SIGMA['3/1 Twill'][k][0]/norm
                      for k,norm in [('yidf',1),('csi',1),('coher',1),('mf',6),('lv',0.05)]]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=plain_ref+[plain_ref[0]], theta=radar_features+[radar_features[0]],
            name='Plain (ref)', line=dict(color='#00d4ff', dash='dot', width=1.5), opacity=0.5))
        fig_radar.add_trace(go.Scatterpolar(
            r=twill21_ref+[twill21_ref[0]], theta=radar_features+[radar_features[0]],
            name='2/1T (ref)', line=dict(color='#7c3aed', dash='dot', width=1.5), opacity=0.5))
        fig_radar.add_trace(go.Scatterpolar(
            r=twill31_ref+[twill31_ref[0]], theta=radar_features+[radar_features[0]],
            name='3/1T (ref)', line=dict(color='#10b981', dash='dot', width=1.5), opacity=0.5))
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_values+[radar_values[0]], theta=radar_features+[radar_features[0]],
            name='This image', fill='toself',
            line=dict(color=WEAVE_COLORS.get(R['pred'],'#f59e0b'), width=2.5),
            fillcolor=hex_to_rgba(WEAVE_COLORS.get(R['pred'],'#f59e0b'), 0.20)))
        fig_radar.update_layout(
            paper_bgcolor='#0a0d14', plot_bgcolor='#0a0d14',
            polar=dict(bgcolor='#111622',
                        radialaxis=dict(visible=True, range=[0,1], gridcolor='#1e2b45', color='#64748b'),
                        angularaxis=dict(gridcolor='#1e2b45', color='#94a3b8')),
            legend=dict(bgcolor='#111622', bordercolor='#1e2b45', font=dict(color='#94a3b8')),
            height=360, margin=dict(l=40,r=40,t=40,b=40), showlegend=True,
            title=dict(text='Structural Feature Profile vs Class References', font=dict(color='#94a3b8', size=11)))
        st.plotly_chart(fig_radar, use_container_width=True)

    # Multi-method comparison table
    st.divider()
    st.markdown("**📊 Weave Classification Method Comparison**")
    m02_comp_rows = []
    for m_name, m_res in R.get('m02_results', {}).items():
        bench = M02_BENCHMARK.get(m_name, {})
        m02_comp_rows.append({
            'Method': m_name,
            'Prediction': m_res.get('pred','—'),
            'Confidence': f"{m_res.get('conf',0):.0%}",
            'Plain %': f"{m_res.get('probs',{}).get('Plain Weave',0):.0%}",
            '2/1T %':  f"{m_res.get('probs',{}).get('2/1 Twill',0):.0%}",
            '3/1T %':  f"{m_res.get('probs',{}).get('3/1 Twill',0):.0%}",
            'Accuracy': bench.get('accuracy','—'),
        })
    if m02_comp_rows:
        import pandas as _pd
        st.dataframe(_pd.DataFrame(m02_comp_rows), use_container_width=True, hide_index=True)

    st.divider()
    # Interlacement matrix
    B = R['B']
    col_m, col_f = st.columns(2)
    with col_m:
        st.markdown("**Binary Interlacement Matrix B[i,j]**")
        st.caption("Green = Warp on top · Red = Weft on top")
        if B is not None and B.size > 4:
            disp = B[:min(20,B.shape[0]),:min(20,B.shape[1])]
            fig_mat = px.imshow(disp, color_continuous_scale=['#ef4444','#10b981'],
                                 zmin=0, zmax=1, aspect='auto')
            fig_mat.update_layout(**PLOTLY_LAYOUT, height=280, margin=PLOTLY_MARGIN,
                                   coloraxis_showscale=False)
            fig_mat.update_xaxes(title='Warp thread index', **AXIS_STYLE)
            fig_mat.update_yaxes(title='Weft thread index', **AXIS_STYLE)
            st.plotly_chart(fig_mat, use_container_width=True)
        else:
            st.warning("Insufficient yarn peaks detected for interlacement matrix.")

    with col_f:
        st.markdown("**Structural Feature Values**")
        feat_table = {
            'Feature': ['YIDF','CSI','CSI Col','Coherence','Mean Float','Max Float','Local Var','Spacing Ratio','Warp Peaks','Weft Peaks'],
            'Value':   [f"{wf.get(k,0):.4f}" for k in ['yidf','csi','csi_col','coherence','mean_float','max_float','local_var','spacing_ratio','n_warp','n_weft']],
            'Class Mean (pred)': [
                f"{CLASS_MU_SIGMA[R['pred']].get(k2,(0,0))[0]:.4f}"
                for k2 in ['yidf','csi','csi','coher','mf','mf','lv','csi','yidf','yidf']],
        }
        st.dataframe(pd.DataFrame(feat_table), use_container_width=True, hide_index=True, height=320)


# ───────────────────────────────────────────────────────────────
# TAB 3: MODULE 03 — FAULT DETECTION
# ───────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown('<div class="module-banner">🔍 Module 03 — Fault Detection · Classical Anomaly Methods</div>', unsafe_allow_html=True)

    ens = R['scores'].get('Ensemble', 0.5)
    verdict = R['verdict']
    verdict_class = f"verdict-{verdict.lower()}"
    verdict_emoji_map = {'PASS':'✅','REVIEW':'⚠️','FAULT':'❌'}

    # Deep model scores if available
    if 'deep_models' in st.session_state:
        deep_models = st.session_state['deep_models']
        ok_models = [k for k in deep_models if not k.startswith('_') and 'err' not in k]
        if ok_models:
            st.markdown("**🤖 Deep Model Scores (your trained models)**")
            dcols = st.columns(len(ok_models))
            for col, key in zip(dcols, ok_models):
                score = score_image_deep(R['img_gray_raw'], deep_models[key], key)
                label = 'PASS' if score < 0.35 else 'REVIEW' if score < 0.55 else 'FAULT'
                col.metric(key[:20], f"{score:.3f}", delta=label)
            st.divider()

    # Verdict banner
    col_v1, col_v2, col_v3 = st.columns(3)
    col_v1.markdown(f"""
    <div style="background:#111622;border:1px solid #1e2b45;border-radius:10px;
                padding:20px;text-align:center">
      <div class="{verdict_class}">{verdict_emoji_map[verdict]} {verdict}</div>
      <div style="color:#64748b;font-size:0.8rem;margin-top:6px">QC Decision</div>
    </div>""", unsafe_allow_html=True)
    col_v2.metric("Ensemble Score", f"{ens:.3f}",
                   delta=f"Threshold: {fault_threshold:.2f}",
                   delta_color="inverse" if ens > fault_threshold else "normal")
    col_v3.metric("Flagged patches", f"{int(np.sum(R['hmap'] > 0.6) / R['hmap'].size * 100):.0f}%",
                   help="% of 8×8 spatial patches above anomaly threshold")

    st.divider()
    col_s, col_h = st.columns(2)

    with col_s:
        # Score bars
        st.markdown("**Anomaly Scores — All Detectors**")
        all_methods = list(R['scores'].keys())
        all_scores  = [R['scores'][m] for m in all_methods]
        bar_colors  = ['#10b981' if s < 0.35 else '#f59e0b' if s < 0.55 else '#ef4444'
                        for s in all_scores]
        fig_scores = go.Figure(go.Bar(
            x=all_methods, y=all_scores,
            marker_color=bar_colors, opacity=0.88,
            text=[f"{s:.3f}" for s in all_scores],
            textposition='outside', textfont=dict(color='#e2e8f0', size=11)))
        fig_scores.add_hline(y=fault_threshold, line_color='#ef4444',
                              line_dash='dash', line_width=2,
                              annotation_text=f"Alert threshold ({fault_threshold:.2f})",
                              annotation_font_color='#ef4444')
        fig_scores.update_layout(**PLOTLY_LAYOUT, height=340, margin=PLOTLY_MARGIN, showlegend=False)
        fig_scores.update_xaxes(**AXIS_STYLE)
        fig_scores.update_yaxes(range=[0, 1.1], title='Anomaly score', **AXIS_STYLE)
        st.plotly_chart(fig_scores, use_container_width=True)

        # Benchmark reference
        st.markdown("**Your Module 03 Benchmark (616-image dataset)**")
        bench_df = pd.DataFrame({
            'Method': ['Isolation Forest ⭐','Local Outlier Factor','One-Class SVM','Classical Ensemble',
                       'CAE (Feature-Space)','PatchCore','MobileNet Recon','EfficientNet Recon',
                       'MobileNet SelfSup','EfficientNet SelfSup'],
            'Silhouette ↑': [0.5426,0.4541,0.4515,0.5019,0.4326,0.3032,0.2573,0.2636,0.0032,0.2328],
            'Davies-Bouldin ↓': [1.1939,2.3187,2.1577,1.9627,1.7368,1.9893,0.7766,0.7585,4.8344,1.6087],
            'CPU (img/s)': ['3.5','—','—','—','38.0','—','49.3','22.0','148.7','31.7'],
        })
        st.dataframe(bench_df.style.background_gradient(subset=['Silhouette ↑'], cmap='Blues'),
                     use_container_width=True, hide_index=True)

    with col_h:
        # Spatial heatmap
        st.markdown("**Spatial Anomaly Heatmap**")
        fig_hmap = make_subplots(rows=1, cols=1)
        fig_hmap.add_trace(go.Heatmap(
            z=cv2.resize(R['hmap'], (128, 128)),
            colorscale=[[0,'#0a0d14'],[0.4,'#f59e0b'],[1,'#ef4444']],
            showscale=True, colorbar=dict(title='Score',tickfont=dict(color='#94a3b8'))))
        fig_hmap.update_layout(**PLOTLY_LAYOUT, height=340, margin=PLOTLY_MARGIN,
                                title='Anomaly Intensity (8×8 → upsampled)')
        fig_hmap.update_xaxes(title='Image column →', **AXIS_STYLE)
        fig_hmap.update_yaxes(title='Image row ↓', **AXIS_STYLE)
        st.plotly_chart(fig_hmap, use_container_width=True)

        # Overlay
        st.markdown("**Heatmap Overlay on Fabric**")
        overlay_gray = img_gray.astype(float) / 255.0
        hmap_rgb = np.zeros((*R['hmap'].shape, 3))
        hmap_rgb[:,:,0] = R['hmap']
        hmap_rgb[:,:,1] = R['hmap'] * 0.39
        hmap_rgb[:,:,2] = 0
        alpha = R['hmap'] * 0.65
        overlay = np.stack([overlay_gray]*3, axis=-1)
        for c in range(3):
            overlay[:,:,c] = (1-alpha)*overlay[:,:,c] + alpha*hmap_rgb[:,:,c]
        overlay = np.clip(overlay, 0, 1)
        fig_ov = px.imshow(overlay, aspect='auto')
        fig_ov.update_layout(**PLOTLY_LAYOUT, height=220, margin=dict(l=0,r=0,t=0,b=0))
        fig_ov.update_xaxes(showticklabels=False, **AXIS_STYLE)
        fig_ov.update_yaxes(showticklabels=False, **AXIS_STYLE)
        st.plotly_chart(fig_ov, use_container_width=True)


# ───────────────────────────────────────────────────────────────
# TAB 4: INTEGRATED REPORT
# ───────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown('<div class="module-banner">📋 Integrated QC Report — All Modules Combined</div>', unsafe_allow_html=True)

    # Header row
    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        st.markdown("#### 🧵 FabricIQ Report")
        st.caption("Integrated Fabric Quality Analysis")
    with col_r2:
        pc = WEAVE_COLORS.get(R['pred'], '#f59e0b')
        st.markdown(f"**Weave Type:** <span style='color:{pc}'>{R['pred']}</span> ({R['conf']:.0%})", unsafe_allow_html=True)
        st.markdown(f"**Warp/Weft:** {cg_w:.1f}px / {cg_f:.1f}px (Confidence-Gated)")
    with col_r3:
        vc = {'PASS':'#10b981','REVIEW':'#f59e0b','FAULT':'#ef4444'}.get(R['verdict'],'#64748b')
        st.markdown(f"**QC Verdict:** <span style='color:{vc};font-weight:700'>{verdict_emoji_map[verdict]} {R['verdict']}</span>", unsafe_allow_html=True)
        st.markdown(f"**Ensemble Score:** {ens:.3f} (threshold={fault_threshold:.2f})")

    st.divider()

    # Three-module summary in one figure
    fig_rep = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            'Fabric Image', 'Warp/Weft: Method Comparison', 'FFT Spectrum',
            'Weave Classification', 'Fault Anomaly Scores', 'Spatial Heatmap'],
        specs=[[{'type':'xy'},{'type':'xy'},{'type':'xy'}],
               [{'type':'xy'},{'type':'xy'},{'type':'xy'}]])

    # [0,0] Fabric image
    fig_rep.add_trace(go.Heatmap(z=img_gray[::-1], colorscale='gray',
                                  showscale=False), row=1, col=1)

    # [0,1] Warp/weft bars (top 4 methods)
    top4 = list(R['ww'].keys())[:4]
    fig_rep.add_trace(go.Bar(name='Warp', x=[m.split('—')[0].strip()[:12] for m in top4],
                              y=[R['ww'][m]['warp'] for m in top4],
                              marker_color='#00d4ff', opacity=0.85, showlegend=False), row=1, col=2)
    fig_rep.add_trace(go.Bar(name='Weft', x=[m.split('—')[0].strip()[:12] for m in top4],
                              y=[R['ww'][m]['weft'] for m in top4],
                              marker_color='#7c3aed', opacity=0.85, showlegend=False), row=1, col=2)

    # [0,2] FFT
    mag2 = np.log(np.abs(np.fft.fftshift(np.fft.fft2(img_gray.astype(float))))+1)
    fig_rep.add_trace(go.Heatmap(z=mag2, colorscale='plasma', showscale=False), row=1, col=3)

    # [1,0] Weave probs
    fig_rep.add_trace(go.Bar(
        x=list(R['probs'].values()), y=list(R['probs'].keys()), orientation='h',
        marker_color=[WEAVE_COLORS.get(c,'#f59e0b') for c in R['probs'].keys()],
        opacity=0.85, showlegend=False), row=2, col=1)

    # [1,1] Fault scores
    fig_rep.add_trace(go.Bar(
        x=list(R['scores'].keys()), y=list(R['scores'].values()),
        marker_color=['#10b981' if v<0.35 else '#f59e0b' if v<0.55 else '#ef4444'
                      for v in R['scores'].values()],
        opacity=0.85, showlegend=False), row=2, col=2)

    # [1,2] Heatmap
    fig_rep.add_trace(go.Heatmap(
        z=cv2.resize(R['hmap'], (64,64)),
        colorscale=[[0,'#0a0d14'],[0.5,'#f59e0b'],[1,'#ef4444']],
        showscale=False), row=2, col=3)

    fig_rep.update_layout(
        paper_bgcolor='#0a0d14', plot_bgcolor='#111622',
        font=dict(family='Space Mono,monospace', color='#94a3b8', size=9),
        height=580, showlegend=False, barmode='group',
        margin=dict(l=30,r=20,t=50,b=30))
    for ann in fig_rep['layout']['annotations']:
        ann['font'] = dict(color='#94a3b8', size=10, family='Space Mono,monospace')
    st.plotly_chart(fig_rep, use_container_width=True)

    # Summary table
    st.markdown("**Integrated QC Summary Table**")
    summary = pd.DataFrame([
        {'Module':'01 — Warp/Weft','Method':'Confidence-Gated Integration ⭐',
         'Result':f"W:{cg_w:.1f}px  F:{cg_f:.1f}px",'Metric':'Multi-method fusion','Status':'✅ Complete'},
        {'Module':'02 — Weave','Method':'Probabilistic Grammar v5',
         'Result':R['pred'],'Metric':f"{R['conf']:.0%} confidence",
         'Status':'✅ Done' if R['conf']>0.6 else '⚠️ Low conf'},
        {'Module':'03 — Fault','Method':'Classical Ensemble (IF+LOF+SVM)',
         'Result':R['verdict'],'Metric':f"Score={ens:.3f}",
         'Status':'✅ PASS' if R['verdict']=='PASS' else '⚠️ REVIEW' if R['verdict']=='REVIEW' else '❌ FAULT'},
    ])
    st.dataframe(summary, use_container_width=True, hide_index=True)

    # Export button
    st.divider()
    if st.button("📥 Download Report as CSV"):
        csv = summary.to_csv(index=False)
        st.download_button("Download CSV", csv, "fabriciq_report.csv", "text/csv")


# ───────────────────────────────────────────────────────────────
# TAB 5: METHOD COMPARISON
# ───────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown('<div class="module-banner">📐 Method Comparison · Academic Benchmark Tables</div>', unsafe_allow_html=True)

    st.markdown("#### Module 01 — All 6 Fusion Methods on Current Image")
    comp_rows = []
    for m, r in R['ww'].items():
        ratio = r['warp'] / (r['weft']+1e-8)
        comp_rows.append({
            'Method': m, 'Warp (px)': round(r['warp'],2),
            'Weft (px)': round(r['weft'],2), 'W/F Ratio': round(ratio,3),
            'Source': r['note'],
            'Recommended': '⭐ Best' if '⭐' in m else '⚠️ Overestimates' if 'Shi' in m else '—'
        })
    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    st.divider()
    col_ca, col_cb = st.columns(2)

    with col_ca:
        st.markdown("#### Module 02 — Grammar v5 vs Dataset Statistics")
        this_wf = R['wf']
        cmp_rows = []
        for cls, params in CLASS_MU_SIGMA.items():
            marker = ' ← predicted' if cls == R['pred'] else ''
            cmp_rows.append({
                'Class': cls + marker,
                'YIDF mean': f"{params['yidf'][0]:.3f}±{params['yidf'][1]:.3f}",
                'CSI mean':  f"{params['csi'][0]:.3f}±{params['csi'][1]:.3f}",
                'Coher mean':f"{params['coher'][0]:.3f}±{params['coher'][1]:.3f}",
                'Float mean':f"{params['mf'][0]:.3f}±{params['mf'][1]:.3f}",
            })
        cmp_rows.append({
            'Class': '→ THIS IMAGE',
            'YIDF mean': f"{this_wf.get('yidf',0):.3f}",
            'CSI mean':  f"{this_wf.get('csi',0):.3f}",
            'Coher mean':f"{this_wf.get('coherence',0):.3f}",
            'Float mean':f"{this_wf.get('mean_float',0):.3f}",
        })
        st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)

    with col_cb:
        st.markdown("#### Module 03 — Benchmark (your 616-image study)")
        bench2 = pd.DataFrame({
            'Model': ['Classical IF ⭐','Classical LOF','Classical SVM','Classical Ensemble',
                      'CAE (FAE)','PatchCore','MobileNet Recon','MobileNet SelfSup'],
            'Silhouette ↑': [0.5426,0.4541,0.4515,0.5019,0.4326,0.3032,0.2573,0.0032],
            'DB Index ↓':   [1.194,2.319,2.158,1.963,1.737,1.989,0.777,4.834],
            'CPU (img/s)':  [3.5,None,None,None,38.0,None,49.3,148.7],
            'Params (M)':   [0,0,0,0,0.32,None,3.95,0.93],
        })
        st.dataframe(bench2, use_container_width=True, hide_index=True)

        fig_bench = go.Figure()
        has_speed = bench2['CPU (img/s)'].notna()
        fig_bench.add_trace(go.Scatter(
            x=bench2.loc[has_speed,'CPU (img/s)'],
            y=bench2.loc[has_speed,'Silhouette ↑'],
            mode='markers+text',
            text=bench2.loc[has_speed,'Model'].str[:12],
            textposition='top center',
            textfont=dict(size=8, color='#e2e8f0'),
            marker=dict(size=14, color=ACCENT[:has_speed.sum()], opacity=0.85)))
        fig_bench.update_layout(**PLOTLY_LAYOUT, height=280, margin=PLOTLY_MARGIN,
                                 title='Speed vs Quality Trade-off', showlegend=False)
        fig_bench.update_xaxes(title='CPU throughput (img/s)', **AXIS_STYLE)
        fig_bench.update_yaxes(title='Silhouette score', **AXIS_STYLE)
        st.plotly_chart(fig_bench, use_container_width=True)
