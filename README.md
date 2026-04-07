# 🧵 FabricIQ — Integrated Fabric Quality Intelligence  v2 (fixed)

**PhD Research Demonstration Dashboard · Streamlit Web App**

---

## 🐛 Bugs Fixed in v2

| # | Issue | Fix |
|---|-------|-----|
| 1 | `TypeError: update_layout() got multiple values for keyword argument 'xaxis'` | Removed `xaxis`/`yaxis` from `PLOTLY_LAYOUT` dict; applied via `update_xaxes()`/`update_yaxes()` |
| 2 | Module 01 default method was Xu(1996) FFT | **Ratio-Consistency** set as default — confirmed best by your study (MAE=3.48) |
| 3 | Module 02 Plain/Twill mismatch | Rebuilt Stage 1 grammar with `mean_float` as primary (d'=1.86, physically guaranteed) |
| 4 | Module 03 trained models unused | Added sidebar loader + `score_image_deep()` for FAE, MobileNet, EfficientNet |

---

## 🚀 Deploy on Streamlit Cloud (Free)

```
1. Push this folder to GitHub (3 files: app.py, fabriciq_models.py, requirements.txt)
2. Go to https://streamlit.io/cloud → New app
3. Select your repo, set main file = app.py
4. Click Deploy → live URL in ~90 seconds
```

---

## 📁 Repository Structure

```
FabricIQ/
├── app.py                   # Streamlit dashboard (950 lines, 6 tabs)
├── fabriciq_models.py       # All ML models — no Colab dependency
├── requirements.txt         # pip dependencies
├── README.md
└── models/                  # ← Place your trained .pt files here
    ├── README.md            # Instructions for model files
    ├── fae_best.pt          # (download from Drive)
    ├── fae_mu.npy
    ├── fae_sig.npy
    ├── mobilenet_v3_small_recon.pt
    ├── mobilenet_v3_small_selfsup.pt
    ├── efficientnet_b0_recon.pt
    └── efficientnet_b0_selfsup.pt
```

---

## 🔬 Module Summary

### Module 01 — Warp/Weft Count
| Method | MAE | Rank |
|--------|-----|------|
| **Ratio-Consistency ⭐** | **3.48** | **1st (BEST)** |
| Weighted Physical | 4.88 | 2nd |
| Energy-Normalised | 4.95 | 3rd |
| Confidence-Gated | 13.31 | 4th |
| Xu (1996) FFT | 13.52 | 5th (baseline) |
| Shi (2014) Wavelet | — | ⚠️ overestimates ×3 |

### Module 02 — Weave Pattern Recognition (Grammar v5.1 fixed)

**Root cause of Plain/Twill mismatch:**
- CSI: Plain=0.312 vs 2/1T=0.301 → d'=0.11 (useless separator)
- YIDF: Plain=0.711 vs 2/1T=0.676 → d'=0.33 (too much overlap)

**Fix applied:**
- Stage 1 primary: `mean_float` (d'=1.86) — physically guaranteed
  - Plain weave by definition: every yarn crosses → float=1.0
  - Twill by definition: floats exist → float ≥ 2.0
- Stage 2 (2/1T vs 3/1T): coherence (d'=3.15) — unchanged, works well

### Module 03 — Fault Detection
| Model | Silhouette ↑ | Speed |
|-------|-------------|-------|
| Isolation Forest ⭐ | 0.543 (best) | 3.5 img/s |
| Classical Ensemble | 0.502 | — |
| FAE | 0.433 | 38 img/s |
| MobileNet Recon | 0.257 | 49 img/s |

**Using your trained models:**
1. Download `.pt` files from Google Drive `output/train_models/`
2. Place in `models/` folder in your GitHub repo
3. In app sidebar → **Deep Models** → enter `models` → **Check models**
4. Scores appear automatically in Module 03 tab

---

## 🌐 Dashboard Tabs

| Tab | Content |
|-----|---------|
| 🖼️ Image Preview | Original · Enhanced · Yarn detection · FFT spectrum · Projection profiles |
| 📊 Module 01 | Bar comparison · Warp/weft scatter · Feature energy · Benchmark table |
| 🧬 Module 02 | Weave verdict · Radar vs class refs · Interlacement matrix · Feature values |
| 🔍 Module 03 | Score bars · Spatial heatmap · Overlay · Deep model scores (if loaded) |
| 📋 Report | One-page QC summary · CSV download |
| 📐 Comparison | All benchmark tables side-by-side for PhD committee |
