# 🧵 FabricIQ — Integrated Fabric Quality Intelligence

**PhD Research Demonstration Dashboard · Streamlit Web App**

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

### Module 02 — Weave Pattern Recognition (Grammar v5.1)


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
