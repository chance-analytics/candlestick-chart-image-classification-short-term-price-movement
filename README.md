# Candlestick Chart Image Classification for Short-Term Price Movement
*Computer Vision + Multi-Modal Fusion (Images + Technical Indicators)*

## Overview
Traders often “read the chart,” but that process is subjective and inconsistent. This project tests a focused question:

**Can a model learn useful signal directly from candlestick chart images to classify short-term price movement?**

I built a fully reproducible pipeline that turns daily OHLCV data into **standardized candlestick images**, applies **triple-barrier labeling** (Up / Down / Hold), and trains:
- a **vision-only** ResNet18 baseline, and
- **multi-modal fusion** models that combine images with numeric indicators (MACD, RSI, Bollinger).

This is a research/learning project, not a production trading system.

---

## Data
- **Assets:** SPY, AAPL, MSFT, NVDA  
- **Source:** Stooq daily OHLCV (CSV endpoint), cached locally for reproducibility  
- **Period:** Jan 2015 – Dec 2025  
- **Raw rows:** 11,064 daily records  
- **Final image-aligned samples:** 10,748 across four tickers (after horizon trimming)

---

## Labeling (Triple-Barrier, 3-class)
For each date *t* (entry at close):
- Look ahead **20 trading days**
- **Up** if price hits **+5%** first
- **Down** if price hits **−5%** first
- **Hold** if neither barrier is reached by day 20

This produces a practical “Up / Down / No clear move” outcome and reduces label noise vs. a single end-of-window return threshold.

---

## Image Generation
For each labeled date, I generated one PNG:
- **Lookback window:** 60 trading days ending on date *t*
- **Resolution:** 224×224 (CNN-friendly)
- **Chart style:** candlesticks + volume panel
- **No axes/labels:** to avoid text leakage
- **Consistent formatting:** fixed rendering settings so the model learns comparable inputs

Images and indexes are saved under `artifacts_project2/` (see “Outputs” below).

---

## Models
### 1) Vision-only baseline
- **Backbone:** ResNet18 (ImageNet pretrained), fine-tuned for 3 classes

### 2) Multi-modal fusion (images + indicators)
- Extract image features from ResNet18
- Feed numeric indicators to a small MLP
- Concatenate embeddings → final classifier

I tested two variants:
- **Fusion v1:** basic price/volume-derived features (limited gain)
- **Fusion v2:** **MACD + RSI + Bollinger Bands** (slightly better)

---

## Evaluation Setup
- **Time-based split per ticker:** 70% train / 15% validation / 15% test  
  (prevents look-ahead bias and is closer to real deployment)
- **Metrics:** Accuracy, Macro F1, Confusion Matrix

---

## Results (What happened)
### Label balance (overall)
- Up ~44%, Down ~31%, Hold ~25%  
Class mix varied by ticker (SPY had more Hold; NVDA had fewer Hold), suggesting fixed ±5% barriers behave differently across volatility regimes.

### Vision-only (ResNet18)
- **Test accuracy:** 0.452  
- **Macro F1:** 0.447  
Training accuracy was near 0.99 while validation peaked ~0.48 → **overfitting**.
The confusion matrix shows frequent **Up vs Down confusion**, with **Down** having the lowest recall.

### Multi-modal (MACD + RSI + Bollinger)
- **Test accuracy:** 0.459  
- **Macro F1:** 0.455  
This is only a marginal improvement over vision-only and only modestly above a majority baseline (~0.439).

**Takeaway:** Under this setup (4 tickers, 20-day horizon, ±5% barriers, 60-day images), candlestick images did not contain a strong enough stable signal for robust direction classification, and adding standard indicators improved performance only slightly.

---

## Outputs (Artifacts)
The notebook generates:
- cached OHLCV CSV files
- chart images under `artifacts_project2/images/<TICKER>/`
- a samples index table linking `(ticker, date)` → `image_path` + features
- evaluation outputs (classification reports + confusion matrices)

---

## How to run
### 1) Clone
```bash
git clone https://github.com/chance-analytics/candlestick-chart-image-classification-short-term-price-movement.git
cd candlestick-chart-image-classification-short-term-price-movement
```

### 2) Environment (example)
```bash
conda create -n chartcv python=3.11 -y
conda activate chartcv
```

### 3) Install dependencies
```bash
pip install numpy pandas matplotlib scikit-learn pillow tqdm
pip install torch torchvision
```

### 4) Run
Open and run:
- `Project Python Notebook.ipynb`

---

## What I would do next
- Use **volatility-adjusted barriers** (ATR/σ-scaled) to reduce cross-asset label distortion (e.g., SPY vs NVDA Hold rates)
- Expand asset universe for better generalization
- Improve generalization with stronger regularization, augmentation, and early stopping
- Evaluate stability by time slices (regime-aware diagnostics) and calibration

---

## Author
**Chance Xu**  
GitHub: https://github.com/chance-analytics
