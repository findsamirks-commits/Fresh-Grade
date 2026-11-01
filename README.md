# Fresh-Grade 🍅🍌📸

**Vision Transformer (ViT) multi-task deep learning model for perishable produce quality grading + continuous shelf-life regression.**

A single image of a tomato, banana, apple, or leafy green goes in; two unified predictions come out:

1. **Quality Grade** — `Fresh / Ripe / Overripe / Spoiled` (4-class Softmax classification)
2. **Days-to-Spoil** — continuous non-negative shelf-life estimate (regression via Softplus)

Together these predictions drive an automated **Markdown Bridge** that chains directly into [`Perishable-Markdown-Optimizer`](https://github.com/findsamirks-commits/Perishable-Markdown-Optimizer) and enforces **First-Expire-First-Out (FEFO)** picking logic for quick-commerce dark stores.

Built with **PyTorch**, **HuggingFace Transformers (ViT-B/16)**, **timm**, and an interactive **Streamlit** visual inspection UI.

---

## 🎯 Business Problem

Perishables (dairy, F&V, bakery, meat) account for **25–40% of a modern grocery basket** but drive **>60% of shrinkage**. In quick-commerce dark stores and high-velocity fulfillment hubs:

| Pain Point | Estimated Loss | Fresh-Grade Solution |
|---|---|---|
| Produce accepted at inbound hub without visual grading | 3–8% of fresh GMV lost to spoilage | Inbound camera audit instantly grades produce and flags defective crates |
| Flat markdown rules (e.g., flat 10% off everything near EOD) | 2–4% margin erosion vs optimal | Dynamic markdown ladder maps days-to-spoil to price elasticity |
| Pickers unable to distinguish Ripe vs Overripe → customer complaints | 1–2% CSAT & reorder churn | FEFO routing tags (`SELL_FIRST` vs `DISCOUNT_40`) guide picking totes |

**Fresh-Grade** replaces manual visual inspection with a camera + a Vision Transformer. One photo → grade + days-left + a pricing action in **< 150 ms**.

---

## 🏗️ Architecture

```
                          ┌─────────────────────────┐
    produce photo ───►    │   ViT-B/16 Backbone     │ ──► 768-d [CLS] embedding
    (.jpg/.png)           │   (ImageNet-21k init,   │
                          │    produce fine-tuned)  │
                          └───────────┬─────────────┘
                                      │
                     ┌────────────────┴──────────────────┐
                     │                                   │
                     ▼                                   ▼
          ┌──────────────────┐                ┌──────────────────┐
          │  Classification  │                │ Regression Head  │
          │  Head (Grade)    │                │ (Shelf Life)     │
          │  Linear(768→4)   │                │ Linear(768→1)    │
          │  + Softmax       │                │ + Softplus       │
          └────────┬─────────┘                └────────┬─────────┘
                   │                                   │
                   ▼                                   ▼
             quality grade                       days-to-spoil
             + confidence                        (continuous, 0–14)
                   │                                   │
                   └─────────────────┬─────────────────┘
                                     │
                                     ▼
                          ┌──────────────────────────┐
                          │     Markdown Bridge      │ ──► Suggested Price (₹)
                          │ (price elasticity rules) │     + Action Label
                          │                          │     (HOLD / SELL_FIRST /
                          │                          │      DISCOUNT_40 / DUMP)
                          └──────────────────────────┘
```

### Joint Multi-Task Loss Formulation
$$\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_{\text{CE}}(\hat{y}_{\text{grade}}, y_{\text{grade}}) + \beta \cdot \mathcal{L}_{\text{MSE}}(\log(\hat{d} + 1), \log(d + 1))$$

- **Log1p transform ($\log(d + 1)$):** Prevents large shelf-life outliers (>10 days) from dominating regression gradients over critical near-expiry produce (0–3 days).
- **Softplus Activation ($\hat{d} = \ln(1 + e^z)$):** Guarantees strictly non-negative shelf-life estimates.
- **Default weights:** $\alpha = 1.0$, $\beta = 0.5$ (configurable in `configs/fresh_grade.yaml`).

### Why ViT (not ResNet / EfficientNet)?
1. **Global Self-Attention for Fine-Grained Texture:** ViT's self-attention captures subtle surface mold, bruising patches, and skin wrinkling across arbitrary distances.
2. **Pretrained ImageNet-21k Transfer:** Pre-training on 14 million images provides rich semantic representations for biological textures.
3. **Interpretability:** Multi-head attention maps show store managers and auditors exactly which pixels drove the spoilage rating.

---

## 📂 Project Structure

```
Fresh-Grade/
├── README.md                      # Architecture, loss formulations, and benchmarks
├── requirements.txt               # Dependencies (torch, transformers, timm, streamlit)
├── .gitignore
├── configs/
│   └── fresh_grade.yaml           # Training hyperparams, loss weights, markdown ladders
├── src/
│   ├── __init__.py
│   ├── model.py                   # Multi-Task ViT (shared [CLS] -> Classification & Regression)
│   ├── data.py                    # FreshGradeDataset, transforms & synthetic produce generator
│   ├── markdown_bridge.py         # Days-to-spoil -> price markdown ladder rules
│   ├── inference.py               # Single-image predictor & JSON report generator
│   └── train.py                   # PyTorch training loop with joint multi-task loss
├── app/
│   └── streamlit_app.py           # Interactive inspection UI with live grade & markdown alerts
├── tests/
│   └── test_markdown.py           # Unit tests for pricing ladders & model tensor dimensions
└── assets/
    └── demo_produce.jpg           # High-resolution synthetic audit asset
```

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/findsamirks-commits/Fresh-Grade.git
cd Fresh-Grade
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 2. Run Self-Tests
```bash
pytest tests/test_markdown.py
```

### 3. Run Inference on Produce Image
```bash
python src/inference.py
```

### 4. Launch the Streamlit Interactive UI
```bash
streamlit run app/streamlit_app.py
```

---

## 📊 Example Output

```json
{
  "image": "demo_produce.jpg",
  "quality_grade": "Overripe",
  "grade_confidence": 0.88,
  "all_grades": {
    "Fresh": 0.02,
    "Ripe": 0.08,
    "Overripe": 0.88,
    "Spoiled": 0.02
  },
  "days_to_spoil": 1.2,
  "shelf_life_band": "12–48h",
  "markdown": {
    "suggested_pct": 40,
    "suggested_price_inr": 30.0,
    "base_price_inr": 50.0,
    "action": "DISCOUNT_40",
    "reason": "Accelerated decay risk (1.2 days remaining). Aggressive 40% clearance discount to stimulate volume sellout."
  },
  "inference_ms": 115.4
}
```

---

## 🧪 Evaluation Targets

| Metric | Target | Measurement |
|---|---|---|
| **Grade Accuracy (4-class)** | $\ge 0.88$ | Weighted average across classes |
| **Macro-F1** | $\ge 0.85$ | Scikit-learn macro-F1 (handles class imbalance) |
| **Days-to-Spoil MAE** | $\le 1.2 \text{ days}$ | Mean Absolute Error against ground-truth shelf life |
| **Markdown Agreement** | $\ge 0.90$ | Decision concordance vs senior merchandiser gold labels |
| **Inference Latency** | $< 150 \text{ ms}$ | Evaluated on $224 \times 224$ frame |

---

## 🔗 Companion Repositories

- [`Perishable-Markdown-Optimizer`](https://github.com/findsamirks-commits/Perishable-Markdown-Optimizer) — Fresh-Grade's days-to-spoil estimate directly triggers dynamic discount clearance ladders.
- [`Shelf-Vision`](https://github.com/findsamirks-commits/Shelf-Vision) — Fresh-Grade inspects produce quality on the crate/pallet; Shelf-Vision audits planogram compliance on the shelf.
- [`Last-Mile-Delivery-Batching-Agent`](https://github.com/findsamirks-commits/Last-Mile-Delivery-Batching-Agent) — Produces tagged with `SELL_FIRST` receive expedited routing priority.

---

## 📄 License
MIT License. Built by [Samir Sandhwar](https://github.com/findsamirks-commits) — Category Head & Retail AI Engineer.
