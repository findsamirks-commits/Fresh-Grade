"""Streamlit Interactive Application for Fresh-Grade.

Vision Transformer (ViT) Perishable Produce Quality Grading + Shelf-Life Regression.
Demonstrates simultaneous grade classification, continuous days-to-spoil estimation,
interactive ViT Attention Rollout heatmap interpretability, and automated markdown pricing recommendations.
"""

import json
import os
import sys
import pandas as pd
from PIL import Image
import streamlit as st

# Ensure project root is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.inference import FreshGradePredictor, generate_attention_overlay
from src.data import generate_synthetic_produce_image

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Fresh-Grade 🍅🍌 | ViT Quality Grading & Shelf-Life",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .metric-badge {
        font-size: 1.2rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 6px;
    }
    .grade-fresh { background-color: #DCFCE7; color: #166534; }
    .grade-ripe { background-color: #FEF9C3; color: #854D0E; }
    .grade-overripe { background-color: #FFEDD5; color: #9A3412; }
    .grade-spoiled { background-color: #FEE2E2; color: #991B1B; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">🍅🍌📸 Fresh-Grade: ViT Quality Grading & Shelf-Life</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">'
    "Dual-head Vision Transformer (ViT) pipeline estimating <b>Quality Grade</b> and "
    "continuous <b>Days-to-Spoil</b> with <b>Attention Rollout</b> interpretability to automate perishable dynamic markdowns and FEFO dark-store picking."
    "</div>",
    unsafe_allow_html=True,
)

GRADE_EMOJIS = {
    "Fresh": "🟢 Fresh",
    "Ripe": "🟡 Ripe",
    "Overripe": "🟠 Overripe",
    "Spoiled": "🔴 Spoiled",
}

# -----------------------------------------------------------------------------
# Sidebar: Setup & Presets
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Evaluation Controls")

    base_price = st.number_input(
        "Base Price (INR ₹)",
        min_value=10.0,
        max_value=500.0,
        value=50.0,
        step=5.0,
        help="Standard retail selling price before markdown",
    )

    st.markdown("---")
    st.subheader("💡 Produce Simulation Presets")
    st.caption("Generate realistic synthetic produce samples:")

    produce_sel = st.selectbox("Produce SKU", ["Tomato", "Banana"], index=0)

    col1, col2 = st.columns(2)
    with col1:
        preset_fresh = st.button("🌱 Fresh", use_container_width=True)
        preset_ripe = st.button("🍎 Ripe", use_container_width=True)
    with col2:
        preset_overripe = st.button("🍂 Overripe", use_container_width=True)
        preset_spoiled = st.button("⚠️ Spoiled", use_container_width=True)

    st.markdown("---")
    st.subheader("🔬 ViT Explainability")
    show_attention = st.toggle(
        "🔥 Show Attention Heatmap",
        value=False,
        help="Superimposes Vision Transformer self-attention rollout matrix showing salient features (bruising, mold, decay).",
    )
    attn_alpha = 0.45
    if show_attention:
        attn_alpha = st.slider("Heatmap Intensity", min_value=0.2, max_value=0.8, value=0.45, step=0.05)

    st.markdown("---")
    st.markdown("### 📋 Dynamic Markdown Ladder")
    st.markdown(
        """
        - **🟢 Fresh (>4 days):** Full Margin (`HOLD`, ₹0 discount)
        - **🟡 Ripe (2–4 days):** FEFO Priority (`SELL_FIRST`, 15% off)
        - **🟠 Overripe (1–2 days):** Clearance (`DISCOUNT_40`, 40% off)
        - **🔴 Spoiled (<1 day):** Zero-Recovery (`DUMP_SALVAGE`)
        """
    )

    st.markdown("---")
    st.markdown("### 👨‍💻 Developer & Architect")
    st.markdown("**Samir Sandhwar** | Category Head & AI Engineer")
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-findsamirks--commits-black?style=flat&logo=github)](https://github.com/findsamirks-commits/Fresh-Grade)")
    st.markdown("[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-findsamirks%2FFresh--Grade--ViT-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/findsamirks/Fresh-Grade-ViT)")
    st.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-samirsandhwar-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/samirsandhwar)")

# -----------------------------------------------------------------------------
# Preset Image Generation or Upload
# -----------------------------------------------------------------------------
demo_asset_path = os.path.join(ROOT_DIR, "assets", "demo_produce.jpg")

if "active_image" not in st.session_state:
    if not os.path.exists(demo_asset_path):
        generate_synthetic_produce_image(produce_type="tomato", grade="Ripe", output_path=demo_asset_path)
    st.session_state.active_image = Image.open(demo_asset_path)
    st.session_state.active_name = "demo_tomato_ripe.jpg"

if preset_fresh:
    img = generate_synthetic_produce_image(produce_type=produce_sel.lower(), grade="Fresh")
    st.session_state.active_image = img
    st.session_state.active_name = f"{produce_sel.lower()}_fresh.jpg"
elif preset_ripe:
    img = generate_synthetic_produce_image(produce_type=produce_sel.lower(), grade="Ripe")
    st.session_state.active_image = img
    st.session_state.active_name = f"{produce_sel.lower()}_ripe.jpg"
elif preset_overripe:
    img = generate_synthetic_produce_image(produce_type=produce_sel.lower(), grade="Overripe")
    st.session_state.active_image = img
    st.session_state.active_name = f"{produce_sel.lower()}_overripe.jpg"
elif preset_spoiled:
    img = generate_synthetic_produce_image(produce_type=produce_sel.lower(), grade="Spoiled")
    st.session_state.active_image = img
    st.session_state.active_name = f"{produce_sel.lower()}_spoiled.jpg"

# File uploader
uploaded_file = st.file_uploader("📸 Or Upload Live Inspection Photo", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    st.session_state.active_image = Image.open(uploaded_file)
    st.session_state.active_name = uploaded_file.name

# -----------------------------------------------------------------------------
# Inference Execution
# -----------------------------------------------------------------------------
@st.cache_resource
def get_predictor():
    return FreshGradePredictor()

predictor = get_predictor()
result = predictor.predict(
    st.session_state.active_image,
    base_price_inr=base_price,
    return_attention=True,
    return_overlay=True,
)

# -----------------------------------------------------------------------------
# Main Visual & Metric Display
# -----------------------------------------------------------------------------
col_img, col_metrics = st.columns([0.45, 0.55])

with col_img:
    if show_attention and "attention_overlay" in result:
        st.subheader("🔥 ViT Attention Rollout Heatmap")
        st.image(
            result["attention_overlay"],
            caption="Jet Colormap: Highlights visual patches driving ripeness and spoilage scoring",
            width="stretch",
        )
    else:
        st.subheader("🔍 Produce Inspection Frame")
        st.image(
            st.session_state.active_image,
            caption=f"Active Sample: {st.session_state.active_name}",
            width="stretch",
        )

with col_metrics:
    st.subheader("📊 Multi-Task Model Predictions")

    m1, m2, m3 = st.columns(3)
    grade = result["quality_grade"]
    conf = result["grade_confidence"] * 100.0
    days = result["days_to_spoil"]

    display_grade = GRADE_EMOJIS.get(grade, grade)
    m1.metric("Predicted Grade", display_grade, delta=f"{conf:.1f}% Conf")
    m2.metric("Remaining Shelf Life", f"{days:.1f} Days", delta=result["shelf_life_band"])
    m3.metric("Inference Latency", f"{result['inference_ms']} ms", help="Evaluated on 224x224 frame")

    # Grade Probability Distribution
    st.markdown("#### 🎯 Class Probability Distribution")
    probs_df = pd.DataFrame(list(result["all_grades"].items()), columns=["Grade", "Probability"])
    st.bar_chart(probs_df.set_index("Grade"))

# -----------------------------------------------------------------------------
# Markdown Bridge Commercial Resolution
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("🏷️ Dynamic Markdown & Replenishment Decision")

md = result["markdown"]
action = md["action"]
discount = md["suggested_pct"]
orig_p = md["base_price_inr"]
sugg_p = md["suggested_price_inr"]

c_act1, c_act2, c_act3 = st.columns(3)
c_act1.metric("Merchandising Action", action)
c_act2.metric("Suggested Discount", f"{discount}% OFF")
c_act3.metric("Adjusted Selling Price", f"₹{sugg_p:.2f}", delta=f"-₹{orig_p - sugg_p:.2f}" if discount > 0 else "Full Price")

# Operational Alert Box
if action == "DUMP_SALVAGE":
    st.error(f"🚨 **IMMEDIATE DISPOSAL REQUIRED (🔴 SPOILED):** {md['reason']}", icon="🚨")
elif action == "DISCOUNT_40":
    st.warning(f"⚠️ **URGENT MARKDOWN CLEARANCE (🟠 40% OFF):** {md['reason']}", icon="⚠️")
elif action == "SELL_FIRST":
    st.info(f"🛒 **FEFO PICKING PRIORITY (🟡 15% OFF):** {md['reason']}", icon="🛒")
else:
    st.success(f"✅ **MAINTAIN FULL MARGIN (🟢 FRESH):** {md['reason']}", icon="✅")

# -----------------------------------------------------------------------------
# JSON Export Payload
# -----------------------------------------------------------------------------
st.markdown("---")
col_exp1, col_exp2 = st.columns([0.7, 0.3])
with col_exp1:
    st.markdown("### 📥 Integration Payload for `Perishable-Markdown-Optimizer`")
    st.caption("Structured JSON contract consumed by automated pricing engines and dark store picking totes.")

# Remove PIL object before JSON dumping
exportable_result = {k: v for k, v in result.items() if k not in ("attention_overlay", "attention_map")}

with col_exp2:
    st.download_button(
        "💾 Download Inference Report (JSON)",
        data=json.dumps(exportable_result, indent=2),
        file_name=f"fresh_grade_audit_{st.session_state.active_name.split('.')[0]}.json",
        mime="application/json",
        use_container_width=True,
    )

with st.expander("🔍 View Raw JSON Audit Report", expanded=False):
    st.json(exportable_result)
