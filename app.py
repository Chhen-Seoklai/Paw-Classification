import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import time

st.set_page_config(page_title="PawClassify", page_icon="🐾", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Caveat:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #F7F3EE; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem !important; max-width: 100% !important; }

/* ── Kill ALL white boxes ── */
div[data-testid="stVerticalBlockBorderWrapper"]       { background: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important; border-radius: 0 !important; }
div[data-testid="stVerticalBlockBorderWrapper"] > div { background: transparent !important; border: none !important; box-shadow: none !important; }
div[data-testid="column"] > div                       { background: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important; }
div[data-testid="stHorizontalBlock"] > div            { background: transparent !important; border: none !important; box-shadow: none !important; }

/* ── CSS Variables ── */
:root {
    --warm-cream:       #F7F3EE;
    --warm-paper:       #FFFDF9;
    --terracotta:       #C4704B;
    --terracotta-light: #E8C4B0;
    --terracotta-dark:  #A0522D;
    --sage:             #7A9E7E;
    --sage-light:       #C5D8C8;
    --sage-dark:        #5A7E5E;
    --warm-brown:       #5C4A3A;
    --soft-gray:        #9A8E82;
    --sand:             #E8DFD3;
    --ink:              #3D3229;
}

/* ── Nav bar ── */
.nav-bar {
    background: var(--warm-paper);
    border: 1.5px solid var(--sand);
    border-radius: 20px;
    padding: 0.6rem 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 8px rgba(92,74,58,0.06);
}
.nav-brand { display: flex; align-items: center; gap: 12px; }
.nav-paw {
    width: 40px; height: 40px; background: var(--terracotta);
    border-radius: 50%; display: flex; align-items: center;
    justify-content: center; font-size: 20px;
    box-shadow: 0 2px 6px rgba(196,112,75,0.25);
}
.nav-title { font-size: 20px; font-weight: 700; color: var(--ink); letter-spacing: -0.02em; }
.nav-sub   { font-size: 12px; color: var(--soft-gray); font-weight: 500; margin-top: -2px; }
.nav-tags  { display: flex; gap: 8px; }
.tag { font-size: 11px; padding: 5px 14px; border-radius: 100px; font-weight: 600; letter-spacing: 0.02em; }
.tag-terra { background: var(--terracotta-light); color: var(--terracotta-dark); }
.tag-sage  { background: var(--sage-light);       color: var(--sage-dark); }
.tag-warn  { background: #F5E0D5;                 color: #A0522D; }

/* ── Sidebar section label ── */
.side-section {
    font-size: 11px; font-weight: 700; color: var(--soft-gray);
    letter-spacing: 0.12em; text-transform: uppercase;
    margin-bottom: 0.6rem; margin-top: 1.1rem; padding-left: 2px;
}
.side-section:first-child { margin-top: 0; }

/* ── Sidebar about card ── */
.side-card {
    background: var(--warm-paper);
    border: 1.5px solid var(--sand);
    border-radius: 18px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.5rem;
    box-shadow: 0 2px 12px rgba(92,74,58,0.04);
}
.side-about { font-size: 13px; color: var(--warm-brown); line-height: 1.8; }
.side-about b { color: var(--ink); font-weight: 700; }
.handwritten {
    font-family: 'Caveat', cursive;
    font-size: 14px; color: var(--terracotta);
    font-weight: 600; margin-top: 0.6rem; display: block;
}

/* ── How it works list ── */
.feat-card {
    background: var(--warm-paper);
    border: 1.5px solid var(--sand);
    border-radius: 18px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.5rem;
    box-shadow: 0 2px 12px rgba(92,74,58,0.04);
}
.feat-item {
    display: flex; align-items: center; gap: 12px;
    padding: 7px 0; border-bottom: 1px solid #F0E8DE;
    font-size: 13px; font-weight: 500; color: var(--warm-brown);
}
.feat-item:last-child { border-bottom: none; padding-bottom: 0; }
.feat-icon {
    width: 30px; height: 30px; border-radius: 9px;
    background: var(--warm-cream); border: 1.5px solid var(--sand);
    display: flex; align-items: center; justify-content: center;
    font-size: 15px; flex-shrink: 0;
}

/* ── Performance card ── */
.perf-card {
    background: var(--warm-paper);
    border: 1.5px solid var(--sand);
    border-radius: 18px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.5rem;
    box-shadow: 0 2px 12px rgba(92,74,58,0.04);
}
.perf-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 7px 0; border-bottom: 1px solid #F0E8DE; font-size: 13px;
}
.perf-row:last-child { border-bottom: none; }
.perf-label { color: var(--soft-gray); font-weight: 500; }
.perf-value { font-weight: 700; color: var(--ink); }

/* ── Info pill ── */
.info-pill {
    background: var(--sage-light); border: 1.5px solid var(--sage);
    border-radius: 14px; padding: 0.9rem 1.1rem;
    font-size: 12px; color: var(--sage-dark); line-height: 1.7; font-weight: 500;
}
.info-pill b { font-weight: 700; }

/* ── Hero card ── */
.hero-card {
    background: var(--warm-paper);
    border: 1.5px solid var(--sand);
    border-radius: 24px;
    padding: 2rem 2.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 12px rgba(92,74,58,0.04);
    position: relative; overflow: hidden;
}
.hero-card::before {
    content:''; position:absolute; top:-40px; right:-40px;
    width:180px; height:180px; background:var(--terracotta-light);
    border-radius:50%; opacity:0.25; filter:blur(40px);
}
.hero-card::after {
    content:''; position:absolute; bottom:-30px; left:-30px;
    width:140px; height:140px; background:var(--sage-light);
    border-radius:50%; opacity:0.2; filter:blur(35px);
}
.hero-inner { position:relative; z-index:1; }
.hero-kicker {
    font-size: 11px; font-weight: 700; color: var(--terracotta);
    letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 0.5rem;
}
.hero-title {
    font-size: 2.6rem; font-weight: 700; color: var(--ink);
    line-height: 1.1; margin-bottom: 0.5rem; letter-spacing: -0.03em;
}
.hero-title span { color: var(--terracotta); }
.hero-desc { font-size: 14px; color: var(--soft-gray); line-height: 1.6; max-width: 520px; }
.hero-handwritten {
    font-family: 'Caveat', cursive; font-size: 20px;
    color: var(--sage-dark); margin-top: 0.75rem;
    transform: rotate(-1.5deg); display: inline-block;
}

/* ── Section label above upload/result ── */
.col-label {
    font-size: 11px; font-weight: 700; color: var(--soft-gray);
    letter-spacing: 0.12em; text-transform: uppercase;
    margin-bottom: 0.6rem; padding-left: 2px;
}



/* ── File uploader — show native dropzone, style it ── */
.stFileUploader                               { margin: 0 !important; padding: 0 !important; }
.stFileUploader label                         { display: none !important; }
.stFileUploader > div                         { background: transparent !important; border: none !important; padding: 0 !important; }
section[data-testid="stFileUploaderDropzone"] {
    background: var(--warm-cream) !important;
    border: 2px dashed var(--sand) !important;
    border-radius: 14px !important;
    padding: 1.5rem 1rem !important;
    transition: all 0.25s ease !important;
}
section[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--terracotta) !important;
    background: #FDF8F3 !important;
}
section[data-testid="stFileUploaderDropzone"] > div:first-child { display: none !important; }
section[data-testid="stFileUploaderDropzone"] small { display: none !important; }
section[data-testid="stFileUploaderDropzone"] button {
    background: var(--terracotta) !important; color: #fff !important;
    border: none !important; border-radius: 10px !important;
    font-size: 14px !important; font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 0.6rem 1.5rem !important;
    width: auto !important; cursor: pointer !important;
    box-shadow: 0 2px 8px rgba(196,112,75,0.25) !important;
}
section[data-testid="stFileUploaderDropzone"] button:hover {
    background: var(--terracotta-dark) !important;
}
div[data-testid="stFileUploaderFile"] {
    background: var(--warm-cream) !important;
    border: 1.5px solid var(--sand) !important;
    border-radius: 10px !important; margin-top: 0.4rem !important;
}
button[data-testid="baseButton-secondary"] { display: none !important; }

/* ── Image preview ── */
div[data-testid="stImage"] img {
    border-radius: 14px; width: 100%;
    max-height: 300px; object-fit: cover;
    border: 1.5px solid var(--sand);
    box-shadow: 0 4px 16px rgba(92,74,58,0.08);
}

/* ── Meta strip ── */
.meta-strip {
    background: var(--warm-cream); border: 1.5px solid var(--sand);
    border-radius: 12px; padding: 0.85rem 1.1rem;
    font-size: 12px; color: var(--soft-gray);
    line-height: 2.2; font-weight: 500; margin-top: 0.75rem;
}
.meta-strip b { color: var(--warm-brown); font-weight: 600; }

/* ── Classify button ── */
.stButton > button {
    background: var(--terracotta) !important; color: #fff !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 15px !important;
    font-weight: 700 !important; border: none !important; border-radius: 12px !important;
    padding: 0.7rem 1.5rem !important; width: 100% !important;
    margin-bottom: 0.6rem !important;
    box-shadow: 0 2px 8px rgba(196,112,75,0.25); transition: all 0.2s ease;
}
.stButton > button:hover { background: var(--terracotta-dark) !important; transform: translateY(-1px); box-shadow: 0 4px 14px rgba(196,112,75,0.35); }
.stSpinner > div { border-top-color: var(--terracotta) !important; }



/* ── Empty state ── */
.empty-state {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; padding: 3.5rem 1rem;
    gap: 0.5rem; text-align: center;
}
.empty-emoji { font-size: 3rem; margin-bottom: 0.4rem; }
.empty-title { font-size: 17px; font-weight: 700; color: var(--warm-brown); }
.empty-hint  { font-size: 12px; color: var(--soft-gray); line-height: 1.6; max-width: 220px; }

/* ── Prediction ── */
.pred-box {
    background: var(--warm-cream); border: 1.5px solid var(--sand);
    border-radius: 16px; padding: 1.2rem 1.4rem;
    display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;
}
.pred-emoji-box {
    width: 60px; height: 60px; border-radius: 50%;
    background: var(--terracotta-light); border: 2px solid var(--terracotta);
    display: flex; align-items: center; justify-content: center;
    font-size: 2rem; flex-shrink: 0;
}
.pred-emoji-box.dog { background: var(--sage-light); border-color: var(--sage); }
.pred-name { font-size: 2.4rem; font-weight: 700; line-height: 1; letter-spacing: -0.03em; }
.pred-name.cat { color: var(--terracotta); }
.pred-name.dog { color: var(--sage-dark); }
.pred-conf { font-size: 12px; color: var(--soft-gray); margin-top: 5px; font-weight: 500; }
.prob-head { font-size: 11px; font-weight: 700; color: var(--soft-gray); letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.75rem; }
.prob-item { margin-bottom: 0.85rem; }
.prob-item:last-child { margin-bottom: 0; }
.prob-top  { display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; }
.prob-name { font-size: 13px; font-weight: 600; color: var(--warm-brown); }
.prob-pct  { font-size: 14px; font-weight: 700; color: var(--ink); }
.prob-track { background: var(--sand); border-radius: 100px; height: 9px; overflow: hidden; }
.bar-cat { height: 100%; border-radius: 100px; background: var(--terracotta); }
.bar-dog { height: 100%; border-radius: 100px; background: var(--sage); }
.verdict {
    display: flex; align-items: center; gap: 8px;
    border-radius: 12px; padding: 0.8rem 1rem;
    margin-top: 1rem; font-size: 12px; font-weight: 600;
}
.verdict-high { background: var(--sage-light);  color: var(--sage-dark);       border: 1.5px solid var(--sage); }
.verdict-mid  { background: #F5E8D0;             color: #8B6914;                border: 1.5px solid #D4B86A; }
.verdict-low  { background: #F5E0D5;             color: var(--terracotta-dark); border: 1.5px solid var(--terracotta-light); }
.warn-box {
    background: #F5E8D0; border: 1.5px solid #D4B86A; border-radius: 10px;
    padding: 0.75rem 1rem; font-size: 11px; color: #8B6914;
    font-weight: 500; margin-top: 0.75rem; line-height: 1.65;
}
.err-box {
    background: #F5E0D5; border: 1.5px solid var(--terracotta-light);
    border-radius: 10px; padding: 0.75rem 1rem; font-size: 12px;
    color: var(--terracotta-dark); font-weight: 500; margin-bottom: 0.75rem; line-height: 1.65;
}

/* ── Footer ── */
.footer {
    background: var(--warm-paper); border: 1.5px solid var(--sand);
    border-radius: 16px; padding: 0.7rem 1.5rem;
    display: flex; justify-content: space-between; align-items: center;
    margin-top: 1rem;
}
.footer-text { font-size: 10px; color: var(--soft-gray); font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# ── Backend ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_classifier():
    try:
        return load_model('cat_dog_final_model.h5'), None
    except Exception as e:
        return None, str(e)

def predict_image(image: Image.Image, model) -> dict:
    img      = image.convert('RGB').resize((128, 128))
    arr      = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, 0)
    prob_dog = float(model.predict(arr, verbose=0)[0][0])
    prob_cat = 1.0 - prob_dog
    label    = 'Dog' if prob_dog > 0.5 else 'Cat'
    conf     = prob_dog if label == 'Dog' else prob_cat
    return {
        'label':    label,
        'conf':     conf,
        'prob_cat': prob_cat,
        'prob_dog': prob_dog,
        'emoji':    '🐶' if label == 'Dog' else '🐱',
        'css':      'dog' if label == 'Dog' else 'cat',
    }

model, load_error = load_classifier()
if 'result' not in st.session_state:
    st.session_state.result = None

# ── NAV BAR ────────────────────────────────────────────────────────────────────
status_tag = '<span class="tag tag-warn">model missing</span>' if load_error else '<span class="tag tag-sage">ready</span>'
st.markdown(f"""
<div class="nav-bar">
  <div class="nav-brand">
    <div class="nav-paw">🐾</div>
    <div>
      <div class="nav-title">PawClassify</div>
      <div class="nav-sub">Pet Image Classifier</div>
    </div>
  </div>
  <div class="nav-tags">
    <span class="tag tag-terra">MobileNetV2</span>
    {status_tag}
  </div>
</div>""", unsafe_allow_html=True)

# ── LAYOUT: sidebar + main ──────────────────────────────────────────────────────
sb, main = st.columns([1, 2.8], gap="medium")

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with sb:
    # About
    st.markdown('<div class="side-section">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="side-card">
      <div class="side-about">
        A cozy little classifier that tells you whether your photo is a
        <b>cat</b> or a <b>dog</b>. Built using transfer learning
        on a <b>MobileNetV2</b> backbone.
      </div>
      <span class="handwritten">— trained on 210 pet photos ✨</span>
    </div>""", unsafe_allow_html=True)

    # How it works
    st.markdown('<div class="side-section">How it works</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="feat-card">
      <div class="feat-item"><div class="feat-icon">🧠</div>MobileNetV2 base model</div>
      <div class="feat-item"><div class="feat-icon">🖼️</div>128 × 128 image input</div>
      <div class="feat-item"><div class="feat-icon">🔄</div>Data augmentation</div>
      <div class="feat-item"><div class="feat-icon">⚡</div>Two-phase fine-tuning</div>
      <div class="feat-item"><div class="feat-icon">💾</div>Saved as .h5 model</div>
    </div>""", unsafe_allow_html=True)

    # Performance
    st.markdown('<div class="side-section">Performance</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="perf-card">
      <div class="perf-row"><span class="perf-label">Test Accuracy</span><span class="perf-value">78.12%</span></div>
      <div class="perf-row"><span class="perf-label">Test Loss</span><span class="perf-value">0.4047</span></div>
      <div class="perf-row"><span class="perf-label">Cat Recall</span><span class="perf-value">1.00 ✅</span></div>
      <div class="perf-row"><span class="perf-label">Dog Precision</span><span class="perf-value">1.00 ✅</span></div>
      <div class="perf-row"><span class="perf-label">Dataset</span><span class="perf-value">210 images</span></div>
      <div class="perf-row"><span class="perf-label">Classes</span><span class="perf-value">Cat · Dog</span></div>
    </div>""", unsafe_allow_html=True)

    # Tip
    st.markdown("""
    <div class="info-pill">
      <b>Tip:</b> For best results, use clear photos with the pet centered and well-lit.
      Blurry or distant shots may reduce accuracy.
    </div>""", unsafe_allow_html=True)

# ── MAIN ───────────────────────────────────────────────────────────────────────
with main:
    # Hero
    st.markdown("""
    <div class="hero-card">
      <div class="hero-inner">
        <div class="hero-kicker">Deep Learning Classifier</div>
        <div class="hero-title">Is it a <span>cat</span> or a dog?</div>
        <div class="hero-desc">
          Upload any pet photo and our little neural network will tell
          you what it sees — instantly and with a smile.
        </div>
        <div class="hero-handwritten">Just drag, drop &amp; discover! 🐾</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Two columns: upload | result
    up_col, res_col = st.columns([1, 1.1], gap="medium")

    # ── UPLOAD COLUMN ─────────────────────────────────────────────────────────
    with up_col:
        st.markdown('<div class="col-label">Upload Image</div>', unsafe_allow_html=True)

        if load_error:
            st.markdown(
                f'<div class="err-box">⚠️ Model not found. '
                f'Place <b>cat_dog_final_model.h5</b> in the same folder as app.py.</div>',
                unsafe_allow_html=True
            )

        # Native uploader (hidden — triggers file dialog)
        uploaded = st.file_uploader(
            "u", type=['jpg','jpeg','png','bmp','webp'],
            label_visibility="collapsed",
            accept_multiple_files=False
        )

        if uploaded and model:
            if st.button("🔍  Classify Image"):
                image = Image.open(uploaded)
                with st.spinner("Analyzing your photo..."):
                    time.sleep(0.3)
                    st.session_state.result = predict_image(image, model)
                st.rerun()

        if uploaded:
            image = Image.open(uploaded)
            st.image(image, use_container_width=True)
        else:
            st.session_state.result = None

        st.markdown("""
        <div class="meta-strip">
          <b>Base model</b>  · MobileNetV2 (ImageNet)<br>
          <b>Input size</b>  · 128 × 128 × 3 RGB<br>
          <b>Training</b>    · frozen → fine-tune<br>
          <b>Model file</b>  · cat_dog_final_model.h5
        </div>""", unsafe_allow_html=True)


    # ── RESULT COLUMN ─────────────────────────────────────────────────────────
    with res_col:
        st.markdown('<div class="col-label">Result</div>', unsafe_allow_html=True)

        result = st.session_state.result

        if not result:
            st.markdown("""
            <div class="empty-state">
              <div class="empty-emoji">📸</div>
              <div class="empty-title">Upload a pet photo</div>
              <div class="empty-hint">
                Drop a cat or dog image and click
                <b>Classify Image</b> to see the magic happen
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            conf_pct    = result['conf']     * 100
            cat_pct     = result['prob_cat'] * 100
            dog_pct     = result['prob_dog'] * 100
            css         = result['css']
            dog_cls     = 'dog' if css == 'dog' else ''
            cat_bar     = max(cat_pct, 1.5)
            dog_bar     = max(dog_pct, 1.5)
            cat_display = f"{cat_pct:.1f}%" if cat_pct >= 1 else "< 1%"
            dog_display = f"{dog_pct:.1f}%" if dog_pct >= 1 else "< 1%"

            if conf_pct >= 80:
                v_cls, v_ico, v_txt = 'verdict-high', '✨', f'High confidence · {conf_pct:.1f}%'
            elif conf_pct >= 65:
                v_cls, v_ico, v_txt = 'verdict-mid',  '🤔', f'Moderate confidence · {conf_pct:.1f}%'
            else:
                v_cls, v_ico, v_txt = 'verdict-low',  '💭', f'Low confidence · {conf_pct:.1f}%'

            st.markdown(f"""
            <div class="pred-box">
              <div class="pred-emoji-box {dog_cls}">{result['emoji']}</div>
              <div>
                <div class="pred-name {css}">{result['label']}</div>
                <div class="pred-conf">confidence score · {conf_pct:.1f}%</div>
              </div>
            </div>
            <div class="prob-head">Probability Breakdown</div>
            <div class="prob-item">
              <div class="prob-top">
                <span class="prob-name">🐱  Cat</span>
                <span class="prob-pct">{cat_display}</span>
              </div>
              <div class="prob-track">
                <div class="bar-cat" style="width:{cat_bar:.1f}%"></div>
              </div>
            </div>
            <div class="prob-item">
              <div class="prob-top">
                <span class="prob-name">🐶  Dog</span>
                <span class="prob-pct">{dog_display}</span>
              </div>
              <div class="prob-track">
                <div class="bar-dog" style="width:{dog_bar:.1f}%"></div>
              </div>
            </div>
            <div class="verdict {v_cls}"><span>{v_ico}</span> {v_txt}</div>
            """, unsafe_allow_html=True)

            if conf_pct < 65:
                st.markdown("""
                <div class="warn-box">
                  🤔 The model is a bit unsure. Try a clearer photo
                  with the pet centered and well-lit!
                </div>""", unsafe_allow_html=True)


# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <span class="footer-text">MobileNetV2 · ImageNet pretrained · Transfer Learning · 210 training images</span>
  <span class="footer-text">Test Acc 78.12% · Cat Recall 1.00 · Dog Precision 1.00</span>
</div>""", unsafe_allow_html=True)