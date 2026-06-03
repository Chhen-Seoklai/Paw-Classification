import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import time

st.set_page_config(page_title="PawClassify", page_icon="🐾", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Caveat:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { 
    font-family: 'DM Sans', sans-serif; 
}
.stApp { 
    background: #F7F3EE; 
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { 
    padding: 1.5rem 2rem !important; 
    max-width: 100% !important; 
}

/* ── Remove all Streamlit default boxes ── */
div[data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="stVerticalBlockBorderWrapper"] > div,
section[data-testid="stSidebar"] { 
    background: transparent !important; 
    border: none !important; 
    box-shadow: none !important; 
}
div[data-testid="column"] > div:first-child { 
    background: transparent !important; 
    border: none !important; 
    box-shadow: none !important; 
    padding: 0 !important; 
}
.stContainer, .stContainer > div { 
    background: transparent !important; 
    border: none !important; 
    box-shadow: none !important; 
    padding: 0 !important; 
}

/* ── Organic warm palette ── */
:root {
    --warm-cream: #F7F3EE;
    --warm-paper: #FFFDF9;
    --terracotta: #C4704B;
    --terracotta-light: #E8C4B0;
    --terracotta-dark: #A0522D;
    --sage: #7A9E7E;
    --sage-light: #C5D8C8;
    --sage-dark: #5A7E5E;
    --warm-brown: #5C4A3A;
    --soft-gray: #9A8E82;
    --sand: #E8DFD3;
    --coral: #D4846A;
    --ink: #3D3229;
}

/* ── Top navigation bar ── */
.nav-bar {
    background: var(--warm-paper);
    border: 1.5px solid var(--sand);
    border-radius: 20px;
    padding: 0.6rem 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 8px rgba(92, 74, 58, 0.06);
}
.nav-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}
.nav-paw {
    width: 40px;
    height: 40px;
    background: var(--terracotta);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    box-shadow: 0 2px 6px rgba(196, 112, 75, 0.25);
}
.nav-title {
    font-size: 20px;
    font-weight: 700;
    color: var(--ink);
    letter-spacing: -0.02em;
}
.nav-sub {
    font-size: 12px;
    color: var(--soft-gray);
    font-weight: 500;
    margin-top: -2px;
}
.nav-tags {
    display: flex;
    gap: 8px;
}
.tag {
    font-size: 11px;
    padding: 5px 14px;
    border-radius: 100px;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.tag-sage {
    background: var(--sage-light);
    color: var(--sage-dark);
}
.tag-terra {
    background: var(--terracotta-light);
    color: var(--terracotta-dark);
}
.tag-warn {
    background: #F5E0D5;
    color: #A0522D;
}

/* ── Sidebar cards ── */
.side-section {
    font-size: 11px;
    font-weight: 700;
    color: var(--soft-gray);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
    padding-left: 4px;
}
.side-card {
    background: var(--warm-paper);
    border: 1.5px solid var(--sand);
    border-radius: 18px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 12px rgba(92, 74, 58, 0.04);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.side-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(92, 74, 58, 0.08);
}
.side-about {
    font-size: 14px;
    color: var(--warm-brown);
    line-height: 1.8;
}
.side-about b {
    color: var(--ink);
    font-weight: 700;
}
.feat-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 0;
    border-bottom: 1px solid #F0E8DE;
    font-size: 13px;
    font-weight: 500;
    color: var(--warm-brown);
}
.feat-item:last-child {
    border-bottom: none;
    padding-bottom: 0;
}
.feat-icon {
    width: 32px;
    height: 32px;
    border-radius: 10px;
    background: var(--warm-cream);
    border: 1.5px solid var(--sand);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
}
.perf-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 0;
    border-bottom: 1px solid #F0E8DE;
    font-size: 13px;
}
.perf-row:last-child {
    border-bottom: none;
}
.perf-label {
    color: var(--soft-gray);
    font-weight: 500;
}
.perf-value {
    font-weight: 700;
    color: var(--ink);
}
.handwritten {
    font-family: 'Caveat', cursive;
    font-size: 15px;
    color: var(--terracotta);
    font-weight: 600;
}
.info-pill {
    background: var(--sage-light);
    border: 1.5px solid var(--sage);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    font-size: 13px;
    color: var(--sage-dark);
    line-height: 1.7;
    font-weight: 500;
}
.info-pill b {
    font-weight: 700;
}

/* ── Hero section ── */
.hero-wrap {
    background: var(--warm-paper);
    border: 1.5px solid var(--sand);
    border-radius: 24px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 2px 12px rgba(92, 74, 58, 0.04);
}
.hero-wrap::before {
    content: '';
    position: absolute;
    top: -40px;
    right: -40px;
    width: 180px;
    height: 180px;
    background: var(--terracotta-light);
    border-radius: 50%;
    opacity: 0.3;
    filter: blur(40px);
}
.hero-wrap::after {
    content: '';
    position: absolute;
    bottom: -30px;
    left: -30px;
    width: 140px;
    height: 140px;
    background: var(--sage-light);
    border-radius: 50%;
    opacity: 0.25;
    filter: blur(35px);
}
.hero-content {
    position: relative;
    z-index: 1;
}
.hero-kicker {
    font-size: 12px;
    font-weight: 700;
    color: var(--terracotta);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    color: var(--ink);
    line-height: 1.15;
    margin-bottom: 0.6rem;
    letter-spacing: -0.03em;
}
.hero-title span {
    color: var(--terracotta);
    position: relative;
}
.hero-title span::after {
    content: '';
    position: absolute;
    bottom: 2px;
    left: 0;
    width: 100%;
    height: 8px;
    background: var(--terracotta-light);
    border-radius: 4px;
    z-index: -1;
    opacity: 0.6;
}
.hero-desc {
    font-size: 15px;
    color: var(--soft-gray);
    line-height: 1.6;
    max-width: 480px;
}
.hero-handwritten {
    font-family: 'Caveat', cursive;
    font-size: 22px;
    color: var(--sage-dark);
    margin-top: 0.8rem;
    transform: rotate(-2deg);
    display: inline-block;
}

/* ── Upload & Result cards ── */
.card {
    background: var(--warm-paper);
    border: 1.5px solid var(--sand);
    border-radius: 20px;
    padding: 1.4rem;
    box-shadow: 0 2px 12px rgba(92, 74, 58, 0.04);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(92, 74, 58, 0.08);
}
.card-header {
    font-size: 11px;
    font-weight: 700;
    color: var(--soft-gray);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* ── File uploader styling ── */
.stFileUploader { margin: 0 !important; padding: 0 !important; }
.stFileUploader label { display: none !important; }
.stFileUploader > div { background: transparent !important; border: none !important; padding: 0 !important; }
section[data-testid="stFileUploaderDropzone"] {
    background: var(--warm-cream) !important; 
    border: 2px dashed var(--sand) !important;
    border-radius: 16px !important; 
    padding: 2rem 1.5rem !important;
    transition: all 0.3s ease;
}
section[data-testid="stFileUploaderDropzone"]:hover { 
    border-color: var(--terracotta) !important; 
    background: #FDF8F3 !important;
}
section[data-testid="stFileUploaderDropzone"] > div:first-child { display: none !important; }
section[data-testid="stFileUploaderDropzone"] small { display: none !important; }
section[data-testid="stFileUploaderDropzone"] button {
    background: var(--terracotta) !important; 
    color: #fff !important; 
    border: none !important;
    border-radius: 12px !important; 
    font-size: 15px !important; 
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important; 
    padding: 0.7rem 1.5rem !important;
    width: 100% !important; 
    cursor: pointer !important;
    box-shadow: 0 2px 8px rgba(196, 112, 75, 0.2);
    transition: all 0.2s ease;
}
section[data-testid="stFileUploaderDropzone"] button:hover { 
    background: var(--terracotta-dark) !important; 
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(196, 112, 75, 0.3);
}
div[data-testid="stFileUploaderFile"] { 
    background: var(--warm-cream) !important; 
    border: 1.5px solid var(--sand) !important; 
    border-radius: 10px !important; 
    margin-top: 0.5rem !important; 
}
button[data-testid="baseButton-secondary"] { display: none !important; }

/* ── Classify button ── */
.stButton > button {
    background: var(--sage) !important; 
    color: #fff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 16px !important; 
    font-weight: 700 !important;
    border: none !important; 
    border-radius: 14px !important;
    padding: 0.75rem 1.5rem !important; 
    width: 100% !important;
    margin-top: 0.8rem !important;
    box-shadow: 0 2px 8px rgba(122, 158, 126, 0.25);
    transition: all 0.2s ease;
}
.stButton > button:hover { 
    background: var(--sage-dark) !important; 
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(122, 158, 126, 0.3);
}
.stSpinner > div { border-top-color: var(--terracotta) !important; }

/* ── Image display ── */
div[data-testid="stImage"] img { 
    border-radius: 16px; 
    width: 100%; 
    max-height: 320px; 
    object-fit: cover;
    box-shadow: 0 4px 16px rgba(92, 74, 58, 0.08);
    border: 1.5px solid var(--sand);
}

/* ── Meta strip ── */
.meta-strip {
    background: var(--warm-cream);
    border: 1.5px solid var(--sand);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    font-size: 12px;
    color: var(--soft-gray);
    line-height: 2.2;
    font-weight: 500;
    margin-top: 1rem;
}
.meta-strip b {
    color: var(--warm-brown);
    font-weight: 600;
}

/* ── Empty state ── */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 3rem 1.5rem;
    gap: 0.6rem;
    text-align: center;
}
.empty-emoji {
    font-size: 3rem;
    margin-bottom: 0.5rem;
    filter: grayscale(0.3);
}
.empty-title {
    font-size: 17px;
    font-weight: 700;
    color: var(--warm-brown);
}
.empty-hint {
    font-size: 13px;
    color: var(--soft-gray);
    line-height: 1.6;
    max-width: 240px;
}

/* ── Prediction result ── */
.pred-card {
    background: var(--warm-cream);
    border: 1.5px solid var(--sand);
    border-radius: 18px;
    padding: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
    margin-bottom: 1.2rem;
}
.pred-emoji {
    width: 64px;
    height: 64px;
    border-radius: 50%;
    background: var(--terracotta-light);
    border: 2px solid var(--terracotta);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2.2rem;
    flex-shrink: 0;
    box-shadow: 0 2px 8px rgba(196, 112, 75, 0.15);
}
.pred-emoji.dog-emoji {
    background: var(--sage-light);
    border-color: var(--sage);
    box-shadow: 0 2px 8px rgba(122, 158, 126, 0.15);
}
.pred-name {
    font-size: 2.6rem;
    font-weight: 700;
    line-height: 1;
    letter-spacing: -0.03em;
}
.pred-name.cat {
    color: var(--terracotta);
}
.pred-name.dog {
    color: var(--sage-dark);
}
.pred-sub {
    font-size: 13px;
    color: var(--soft-gray);
    margin-top: 6px;
    font-weight: 500;
}
.prob-header {
    font-size: 11px;
    font-weight: 700;
    color: var(--soft-gray);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.prob-item {
    margin-bottom: 1rem;
}
.prob-item:last-child {
    margin-bottom: 0;
}
.prob-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}
.prob-label {
    font-size: 14px;
    font-weight: 600;
    color: var(--warm-brown);
}
.prob-value {
    font-size: 15px;
    font-weight: 700;
    color: var(--ink);
}
.prob-track {
    background: var(--sand);
    border-radius: 100px;
    height: 10px;
    overflow: hidden;
}
.bar-cat {
    height: 100%;
    border-radius: 100px;
    background: var(--terracotta);
    transition: width 0.6s ease;
}
.bar-dog {
    height: 100%;
    border-radius: 100px;
    background: var(--sage);
    transition: width 0.6s ease;
}
.verdict {
    display: flex;
    align-items: center;
    gap: 10px;
    border-radius: 14px;
    padding: 0.9rem 1.2rem;
    margin-top: 1.2rem;
    font-size: 13px;
    font-weight: 600;
}
.verdict-high {
    background: var(--sage-light);
    color: var(--sage-dark);
    border: 1.5px solid var(--sage);
}
.verdict-mid {
    background: #F5E8D0;
    color: #8B6914;
    border: 1.5px solid #D4B86A;
}
.verdict-low {
    background: #F5E0D5;
    color: var(--terracotta-dark);
    border: 1.5px solid var(--terracotta-light);
}
.warn-box {
    background: #F5E8D0;
    border: 1.5px solid #D4B86A;
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    font-size: 12px;
    color: #8B6914;
    font-weight: 500;
    margin-top: 1rem;
    line-height: 1.65;
}
.err-box {
    background: #F5E0D5;
    border: 1.5px solid var(--terracotta-light);
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    font-size: 13px;
    color: var(--terracotta-dark);
    font-weight: 500;
    line-height: 1.65;
    margin-bottom: 1rem;
}

/* ── Footer ── */
.footer {
    background: var(--warm-paper);
    border: 1.5px solid var(--sand);
    border-radius: 18px;
    padding: 0.8rem 1.8rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 1.2rem;
}
.footer-text {
    font-size: 11px;
    color: var(--soft-gray);
    font-weight: 500;
}

/* ── Smooth transitions ── */
.hero-wrap, .card, .side-card, .pred-card, .footer, .nav-bar {
    transition: all 0.3s ease;
}
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

# ── NAVIGATION BAR ─────────────────────────────────────────────────────────────
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

# ── LAYOUT ─────────────────────────────────────────────────────────────────────
sb, main = st.columns([1, 2.8], gap="medium")

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with sb:
    st.markdown('<div class="side-section">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="side-card">
      <div class="side-about">
        A cozy little classifier that tells you whether your photo is a 
        <b>cat</b> or a <b>dog</b>. Built with love using transfer learning 
        on a MobileNetV2 backbone.
      </div>
      <div class="handwritten" style="margin-top: 0.8rem;">— trained on 210 pet photos ✨</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="side-section">How it works</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="side-card" style="padding:1rem 1.2rem;">
      <div class="feat-item"><div class="feat-icon">🧠</div>MobileNetV2 base model</div>
      <div class="feat-item"><div class="feat-icon">🖼️</div>128 × 128 image input</div>
      <div class="feat-item"><div class="feat-icon">🔄</div>Data augmentation</div>
      <div class="feat-item"><div class="feat-icon">📊</div>5-Fold Cross Validation</div>
      <div class="feat-item"><div class="feat-icon">⚡</div>Two-phase fine-tuning</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="side-section">Performance</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="side-card" style="padding:1rem 1.2rem;">
      <div class="perf-row"><span class="perf-label">Test Accuracy</span><span class="perf-value">87.50%</span></div>
      <div class="perf-row"><span class="perf-label">Validation</span><span class="perf-value">91.51% <span style="font-weight:400;color:var(--soft-gray)">±6.29%</span></span></div>
      <div class="perf-row"><span class="perf-label">Dataset</span><span class="perf-value">210 images</span></div>
      <div class="perf-row"><span class="perf-label">Classes</span><span class="perf-value">Cat · Dog</span></div>
      <div class="perf-row"><span class="perf-label">Folds</span><span class="perf-value">5-fold CV</span></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="info-pill">
      <b>Tip:</b> For best results, use clear photos with the pet centered and well-lit. Blurry or distant shots may confuse the model.
    </div>""", unsafe_allow_html=True)

# ── MAIN CONTENT ───────────────────────────────────────────────────────────────
with main:
    # Hero with organic blobs
    st.markdown("""
    <div class="hero-wrap">
      <div class="hero-content">
        <div class="hero-kicker">Deep Learning Classifier</div>
        <div class="hero-title">Is it a <span>cat</span> or a <span>dog</span>?</div>
        <div class="hero-desc">Upload any pet photo and our little neural network will tell you what it sees — instantly and with a smile.</div>
        <div class="hero-handwritten">Just drag, drop & discover! 🐾</div>
      </div>
    </div>""", unsafe_allow_html=True)

    up_col, res_col = st.columns([1, 1.1], gap="medium")

    # ── UPLOAD COLUMN ─────────────────────────────────────────────────────────
    with up_col:
        st.markdown('<div class="card-header">Upload Image</div>', unsafe_allow_html=True)
        
        if load_error:
            st.markdown(f'<div class="err-box">⚠️ Model file not found.<br>Place <b>cat_dog_best_model.h5</b> in the same folder as this app.</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "u", type=['jpg','jpeg','png','bmp','webp'],
            label_visibility="collapsed",
            accept_multiple_files=False
        )

        if uploaded and model:
            if st.button("🔍  Classify Image", use_container_width=True):
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
          <b>Base model</b>   · MobileNetV2 (ImageNet)<br>
          <b>Input size</b>  · 128 × 128 × 3 RGB<br>
          <b>Training</b> · frozen → fine-tune<br>
          <b>Model file</b>   · cat_dog_best_model.h5
        </div>""", unsafe_allow_html=True)

    # ── RESULT COLUMN ─────────────────────────────────────────────────────────
    with res_col:
        st.markdown('<div class="card-header">Result</div>', unsafe_allow_html=True)
        
        result = st.session_state.result

        if not result:
            st.markdown("""
            <div class="empty-state">
              <div class="empty-emoji">📸</div>
              <div class="empty-title">Upload a pet photo</div>
              <div class="empty-hint">Drop a cat or dog image here and click <b>Classify Image</b> to see the magic happen</div>
            </div>""", unsafe_allow_html=True)
        else:
            conf_pct = result['conf']     * 100
            cat_pct  = result['prob_cat'] * 100
            dog_pct  = result['prob_dog'] * 100
            css      = result['css']
            emoji_cls = 'dog-emoji' if css == 'dog' else ''

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
            <div class="pred-card">
              <div class="pred-emoji {emoji_cls}">{result['emoji']}</div>
              <div>
                <div class="pred-name {css}">{result['label']}</div>
                <div class="pred-sub">confidence score · {conf_pct:.1f}%</div>
              </div>
            </div>
            <div class="prob-header">Probability Breakdown</div>
            <div class="prob-item">
              <div class="prob-top">
                <span class="prob-label">🐱  Cat</span>
                <span class="prob-value">{cat_display}</span>
              </div>
              <div class="prob-track"><div class="bar-cat" style="width:{cat_bar:.1f}%"></div></div>
            </div>
            <div class="prob-item">
              <div class="prob-top">
                <span class="prob-label">🐶  Dog</span>
                <span class="prob-value">{dog_display}</span>
              </div>
              <div class="prob-track"><div class="bar-dog" style="width:{dog_bar:.1f}%"></div></div>
            </div>
            <div class="verdict {v_cls}"><span>{v_ico}</span> {v_txt}</div>
            """, unsafe_allow_html=True)

            if conf_pct < 65:
                st.markdown("""
                <div class="warn-box">
                  🤔 The model is feeling a bit unsure about this one. Try a clearer photo with the pet centered and good lighting!
                </div>
                """, unsafe_allow_html=True)

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <span class="footer-text">MobileNetV2 · ImageNet pretrained · 5-Fold CV · 210 training images</span>
  <span class="footer-text">Test Acc 87.50% · Val Acc 91.51% ±6.29%</span>
</div>""", unsafe_allow_html=True)