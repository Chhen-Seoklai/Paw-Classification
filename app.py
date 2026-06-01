import streamlit as st
import numpy as np
from PIL import Image
from tf_keras.models import load_model
import time

st.set_page_config(page_title="PawClassify", page_icon="🐾", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: #f2ece0; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1rem 1.5rem !important; max-width: 100% !important; }

/* ── Kill ALL Streamlit white boxes ── */
div[data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="stVerticalBlockBorderWrapper"] > div,
section[data-testid="stSidebar"] { background: transparent !important; border: none !important; box-shadow: none !important; }
div[data-testid="column"] > div:first-child { background: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important; }
.stContainer, .stContainer > div { background: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important; }

/* ── Topbar ── */
.topbar {
    background: #fff; border: 0.5px solid #d8d0c4; border-radius: 14px;
    padding: 0 1.5rem; height: 56px;
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 1rem;
}
.topbar-left { display: flex; align-items: center; gap: 12px; }
.logo-box { width: 36px; height: 36px; border-radius: 10px; background: #185FA5; display: flex; align-items: center; justify-content: center; font-size: 18px; flex-shrink: 0; }
.logo-name { font-size: 16px; font-weight: 700; color: #1a1a1a; letter-spacing: -0.01em; }
.logo-sub  { font-size: 11px; color: #a09080; font-family: 'IBM Plex Mono', monospace; }
.topbar-pills { display: flex; gap: 6px; }
.pill { font-size: 11px; padding: 4px 12px; border-radius: 100px; font-weight: 600; font-family: 'IBM Plex Mono', monospace; }
.pill-blue  { background: #deeeff; color: #0C447C; }
.pill-green { background: #d6f5e8; color: #085041; }
.pill-red   { background: #fde8e8; color: #791F1F; }

/* ── Sidebar ── */
.sec-head { font-size: 11px; font-weight: 700; color: #8a8070; letter-spacing: 0.1em; text-transform: uppercase; font-family: 'IBM Plex Mono', monospace; margin-bottom: 0.5rem; padding-left: 2px; }
.sb-card { background: #fff; border: 0.5px solid #d8d0c4; border-radius: 12px; padding: 1.1rem 1.2rem; margin-bottom: 0.75rem; }
.sb-about { font-size: 13px; color: #3a3028; line-height: 1.8; }
.sb-about b { color: #1a1a1a; font-weight: 700; }
.feat-row { display: flex; align-items: center; gap: 10px; padding: 7px 0; border-bottom: 0.5px solid #ede8de; font-size: 13px; font-weight: 500; color: #3a3028; }
.feat-row:last-child { border-bottom: none; padding-bottom: 0; }
.feat-pip { width: 28px; height: 28px; border-radius: 8px; background: #f2ece0; border: 0.5px solid #d8d0c4; display: flex; align-items: center; justify-content: center; font-size: 14px; flex-shrink: 0; }
.perf-row { display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 0.5px solid #ede8de; font-size: 12px; }
.perf-row:last-child { border-bottom: none; }
.perf-key { color: #8a8070; font-family: 'IBM Plex Mono', monospace; }
.perf-val { font-weight: 700; color: #1a1a1a; font-family: 'IBM Plex Mono', monospace; }
.info-box { background: #deeeff; border: 0.5px solid #85B7EB; border-radius: 10px; padding: 0.9rem 1rem; font-size: 12px; color: #0C447C; line-height: 1.7; font-weight: 500; }
.info-box b { font-weight: 700; }

/* ── Hero ── */
.hero-card {
    background: #185FA5; border-radius: 16px;
    padding: 1.75rem 2rem; margin-bottom: 0.75rem;
    display: flex; align-items: center; justify-content: space-between;
    gap: 1rem;
}
.hero-left {}
.hero-eye { font-size: 10px; font-weight: 700; color: rgba(255,255,255,0.6); letter-spacing: 0.15em; text-transform: uppercase; font-family: 'IBM Plex Mono', monospace; margin-bottom: 0.4rem; }
.hero-h { font-size: 2rem; font-weight: 700; color: #fff; line-height: 1.1; margin-bottom: 0.4rem; letter-spacing: -0.02em; }
.hero-h span { color: #FFE082; }
.hero-p { font-size: 13px; color: rgba(255,255,255,0.75); line-height: 1.5; }
.hero-right { display: flex; gap: 10px; flex-shrink: 0; }
.hero-stat { background: rgba(255,255,255,0.12); border: 0.5px solid rgba(255,255,255,0.2); border-radius: 10px; padding: 0.75rem 1.1rem; text-align: center; }
.hero-stat-val { font-size: 1.3rem; font-weight: 700; color: #fff; font-family: 'IBM Plex Mono', monospace; line-height: 1; }
.hero-stat-key { font-size: 10px; color: rgba(255,255,255,0.6); text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px; font-family: 'IBM Plex Mono', monospace; }

.hero-card,
.up-card,
.res-card,
.sb-card {
    transition: all .25s ease;
}

.hero-card:hover,
.up-card:hover,
.res-card:hover,
.sb-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 30px rgba(0,0,0,.08);
}
            
/* ── Upload card ── */
.up-card { background: #fff; border: 0.5px solid #d8d0c4; border-radius: 12px; padding: 1.1rem; }
.meta-strip { background: #f8f4ec; border: 0.5px solid #d8d0c4; border-radius: 8px; padding: 0.7rem 1rem; font-size: 11px; color: #8a8070; line-height: 2.2; font-family: 'IBM Plex Mono', monospace; margin-top: 0.75rem; }
.meta-strip b { color: #3a3028; font-weight: 600; }

/* ── Result card ── */
.res-card { background: #fff; border: 0.5px solid #d8d0c4; border-radius: 12px; padding: 1.1rem; }
.empty-state { display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 2.5rem 1rem; gap: 0.5rem; text-align: center; }
.empty-ico { font-size: 2.5rem; margin-bottom: 0.25rem; }
.empty-t1 { font-size: 15px; font-weight: 600; color: #3a3028; }
.empty-t2 { font-size: 12px; color: #c8c0b4; font-family: 'IBM Plex Mono', monospace; line-height: 1.6; }

/* ── Prediction ── */
.pred-box { background: #f8f4ec; border: 0.5px solid #d8d0c4; border-radius: 10px; padding: 1rem 1.1rem; display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem; }
.pred-icon { width: 58px; height: 58px; border-radius: 12px; background: #deeeff; border: 1px solid #85B7EB; display: flex; align-items: center; justify-content: center; font-size: 1.9rem; flex-shrink: 0; }
.pred-name { font-size: 2.4rem; font-weight: 700; line-height: 1; letter-spacing: -0.02em; }
.pred-name.cat { color: #185FA5; }
.pred-name.dog { color: #0F6E56; }
.pred-sub { font-size: 11px; color: #a09080; margin-top: 5px; font-family: 'IBM Plex Mono', monospace; }
.prob-head { font-size: 11px; font-weight: 700; color: #8a8070; letter-spacing: 0.08em; text-transform: uppercase; font-family: 'IBM Plex Mono', monospace; margin-bottom: 0.75rem; }
.prob-item { margin-bottom: 0.9rem; }
.prob-item:last-child { margin-bottom: 0; }
.prob-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; }
.prob-name { font-size: 13px; font-weight: 600; color: #3a3028; }
.prob-num  { font-size: 14px; font-weight: 700; color: #1a1a1a; font-family: 'IBM Plex Mono', monospace; }
.prob-track { background: #ede8de; border-radius: 100px; height: 9px; overflow: hidden; }
.bar-cat { height: 100%; border-radius: 100px; background: #185FA5; }
.bar-dog { height: 100%; border-radius: 100px; background: #0F6E56; }
.verdict { display: flex; align-items: center; gap: 8px; border-radius: 10px; padding: 0.75rem 1rem; margin-top: 1rem; font-size: 12px; font-weight: 600; font-family: 'IBM Plex Mono', monospace; }
.verdict-high { background: #d6f5e8; color: #085041; border: 0.5px solid #5DCAA5; }
.verdict-mid  { background: #fff3db; color: #633806; border: 0.5px solid #EF9F27; }
.verdict-low  { background: #fde8e8; color: #791F1F; border: 0.5px solid #F09595; }
.warn-msg { background: #fff3db; border: 0.5px solid #EF9F27; border-radius: 8px; padding: 0.75rem 1rem; font-size: 11px; color: #633806; font-family: 'IBM Plex Mono', monospace; margin-top: 0.75rem; line-height: 1.65; }
.err-msg  { background: #fde8e8; border: 0.5px solid #F09595; border-radius: 8px; padding: 0.75rem 1rem; font-size: 12px; color: #791F1F; font-family: 'IBM Plex Mono', monospace; line-height: 1.65; margin-bottom: 0.75rem; }

/* ── Footer ── */
.foot { background: #fff; border: 0.5px solid #d8d0c4; border-radius: 12px; padding: 0.65rem 1.5rem; display: flex; justify-content: space-between; align-items: center; margin-top: 0.75rem; }
.foot-t { font-size: 10px; color: #b8b0a4; font-family: 'IBM Plex Mono', monospace; }

/* ── File uploader ── */
.stFileUploader { margin: 0 !important; padding: 0 !important; }
.stFileUploader label { display: none !important; }
.stFileUploader > div { background: transparent !important; border: none !important; padding: 0 !important; }
section[data-testid="stFileUploaderDropzone"] {
    background: #f8f4ec !important; border: 2px dashed #c8c0b4 !important;
    border-radius: 10px !important; padding: 1.5rem 1rem !important;
}
section[data-testid="stFileUploaderDropzone"]:hover { border-color: #185FA5 !important; }
section[data-testid="stFileUploaderDropzone"] > div:first-child { display: none !important; }
section[data-testid="stFileUploaderDropzone"] small { display: none !important; }
section[data-testid="stFileUploaderDropzone"] button {
    background: #185FA5 !important; color: #fff !important; border: none !important;
    border-radius: 8px !important; font-size: 14px !important; font-weight: 600 !important;
    font-family: 'IBM Plex Sans', sans-serif !important; padding: 0.55rem 1.25rem !important;
    width: 100% !important; cursor: pointer !important;
}
section[data-testid="stFileUploaderDropzone"] button:hover { background: #0C447C !important; }
div[data-testid="stFileUploaderFile"] { background: #f8f4ec !important; border: 0.5px solid #d8d0c4 !important; border-radius: 8px !important; margin-top: 0.4rem !important; }
button[data-testid="baseButton-secondary"] { display: none !important; }

/* ── Image ── */
div[data-testid="stImage"] img { border-radius: 10px; width: 100%; max-height: 300px; object-fit: cover; }

/* ── Classify button ── */
.stButton > button {
    background: #185FA5 !important; color: #fff !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 15px !important; font-weight: 700 !important;
    border: none !important; border-radius: 10px !important;
    padding: 0.65rem 1.5rem !important; width: 100% !important;
    margin-bottom: 0.6rem !important;
}
.stButton > button:hover { background: #0C447C !important; }
.stSpinner > div { border-top-color: #185FA5 !important; }
</style>
""", unsafe_allow_html=True)


# ── Backend ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_classifier():
    try:
        return load_model('cat_dog_best_model.h5'), None
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

# ── TOPBAR ─────────────────────────────────────────────────────────────────────
spill = '<span class="pill pill-red">model error</span>' if load_error else '<span class="pill pill-green">model loaded</span>'
st.markdown(f"""
<div class="topbar">
  <div class="topbar-left">
    <div class="logo-box">🐾</div>
    <div>
      <div class="logo-name">PawClassify</div>
      <div class="logo-sub">CNN Image Classifier</div>
    </div>
  </div>
  <div class="topbar-pills">
    <span class="pill pill-blue">MobileNetV2</span>
    {spill}
  </div>
</div>""", unsafe_allow_html=True)

# ── LAYOUT ─────────────────────────────────────────────────────────────────────
sb, main = st.columns([1, 2.8], gap="medium")

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with sb:
    st.markdown('<div class="sec-head">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-card">
      <div class="sb-about">Predicts whether an image is a <b>cat</b> or a <b>dog</b> using a CNN built on <b>MobileNetV2</b> with transfer learning.</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-head">Features used</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-card" style="padding:0.9rem 1.1rem;">
      <div class="feat-row"><div class="feat-pip">🧠</div>MobileNetV2 base model</div>
      <div class="feat-row"><div class="feat-pip">🖼️</div>128 × 128 image input</div>
      <div class="feat-row"><div class="feat-pip">🔄</div>Data augmentation</div>
      <div class="feat-row"><div class="feat-pip">📊</div>5-Fold K-Fold CV</div>
      <div class="feat-row"><div class="feat-pip">⚡</div>Two-phase fine-tuning</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-head">Model performance</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-card" style="padding:0.9rem 1.1rem;">
      <div class="perf-row"><span class="perf-key">test acc</span><span class="perf-val">87.50%</span></div>
      <div class="perf-row"><span class="perf-key">val acc</span><span class="perf-val">91.51% <span style="font-weight:400;color:#a09080">±6.29%</span></span></div>
      <div class="perf-row"><span class="perf-key">dataset</span><span class="perf-val">210 images</span></div>
      <div class="perf-row"><span class="perf-key">classes</span><span class="perf-val">Cat · Dog</span></div>
      <div class="perf-row"><span class="perf-key">folds</span><span class="perf-val">5-fold</span></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
      Prediction is based on a <b>MobileNetV2</b> CNN model trained with
      transfer learning and validated using 5-fold cross validation.
    </div>""", unsafe_allow_html=True)

# ── MAIN ───────────────────────────────────────────────────────────────────────
with main:
    # Hero with inline stats — no more blank boxes from st.columns
    st.markdown("""
    <div class="hero-card">
      <div class="hero-left">
        <div class="hero-eye">Deep learning classifier</div>
        <div class="hero-h">Is it a <span>cat</span> or a <span>dog</span> ?</div>
        <div class="hero-p">Upload any photo — our CNN identifies it instantly</div>
      </div>
      
    </div>""", unsafe_allow_html=True)

    up_col, res_col = st.columns([1, 1.1], gap="medium")

    # ── UPLOAD ────────────────────────────────────────────────────────────────
    with up_col:
        st.markdown('<div class="sec-head">Upload image</div>', unsafe_allow_html=True)
        

        if load_error:
            st.markdown(f'<div class="err-msg">⚠ Model not found.<br>Place <b>cat_dog_best_model.h5</b> in the same folder as app.py.</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "u", type=['jpg','jpeg','png','bmp','webp'],
            label_visibility="collapsed",
            accept_multiple_files=False
        )

        if uploaded and model:
            if st.button("🔍  Classify Image"):
                image = Image.open(uploaded)
                with st.spinner("Analyzing..."):
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
          <b>base</b>   · MobileNetV2 (ImageNet)<br>
          <b>input</b>  · 128 × 128 × 3 RGB<br>
          <b>phases</b> · frozen → fine-tune<br>
          <b>file</b>   · cat_dog_best_model.h5
        </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── RESULT ────────────────────────────────────────────────────────────────
    with res_col:
        st.markdown('<div class="sec-head">Result</div>', unsafe_allow_html=True)
        

        result = st.session_state.result

        if not result:
            st.markdown("""
            <div class="empty-state">
              <div class="empty-ico">📸</div>
              <div class="empty-t1">Upload a pet photo</div>
              <div class="empty-t2">Drop a cat or dog image and<br>click Classify Image</div>
            </div>""", unsafe_allow_html=True)
        else:
            conf_pct = result['conf']     * 100
            cat_pct  = result['prob_cat'] * 100
            dog_pct  = result['prob_dog'] * 100
            css      = result['css']

            cat_bar     = max(cat_pct, 1.5)
            dog_bar     = max(dog_pct, 1.5)
            cat_display = f"{cat_pct:.1f}%" if cat_pct >= 1 else "< 1%"
            dog_display = f"{dog_pct:.1f}%" if dog_pct >= 1 else "< 1%"

            if conf_pct >= 80:
                v_cls, v_ico, v_txt = 'verdict-high', '✅', f'High confidence · {conf_pct:.1f}%'
            elif conf_pct >= 65:
                v_cls, v_ico, v_txt = 'verdict-mid',  '⚠️', f'Moderate confidence · {conf_pct:.1f}%'
            else:
                v_cls, v_ico, v_txt = 'verdict-low',  '❌', f'Low confidence · {conf_pct:.1f}%'

            st.markdown(f"""
            <div class="pred-box">
              <div class="pred-icon">{result['emoji']}</div>
              <div>
                <div class="pred-name {css}">{result['label']}</div>
                <div class="pred-sub">confidence · {conf_pct:.1f}%</div>
              </div>
            </div>
            <div class="prob-head">Probability breakdown</div>
            <div class="prob-item">
              <div class="prob-top">
                <span class="prob-name">🐱  Cat</span>
                <span class="prob-num">{cat_display}</span>
              </div>
              <div class="prob-track"><div class="bar-cat" style="width:{cat_bar:.1f}%"></div></div>
            </div>
            <div class="prob-item">
              <div class="prob-top">
                <span class="prob-name">🐶  Dog</span>
                <span class="prob-num">{dog_display}</span>
              </div>
              <div class="prob-track"><div class="bar-dog" style="width:{dog_bar:.1f}%"></div></div>
            </div>
            <div class="verdict {v_cls}"><span>{v_ico}</span> {v_txt}</div>
            """, unsafe_allow_html=True)

            if conf_pct < 65:
                st.markdown("""
                <div class="warn-msg">⚠ Model is uncertain — try a clearer photo with the animal centered and well-lit.</div>
                """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="foot">
  <span class="foot-t">MobileNetV2 · ImageNet pretrained · 5-Fold CV · 210 training images</span>
  <span class="foot-t">Test Acc 87.50% · K-Fold Val 91.51% ±6.29%</span>
</div>""", unsafe_allow_html=True)