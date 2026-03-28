import streamlit as st
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import json
import torch.nn as nn

st.set_page_config(
    page_title="Brain Tumor Classifier",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0a0e1a;
    color: #e2e8f0;
}

section[data-testid="stSidebar"] {
    background: #0f1525;
    border-right: 1px solid #1e2d4a;
}

section[data-testid="stSidebar"] * {
    color: #94a3b8 !important;
}

.stFileUploader > div {
    background: #0f1525;
    border: 1.5px dashed #1e3a5f;
    border-radius: 16px;
    padding: 2rem;
    transition: border-color 0.2s;
}
.stFileUploader > div:hover { border-color: #3b82f6; }

.stButton > button {
    background: #1d4ed8;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1.5rem;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    transition: background 0.2s;
}
.stButton > button:hover { background: #2563eb; }

.stSpinner > div { border-top-color: #3b82f6 !important; }

div[data-testid="stImage"] img {
    border-radius: 12px;
    border: 1px solid #1e2d4a;
}

h1, h2, h3 { font-family: 'DM Sans', sans-serif !important; }

.result-card {
    background: #0f1525;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    border: 1px solid #1e2d4a;
    margin-bottom: 1rem;
}

.result-label {
    font-size: 13px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 6px;
}

.result-value {
    font-size: 32px;
    font-weight: 600;
    letter-spacing: -0.02em;
}

.tumor-detected { color: #f87171; }
.no-tumor      { color: #34d399; }
.confidence-val { color: #60a5fa; }

.bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    font-family: 'DM Mono', monospace;
    margin-bottom: 5px;
}

.bar-name  { color: #94a3b8; text-transform: capitalize; }
.bar-score { color: #60a5fa; font-weight: 500; }

.bar-track {
    background: #1e2d4a;
    border-radius: 99px;
    height: 8px;
    overflow: hidden;
    margin-bottom: 14px;
}

.bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.8s cubic-bezier(.4,0,.2,1);
}

.pill {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 99px;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.04em;
}

.pill-danger  { background: rgba(248,113,113,0.12); color: #f87171; border: 1px solid rgba(248,113,113,0.25); }
.pill-success { background: rgba(52,211,153,0.12);  color: #34d399; border: 1px solid rgba(52,211,153,0.25); }
.pill-warning { background: rgba(251,191,36,0.12);  color: #fbbf24; border: 1px solid rgba(251,191,36,0.25); }

.divider {
    border: none;
    border-top: 1px solid #1e2d4a;
    margin: 1.5rem 0;
}

.mono { font-family: 'DM Mono', monospace; }
</style>
""", unsafe_allow_html=True)

# Sidebar 
with st.sidebar:
    st.markdown("## NeuroScan")
    st.markdown("Brain MRI analysis using a fine-tuned ResNet-50 model trained on labeled MRI scans.")
    st.markdown("---")
    st.markdown("**Detectable conditions**")
    st.markdown("- Glioma\n- Meningioma\n- Pituitary tumor\n- No tumor")
    st.markdown("---")
    st.markdown("**Supported formats**")
    st.markdown("JPG · JPEG · PNG")
    st.markdown("")
    st.caption("⚠️ For research use only. Not a substitute for clinical diagnosis.")

# Load model 
with open('classes.json') as f:
    class_names = json.load(f)

@st.cache_resource
def load_model():
    m = models.resnet50(weights=None)
    m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.fc.in_features, 4))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m.load_state_dict(torch.load('brain_tumor_model_best.pth', map_location=device))
    m.eval()
    return m.to(device)

model  = load_model()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Header 
st.markdown("# Brain Tumor Classifier")
st.markdown('<p style="color:#64748b;margin-top:-12px;margin-bottom:24px">Brain Tumor Multi-Class Classification using ResNet50 with transfer learning, achieving 98%+ validation accuracy.</p>', unsafe_allow_html=True)

# Upload 
uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is None:
    st.markdown("""
    <div style="text-align:center;padding:3rem 0;color:#334155">
        <div style="font-size:48px;margin-bottom:12px">⬆</div>
        <div style="font-size:15px">Drop an MRI image above to begin analysis</div>
    </div>
    """, unsafe_allow_html=True)

else:
    image = Image.open(uploaded_file).convert("RGB")

    col_img, col_results = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown('<p class="result-label">Uploaded scan</p>', unsafe_allow_html=True)
        st.image(image, width=1200)
        st.markdown(f'<p class="mono" style="font-size:12px;color:#475569;margin-top:6px">{uploaded_file.name} &nbsp;·&nbsp; {image.size[0]}×{image.size[1]}px</p>', unsafe_allow_html=True)

    with col_results:
        with st.spinner("Running inference..."):
            img_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output  = model(img_tensor)
                probs   = torch.softmax(output, dim=1)[0]
                pred_idx    = torch.argmax(probs).item()
                confidence  = float(probs[pred_idx]) * 100

        label = class_names[pred_idx]

        # Result
        if label == 'notumor':
            pill_html   = '<span class="pill pill-success">No tumor detected</span>'
            label_color = 'no-tumor'
            display_name = 'No Tumor'
        elif label == 'glioma':
            pill_html   = '<span class="pill pill-danger">Glioma detected</span>'
            label_color = 'tumor-detected'
            display_name = 'Glioma'
        elif label == 'meningioma':
            pill_html   = '<span class="pill pill-warning">Meningioma detected</span>'
            label_color = 'tumor-detected'
            display_name = 'Meningioma'
        elif label == 'pituitary':
            pill_html   = '<span class="pill pill-warning">Pituitary tumor detected</span>'
            label_color = 'tumor-detected'
            display_name = 'Pituitary Tumor'

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Prediction</div>
            <div class="result-value {label_color}">{display_name}</div>
            <div style="margin-top:10px">{pill_html}</div>
        </div>
        <div class="result-card">
            <div class="result-label">Confidence</div>
            <div class="result-value confidence-val">{confidence:.1f}%</div>
            <div class="bar-track" style="margin-top:12px">
                <div class="bar-fill" style="width:{confidence:.1f}%;background:#3b82f6"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        #Per-class breakdown
        st.markdown('<div class="result-label" style="margin-bottom:14px">Class breakdown</div>', unsafe_allow_html=True)

        colors = {
            'glioma':     '#f87171',
            'meningioma': '#fbbf24',
            'notumor':    '#34d399',
            'pituitary':  '#a78bfa'
        }

        bars_html = ""
        sorted_classes = sorted(enumerate(class_names), key=lambda x: float(probs[x[0]]), reverse=True)
        for i, name in sorted_classes:
            pct   = float(probs[i]) * 100
            color = colors.get(name, '#60a5fa')
            display = name.replace('notumor', 'No Tumor').replace('_', ' ').title()
            bars_html += f"""
            <div class="bar-label">
                <span class="bar-name">{display}</span>
                <span class="bar-score">{pct:.1f}%</span>
            </div>
            <div class="bar-track">
                <div class="bar-fill" style="width:{pct:.1f}%;background:{color}"></div>
            </div>
            """

        st.markdown(bars_html + "</div>", unsafe_allow_html=True)