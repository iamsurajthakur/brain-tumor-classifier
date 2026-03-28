import streamlit as st
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import json
import torch.nn as nn

with open('classes.json') as f:
    class_names = json.load(f) 

@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, 4)
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load('brain_tumor_model_best.pth', map_location=device))
    model.eval()
    return model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

model = load_model()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

st.set_page_config(page_title="Brain Tumor Classifier", page_icon="🧠")
st.title(" Brain Tumor Classifier")
st.write("Upload a brain MRI image and the model will predict whether a tumor is present.")

uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", width=700)

    # ── Predict ───────────────────────────────────────────────
    with st.spinner("Analyzing..."):
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            confidence = float(probs[pred_idx]) * 100

    #  Show result 
    label = class_names[pred_idx]

    st.divider()
    st.write("### Result")

    if label == 'notumor':
        st.success(f"✅ No Tumor Detected — {confidence:.1f}% confidence")
    elif label == 'glioma':
        st.error(f"⚠️ Glioma Tumor Detected — {confidence:.1f}% confidence")
    elif label == 'meningioma':
        st.error(f"⚠️ Meningioma Tumor Detected — {confidence:.1f}% confidence")
    elif label == 'pituitary':
        st.warning(f"⚠️ Pituitary Tumor Detected — {confidence:.1f}% confidence")

    st.write("### Confidence for Each Class")
    for i, name in enumerate(class_names):
        st.write(f"**{name.replace('_', ' ').title()}**")
        st.progress(float(probs[i]))