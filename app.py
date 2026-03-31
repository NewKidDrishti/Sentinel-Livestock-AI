import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO
import os
import os
os.environ["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="SENTINEL | AI Audit", page_icon="🛡️", layout="wide")

# --- 2. AI ARCHITECTURE ---
class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Linear(512, 512)
    def forward_once(self, x): return self.backbone(x)
    def forward(self, i1, i2): return self.forward_once(i1), self.forward_once(i2)

# --- 3. CACHED MODEL LOADING (Fast!) ---
@st.cache_resource
def load_engines():
    yolo = YOLO("best.pt") # Put your weights in the same folder
    siamese = SiameseNetwork()
    if os.path.exists("siamese_identity.pt"):
        siamese.load_state_dict(torch.load("siamese_identity.pt", map_location="cpu"))
    siamese.eval()
    return yolo, siamese

yolo_model, siamese_model = load_engines()

# --- 4. LOGIC ---
def process_audit(live_pil, dead_pil):
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    t1, t2 = transform(live_pil).unsqueeze(0), transform(dead_pil).unsqueeze(0)
    
    with torch.no_grad():
        f1, f2 = siamese_model(t1, t2)
        cos_sim = F.cosine_similarity(f1, f2).item()
        conf = max(0, cos_sim) * 100
    
    # Posture check via YOLO
    res = yolo_model(dead_pil, verbose=False)[0]
    is_dead = False
    if len(res.boxes) > 0:
        b = res.boxes[0].xyxy[0].cpu().numpy()
        is_dead = (b[2]-b[0])/(b[3]-b[1]) > 1.0 # Lying down check
        
    return conf, is_dead

# --- 5. STREAMLIT UI ---
st.title("🛡️ Sentinel Agentic Audit")
st.markdown("Automated Livestock Identity & Mortality Verification")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Registration")
    live_file = st.file_uploader("Upload Live Sample", type=['jpg', 'png', 'jpeg'], key="live")
    if live_file: st.image(live_file, use_container_width=True)

with col2:
    st.subheader("Evidence")
    dead_file = st.file_uploader("Upload Mortality Sample", type=['jpg', 'png', 'jpeg'], key="dead")
    if dead_file: st.image(dead_file, use_container_width=True)

thresh = st.sidebar.slider("Sensitivity Threshold", 10, 80, 25)

if st.button("EXECUTE AGENTIC AUDIT", use_container_width=True):
    if live_file and dead_file:
        with st.spinner("Analyzing biometric patterns..."):
            conf, is_dead = process_audit(Image.open(live_file).convert("RGB"), Image.open(dead_file).convert("RGB"))
            
            st.divider()
            if conf >= thresh and is_dead:
                st.success(f"✅ AUDIT APPROVED")
                st.balloons()
            else:
                st.error(f"❌ AUDIT REJECTED")
            
            # Metrics
            m1, m2 = st.columns(2)
            m1.metric("Identity Confidence", f"{conf:.2f}%")
            m2.metric("Posture Result", "Lying Down" if is_dead else "Standing")
    else:
        st.warning("Please upload both images first!")
