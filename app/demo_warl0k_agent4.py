# demo_warl0k_agent4.py
# -------------------------------------------------------------
# WARL0K Micro-AI demo dashboard  (Streamlit, CPU-only Torch)
# -------------------------------------------------------------
import streamlit as st
st.set_page_config(page_title="WARL0K Micro-AI Demo", layout="wide")

# ── Standard imports (after page config) ───────────────────────
import os, json, random, string, time, platform, psutil, shutil
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from model import train_secret_regenerator, evaluate_secret_regenerator

# ------------------------------------------------------------------
# 1️⃣  Helper functions
# ------------------------------------------------------------------
def generate_secret(n=16):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=n))

def visualize_secret(secret, title):
    fig, ax = plt.subplots(figsize=(6, 2.2))
    vals = [ord(c) for c in secret]
    ax.bar(range(len(secret)), vals, tick_label=list(secret))
    ax.set_title(title)
    ax.set_ylim(min(vals) - 2, max(vals) + 2)
    ax.tick_params(axis='x', labelrotation=90)
    return fig

def inject_noise(s, ratio=.25, vocab=None):
    if vocab is None:
        vocab = list(string.ascii_letters + string.digits)
    chars = list(s)
    for i in range(len(chars)):
        if random.random() < ratio:
            chars[i] = random.choice(vocab)
    return ''.join(chars)

def fake_loss_curve(steps):
    base = np.linspace(1.0, 0.1, steps)
    noise = np.random.uniform(-0.05, 0.05, steps)
    return np.clip(base + noise, 0.05, 1.0)
# --------------------------------------------------------------
# 0️⃣  Config – load / default
# --------------------------------------------------------------
CFG_PATH = "demo_config.json"
DEFAULTS = { "epochs_anim": 50, "noise_ratio": 0.25 }

if os.path.isfile(CFG_PATH):
    CFG = { **DEFAULTS, **json.load(open(CFG_PATH)) }
else:
    CFG = DEFAULTS.copy()

# --------------------------------------------------------------
# 1️⃣  Session state: flags & logs
# --------------------------------------------------------------
if "demo_running" not in st.session_state:
    st.session_state.demo_running = False
if "logs" not in st.session_state:
    st.session_state.logs = []

def log(msg:str):
    st.session_state.logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

# --------------------------------------------------------------
# 2️⃣  Sidebar
# --------------------------------------------------------------
with st.sidebar:
    st.title("WARL0K Controls")

    # -- Configuration inputs
    st.markdown("### ⚙️ Parameters")
    cfg_epochs = st.number_input("Animation epochs", 10, 300, CFG["epochs_anim"])
    cfg_noise  = st.slider("Noise ratio (%)", 0, 100, int(CFG["noise_ratio"]*100))

    if st.button("💾 Save & Apply"):
        CFG = { "epochs_anim": cfg_epochs, "noise_ratio": cfg_noise / 100 }
        json.dump(CFG, open(CFG_PATH, "w"), indent=2)
        st.success("Saved. Click ▶ Start New Demo.")
        st.stop()

    # -- Start demo
    if st.button("▶ Start New Demo"):
        st.session_state.demo_running = True
        st.session_state.logs = []
        st.rerun()

    # -- Erase data
    st.divider()
    st.markdown("## 🗑️ Danger Zone")
    if st.button("Erase ALL data"):
        st.session_state["erase_mode"] = True

    if st.session_state.get("erase_mode"):
        st.warning("Delete **models/**, **sessions/**, logs?")
        if st.button("✅ Confirm erase"):
            for d in ("models","sessions"):
                shutil.rmtree(d, ignore_errors=True)
            for f in ("archive_success.jsonl","archive_failed.jsonl","server_log.txt"):
                if os.path.isfile(f): os.remove(f)
            st.success("Erased. Reloading…")
            st.session_state.clear()
            st.rerun()

    # -- System stats
    st.divider()
    st.markdown("## 📊 System")
    st.metric("CPU", f"{psutil.cpu_percent()} %")
    mem = psutil.virtual_memory()
    st.metric("Memory", f"{mem.percent} % of {round(mem.total/1e9,1)} GB")
    st.caption(f"Python {platform.python_version()} · Torch {torch.__version__}")

# --------------------------------------------------------------
# 3️⃣  Idle screen
# --------------------------------------------------------------
if not st.session_state.demo_running:
    st.write("## 👋 Use the sidebar to configure and **Start New Demo**.")
    st.stop()

# --------------------------------------------------------------
# 4️⃣  Demo run (fresh at every rerun)
# --------------------------------------------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("sessions", exist_ok=True)

SESSION_ID    = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
MASTER_SECRET = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
OBFUSC_SECRET = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
VOCAB         = list(string.ascii_letters + string.digits)
NOISE_RATIO   = CFG["noise_ratio"]
EPOCHS_ANIM   = CFG["epochs_anim"]

st.title("🔐 WARL0K Micro-AI Agent Demo")
st.caption(f"Session `{SESSION_ID}` — noise {NOISE_RATIO*100:.0f}%")

# 4.1  Progress bars + chart
fake_loss = np.linspace(1.0, 0.1, EPOCHS_ANIM*2) + np.random.uniform(-0.05,0.05,EPOCHS_ANIM*2)

chart_ph   = st.empty()
loss_chart = chart_ph.line_chart(pd.DataFrame({"loss":[fake_loss[0]]}))

pb1_ph = st.empty(); pb2_ph = st.empty()
pb1   = pb1_ph.progress(0, text="Master→Obf 0%")
pb2   = pb2_ph.progress(0, text="Obf→Master 0%")

log("Starting fake phase 1\n")
for ep in range(EPOCHS_ANIM):
    pb1.progress((ep+1)/EPOCHS_ANIM, text=f"Master→Obf {ep+1}/{EPOCHS_ANIM}")
    loss_chart.add_rows({"loss":[fake_loss[ep]]})
    time.sleep(0.01)

log("Training model_master_to_obf\n")
model_master_to_obf = train_secret_regenerator(
    secret_str=OBFUSC_SECRET, input_override=MASTER_SECRET, vocab=VOCAB, epochs=CFG["epochs_anim"])

log("Starting fake phase 2\n")
for ep in range(EPOCHS_ANIM):
    idx = EPOCHS_ANIM+ep
    pb2.progress((ep+1)/EPOCHS_ANIM, text=f"Obf→Master {ep+1}/{EPOCHS_ANIM}")
    loss_chart.add_rows({"loss":[fake_loss[idx]]})
    time.sleep(0.01)

log("Training model_obf_to_master\n")
model_obf_to_master = train_secret_regenerator(
    secret_str=MASTER_SECRET, input_override=OBFUSC_SECRET, vocab=VOCAB, epochs=CFG["epochs_anim"])

pb1_ph.empty(); pb2_ph.empty()
st.success("✔ Training complete")
log("Training finished\n")

# 4.2  Save artefacts
master_path = f"models/master_to_obf_{SESSION_ID}.pt"
obf_path    = f"models/obf_to_master_{SESSION_ID}.pt"
torch.save(model_master_to_obf, master_path)
torch.save(model_obf_to_master,  obf_path)
log("Models saved\n")

session_json = {
    "session_id": SESSION_ID,
    "master_secret": MASTER_SECRET,
    "obfuscated_secret": OBFUSC_SECRET,
    "model_master_path": master_path,
    "model_obf_path": obf_path
}
json.dump(session_json, open(f"sessions/{SESSION_ID}.json","w"), indent=2)

# 4.3  Main three-column visuals
noisy_obf = ''.join(
    random.choice(VOCAB) if random.random()<NOISE_RATIO else c
    for c in OBFUSC_SECRET)

c1,c2,c3 = st.columns(3)
with c1:
    st.subheader("🎯 Master")
    st.code(MASTER_SECRET)
    st.pyplot(visualize_secret(MASTER_SECRET,"MASTER"))
with c2:
    st.subheader("🕵️ Obfuscated")
    st.code(OBFUSC_SECRET)
    st.pyplot(visualize_secret(OBFUSC_SECRET,"OBF"))
with c3:
    st.subheader("🔧 Noisy")
    st.code(noisy_obf)
    st.pyplot(visualize_secret(noisy_obf,"Noisy"))

# 4.4  Reconstruction check
tensor_in = torch.tensor([[VOCAB.index(c)] for c in OBFUSC_SECRET], dtype=torch.long)
recon     = evaluate_secret_regenerator(model_obf_to_master, tensor_in, VOCAB)

st.markdown("### 🔁 Reconstructed MASTER_SECRET")
st.code(recon)
st.success("✅ Match" if recon==MASTER_SECRET else "❌ Mismatch")

# 4.5  Logs expander
with st.expander("🪵 Runtime Log"):
    st.write("\n".join(st.session_state.logs))

# 4.6  Model Manager tab
tab_demo, tab_mgr = st.tabs(["📊 Demo Output", "🗂 Model Manager"])
with tab_demo:
    st.subheader("📄 Session Metadata")
    st.json(session_json)

with tab_mgr:
    st.header("🗂 Stored Models")
    mdl_files = sorted(f for f in os.listdir("models") if f.startswith("master_to_obf"))
    if not mdl_files:
        st.info("No models yet.")
    for f in mdl_files:
        sid  = f.split("_")[-1].split(".")[0]
        obf_match = f.replace("master_to_obf","obf_to_master")
        cols = st.columns([1,3,3,2])
        cols[0].markdown(f"**{sid}**")
        cols[1].code(f)
        cols[2].code(obf_match)
        if cols[3].button("▶ Retrain", key=f"run_{f}"):
            NEW_OBF = generate_secret()
            log(f"Retrain for {sid} with NEW_OBF {NEW_OBF}\n")
            m1 = train_secret_regenerator(secret_str=NEW_OBF, input_override=MASTER_SECRET, vocab=VOCAB, epochs=50)
            m2 = train_secret_regenerator(secret_str=MASTER_SECRET, input_override=NEW_OBF, vocab=VOCAB, epochs=50)
            torch.save(m1, f"models/master_to_obf_{sid}_next.pt")
            torch.save(m2, f"models/obf_to_master_{sid}_next.pt")
            st.success(f"Models retrained for {sid}")
            st.rerun()
