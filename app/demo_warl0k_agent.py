# demo_warl0k_agent.py
# -------------------------------------------------------------
# WARL0K Micro-AI demo dashboard  (Streamlit, CPU-only Torch)
# -------------------------------------------------------------

import streamlit as st                # 1️⃣  FIRST import
st.set_page_config(page_title="WARL0K Micro-AI Demo", layout="wide")

# --- now safe to import others & use Streamlit -----------------------------
import os, json, random, string, time, platform, psutil, shutil
import numpy as np
import torch
import matplotlib.pyplot as plt
from model import train_secret_regenerator, evaluate_secret_regenerator
import pandas as pd
# ------------------------------------------------------------------
# 0️⃣  Session-state: run flag
# ------------------------------------------------------------------
if "demo_running" not in st.session_state:
    st.session_state.demo_running = False

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

# ------------------------------------------------------------------
# 2️⃣  Sidebar controls  (always visible)
# ------------------------------------------------------------------
with st.sidebar:
    st.title("WARL0K Controls")

    # Start demo button
    if st.button("▶ Start New Demo"):
        st.session_state.demo_running = True
        st.rerun()

    # Danger-zone erase
    st.divider()
    st.markdown("## 🗑️ Danger Zone")
    if st.button("Erase ALL data", key="erase_request"):
        st.session_state["erase_mode"] = True

    if st.session_state.get("erase_mode"):
        st.warning("This removes **models/**, **sessions/** and log files!")
        if st.button("✅ Yes, erase", key="confirm_delete"):
            for d in ("models", "sessions"):
                shutil.rmtree(d, ignore_errors=True)
            for f in ("archive_success.jsonl", "archive_failed.jsonl", "server_log.txt"):
                if os.path.isfile(f):
                    os.remove(f)
            st.success("Data erased. Reloading …")
            st.session_state.pop("erase_mode", None)
            st.session_state.demo_running = False
            st.rerun()

    # System stats (always)
    st.divider()
    st.markdown("## ⚙️ System")
    st.metric("CPU", f"{psutil.cpu_percent()} %")
    mem = psutil.virtual_memory()
    st.metric("Memory", f"{mem.percent} % of {round(mem.total/1e9,1)} GB")
    st.caption(f"Python {platform.python_version()} · Torch {torch.__version__}")

# ------------------------------------------------------------------
# 3️⃣  If demo not running → show placeholder
# ------------------------------------------------------------------
if not st.session_state.demo_running:
    st.write("## 👋 Click **Start New Demo** in the sidebar to begin.")
    st.stop()

# ------------------------------------------------------------------
# 4️⃣  DEMO starts here  (fresh each rerun)
# ------------------------------------------------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("sessions", exist_ok=True)

SESSION_ID    = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
MASTER_SECRET = generate_secret()
OBFUSC_SECRET = generate_secret()
VOCAB         = list(string.ascii_letters + string.digits)
NOISE_RATIO   = 0.25

# st.set_page_config(page_title="WARL0K Micro-AI Demo", layout="wide")
st.title("🔐 WARL0K Micro-AI Agent Demo")
st.caption(f"Session `{SESSION_ID}`")

# --- Live training animation ----------------------------------------------
EPOCHS_ANIM = 80
fake_loss   = fake_loss_curve(EPOCHS_ANIM * 2)

# 1️⃣  loss chart (create once via placeholder)
chart_ph   = st.empty()
loss_chart = chart_ph.line_chart(pd.DataFrame({"loss": [fake_loss[0]]}))

# 2️⃣  two persistent progress bars
ph_bar1    = st.empty()
ph_bar2    = st.empty()
prog1      = ph_bar1.progress(0, text="Master→Obf 0%")
prog2      = ph_bar2.progress(0, text="Obf→Master 0%")

# ---------------- phase 1  (fake animation) -----------------
for ep in range(EPOCHS_ANIM):
    percent = (ep + 1) / EPOCHS_ANIM
    prog1.progress(percent, text=f"Master→Obf {ep+1}/{EPOCHS_ANIM}")
    loss_chart.add_rows({"loss": [fake_loss[ep]]})
    time.sleep(0.04)

# ---------------- real training 1 ----------------------------
model_master_to_obf = train_secret_regenerator(
    secret_str     = OBFUSC_SECRET,
    input_override = MASTER_SECRET,
    vocab          = VOCAB,
    epochs         = 50)

# ---------------- phase 2  (fake animation) -----------------
for ep in range(EPOCHS_ANIM):
    percent = (ep + 1) / EPOCHS_ANIM
    prog2.progress(percent, text=f"Obf→Master {ep+1}/{EPOCHS_ANIM}")
    loss_chart.add_rows({"loss": [fake_loss[EPOCHS_ANIM + ep]]})
    time.sleep(0.04)

# ---------------- real training 2 ----------------------------
model_obf_to_master = train_secret_regenerator(
    secret_str     = MASTER_SECRET,
    input_override = OBFUSC_SECRET,
    vocab          = VOCAB,
    epochs         = 50)

# clear bars when done
ph_bar1.empty()
ph_bar2.empty()
st.success("✔ Micro-models trained")


# --- Save models & JSON -----------------------------------------------------
master_path = f"models/master_to_obf_{SESSION_ID}.pt"
obf_path    = f"models/obf_to_master_{SESSION_ID}.pt"
torch.save(model_master_to_obf, master_path)
torch.save(model_obf_to_master,  obf_path)

json.dump({
    "session_id"       : SESSION_ID,
    "master_secret"    : MASTER_SECRET,
    "obfuscated_secret": OBFUSC_SECRET,
    "model_master_path": master_path,
    "model_obf_path"   : obf_path
}, open(f"sessions/{SESSION_ID}.json","w"), indent=2)

# --- Sidebar extra info -----------------------------------------------------
with st.sidebar:
    st.divider()
    st.markdown("## 📦 Current Models")
    st.code(master_path)
    st.code(obf_path)
    st.divider()
    st.markdown("## 🔑 Current Secrets")
    st.text(f"MASTER : {MASTER_SECRET}")
    st.text(f"OBFSC  : {OBFUSC_SECRET}")

# --- Three-column grid ------------------------------------------------------
noisy_obf = inject_noise(OBFUSC_SECRET, NOISE_RATIO, VOCAB)

c1,c2,c3 = st.columns(3)
with c1:
    st.subheader("🎯 Master Secret")
    st.code(MASTER_SECRET)
    st.pyplot(visualize_secret(MASTER_SECRET,"MASTER_SECRET"))
with c2:
    st.subheader("🕵️ Obfuscated Secret")
    st.code(OBFUSC_SECRET)
    st.pyplot(visualize_secret(OBFUSC_SECRET,"OBFUSC_SECRET"))
with c3:
    st.subheader(f"🔧 Noisy ({NOISE_RATIO*100:.0f}% noise)")
    st.code(noisy_obf)
    st.pyplot(visualize_secret(noisy_obf,"Noisy Obf"))

# --- Reconstruction validation ---------------------------------------------
tensor_in   = torch.tensor([[VOCAB.index(c)] for c in OBFUSC_SECRET], dtype=torch.long)
recon       = evaluate_secret_regenerator(model_obf_to_master, tensor_in, VOCAB)
st.markdown("### 🔁 Reconstructed MASTER_SECRET")
st.code(recon)
st.success("✅ Match" if recon == MASTER_SECRET else "❌ Mismatch")

# --- Noise diff view --------------------------------------------------------
diffs = [f"{a}->{b}" if a!=b else "·" for a,b in zip(OBFUSC_SECRET, noisy_obf)]
st.markdown("### 👁️ Noise Character Diff")
st.write(" ".join(diffs))

# --- Session metadata tab ---------------------------------------------------
tab_demo, tab_manager = st.tabs(["📊 Demo Output", "🗂 Model Manager"])

with tab_demo:
    st.subheader("📄 Session Metadata")
    st.json(json.load(open(f"sessions/{SESSION_ID}.json")))

with tab_manager:
    st.header("🗂 Stored Models")
    model_files = sorted(f for f in os.listdir("models") if f.startswith("master_to_obf"))
    if not model_files:
        st.info("No models saved yet.")
    for f in model_files:
        sid = f.split("_")[-1].split(".")[0]
        obf_match = f.replace("master_to_obf", "obf_to_master")
        cols = st.columns([1,3,3,2])
        cols[0].markdown(f"**{sid}**")
        cols[1].code(f)
        cols[2].code(obf_match)
        if cols[3].button("▶ Run", key=f"run_{f}"):
            NEW_OBF  = generate_secret()
            st.success(f"New OBFUSC_SECRET for {sid}: {NEW_OBF}")

            m1 = train_secret_regenerator(
                    secret_str=NEW_OBF, input_override=MASTER_SECRET, vocab=VOCAB, epochs=50)
            m2 = train_secret_regenerator(
                    secret_str=MASTER_SECRET, input_override=NEW_OBF, vocab=VOCAB, epochs=50)

            new_master = f"models/master_to_obf_{sid}_next.pt"
            new_obf    = f"models/obf_to_master_{sid}_next.pt"
            torch.save(m1,new_master); torch.save(m2,new_obf)

            sess_file = f"sessions/{sid}.json"
            sess = json.load(open(sess_file)) if os.path.exists(sess_file) else {}
            sess.update({
                "obfuscated_secret": NEW_OBF,
                "model_master_path": new_master,
                "model_obf_path"   : new_obf,
                "timestamp"        : time.strftime("%Y-%m-%d %H:%M:%S")
            })
            json.dump(sess, open(sess_file,"w"), indent=2)
            st.balloons()
            st.rerun()
