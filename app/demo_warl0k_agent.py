# demo_warl0k_agent.py
# ---------------------------------------------------------------------------
# A complete Streamlit micro-demo of WARL0K secret generation & model training
# ---------------------------------------------------------------------------
import os, json, random, string, time, platform, psutil
import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt
from model import train_secret_regenerator, evaluate_secret_regenerator   # <- your existing model utils

# ---------------------------------------------------------------------------
# 0️⃣  𝗦𝗲𝘁-𝘂𝗽  &  U𝘁𝗶𝗹𝘀
# ---------------------------------------------------------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("sessions", exist_ok=True)

def generate_secret(n: int = 16) -> str:
    return ''.join(random.choices(string.ascii_letters + string.digits, k=n))

def visualize_secret(secret: str, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 2.3))
    ascii_vals = [ord(c) for c in secret]
    ax.bar(range(len(secret)), ascii_vals, tick_label=list(secret))
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("ASCII")
    ax.set_ylim(min(ascii_vals)-2, max(ascii_vals)+2)
    ax.tick_params(axis='x', labelrotation=90)
    return fig

def inject_noise(s: str, ratio: float = .25, vocab=None) -> str:
    """Randomly flip `ratio` of characters in s."""
    if vocab is None:
        vocab = list(string.ascii_letters + string.digits)
    chars = list(s)
    for i in range(len(chars)):
        if random.random() < ratio:
            chars[i] = random.choice(vocab)
    return ''.join(chars)

def fake_loss_curve(steps: int) -> np.ndarray:
    base  = np.linspace(1.0, 0.1, steps)
    noise = np.random.uniform(-0.05, 0.05, steps)
    return np.clip(base + noise, 0.05, 1.0)

def visualize_noisy_diff(clean: str, noisy: str, title: str) -> plt.Figure:
    """
    Bar-plot of ASCII values, colouring bars RED only where `noisy[i] != clean[i]`.
    """
    fig, ax = plt.subplots(figsize=(6, 2.3))
    ascii_vals = [ord(c) for c in noisy]
    colours = ["red" if c1 != c2 else "lightgray"
               for c1, c2 in zip(clean, noisy)]

    ax.bar(range(len(noisy)), ascii_vals,
           tick_label=list(noisy),
           color=colours, edgecolor='black')
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("ASCII")
    ax.set_ylim(min(ascii_vals) - 2, max(ascii_vals) + 2)
    ax.tick_params(axis='x', labelrotation=90)
    return fig

# ---------------------------------------------------------------------------
# 1️⃣  𝗚𝗲𝗻𝗲𝗿𝗮𝘁𝗲 𝗦𝗲𝗰𝗿𝗲𝘁𝘀  &  𝗩𝗼𝗰𝗮𝗯
# ---------------------------------------------------------------------------
SESSION_ID    = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
MASTER_SECRET = generate_secret()
OBFUSC_SECRET = generate_secret()
VOCAB         = list(string.ascii_letters + string.digits)
NOISE_RATIO   = 0.25

# ---------------------------------------------------------------------------
# 2️⃣  𝗦𝘁𝗿𝗲𝗮𝗺𝗹𝗶𝘁 𝗣𝗮𝗴𝗲 𝗖𝗼𝗻𝗳𝗶𝗴
# ---------------------------------------------------------------------------
st.set_page_config(page_title="WARL0K Micro-AI Demo", layout="wide")
st.title("🔐 WARL0K Micro-AI Agent Demo")
st.caption(f"Session ID `{SESSION_ID}`")

# ---------------------------------------------------------------------------
# 3️⃣  𝗟𝗶𝘃𝗲 𝗣𝗿𝗼𝗴𝗿𝗲𝘀𝘀-𝗣𝗮𝗿 / 𝗧𝗿𝗮𝗶𝗻 𝗠𝗼𝗱𝗲𝗹𝘀
# ---------------------------------------------------------------------------
EPOCHS_ANIM = 80               # for progress animation
progress_bar = st.progress(0, text="Initialising …")
loss_chart   = st.line_chart()
fake_loss    = fake_loss_curve(EPOCHS_ANIM * 2)

# phase-1 animation
for ep in range(EPOCHS_ANIM):
    time.sleep(0.04)
    progress_bar.progress((ep+1)/(EPOCHS_ANIM*2),
                          text=f"Master→Obf (fake) {ep+1}/{EPOCHS_ANIM}")
    loss_chart.add_rows({"loss": [fake_loss[ep]]})

# real training 1
model_master_to_obf = train_secret_regenerator(
    secret_str     = OBFUSC_SECRET,
    input_override = MASTER_SECRET,
    vocab          = VOCAB,
    epochs         = EPOCHS_ANIM)

# phase-2 animation
for ep in range(EPOCHS_ANIM):
    time.sleep(0.04)
    idx = EPOCHS_ANIM + ep
    progress_bar.progress((idx+1)/(EPOCHS_ANIM*2),
                          text=f"Obf→Master (fake) {ep+1}/{EPOCHS_ANIM}")
    loss_chart.add_rows({"loss": [fake_loss[idx]]})

# real training 2
model_obf_to_master = train_secret_regenerator(
    secret_str     = MASTER_SECRET,
    input_override = OBFUSC_SECRET,
    vocab          = VOCAB,
    epochs         = EPOCHS_ANIM)

progress_bar.empty()
st.success("✔ Training complete")

# ---------------------------------------------------------------------------
# 4️⃣  𝗦𝗮𝘃𝗲 𝗺𝗼𝗱𝗲𝗹𝘀 & 𝘀𝗲𝘀𝘀𝗶𝗼𝗻
# ---------------------------------------------------------------------------
master_path = f"models/master_to_obf_{SESSION_ID}.pt"
obf_path    = f"models/obf_to_master_{SESSION_ID}.pt"
torch.save(model_master_to_obf, master_path)
torch.save(model_obf_to_master,  obf_path)

session_json = f"sessions/{SESSION_ID}.json"
json.dump({
    "session_id"      : SESSION_ID,
    "master_secret"   : MASTER_SECRET,
    "obfuscated_secret": OBFUSC_SECRET,
    "model_master_path": master_path,
    "model_obf_path"  : obf_path
}, open(session_json, "w"), indent=2)

# ---------------------------------------------------------------------------
# 5️⃣  𝗦𝗶𝗱𝗲𝗯𝗮𝗿 — 𝘀𝘆𝘀𝘁𝗲𝗺 & 𝗺𝗼𝗱𝗲𝗹 𝘀𝘁𝗮𝘁𝘀
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("signal-2025-05-07-170157.png",
             width=280)  # replace with your local logo if needed
    st.markdown("## ⚙️ System")
    st.metric("CPU", f"{psutil.cpu_percent()} %")
    mem = psutil.virtual_memory()
    st.metric("Memory", f"{mem.percent} % of {round(mem.total/1e9,1)} GB")
    st.caption(f"Python {platform.python_version()} · Torch {torch.__version__}")

    st.divider()
    st.markdown("## 🏋️ Training")
    st.write("- Epochs/model : 50")
    st.write("- Fake loss    : 1.0 → 0.1")
    st.divider()
    st.markdown("## 📦 Model Paths")
    st.code(master_path)
    st.code(obf_path)
    st.divider()
    st.markdown("## 🔑 Secrets")
    st.text(f"MASTER : {MASTER_SECRET}")
    st.text(f"OBFSC  : {OBFUSC_SECRET}")
    st.divider()
    if st.button("🔄  Start New Demo"):
        st.rerun()

# ---------------------------------------------------------------------------
# 6️⃣  𝗠𝗮𝗶𝗻 — 𝟯-𝗰𝗼𝗹 𝗴𝗿𝗶𝗱 (master / clean obf / noisy obf)
# ---------------------------------------------------------------------------
noisy_obf = inject_noise(OBFUSC_SECRET, NOISE_RATIO, VOCAB)

col1, col2, col3 = st.columns([1,1,1])
with col1:
    st.subheader("🎯 Master Secret")
    st.code(MASTER_SECRET)
    st.pyplot(visualize_secret(MASTER_SECRET, "MASTER_SECRET"))
with col2:
    st.subheader("🕵️ Obfuscated Secret")
    st.code(OBFUSC_SECRET)
    st.pyplot(visualize_secret(OBFUSC_SECRET, "OBFUSC_SECRET"))

# with col3:
#     st.subheader(f"🔧 Noisy Obf ({NOISE_RATIO*100:.0f} %)")
#     st.code(noisy_obf)
#     st.pyplot(visualize_secret(noisy_obf, "Noisy Obf"))

with col3:
    st.subheader(f"🔧 Noisy Obf ({NOISE_RATIO*100:.0f} % noise)")
    st.code(noisy_obf)
    st.pyplot(visualize_noisy_diff(OBFUSC_SECRET, noisy_obf, "Noisy Obf – changed chars in red"))


# ---------------------------------------------------------------------------
# 7️⃣  𝗥𝗲𝗰𝗼𝗻𝘀𝘁𝗿𝘂𝗰𝘁 𝗠𝗮𝘀𝘁𝗲𝗿
# ---------------------------------------------------------------------------
tensor_in  = torch.tensor([[VOCAB.index(c)] for c in OBFUSC_SECRET], dtype=torch.long)
recon      = evaluate_secret_regenerator(model_obf_to_master, tensor_in, VOCAB)

st.markdown("### 🔁 Reconstructed MASTER_SECRET")
st.code(recon)
st.success("✅ Match" if recon == MASTER_SECRET else "❌ Mismatch")

# ---------------------------------------------------------------------------
# 8️⃣  𝗡𝗼𝗶𝘀𝗲 𝗱𝗲𝗹𝘁𝗮 & metadata
# ---------------------------------------------------------------------------
diffs = [f"{a}->{b}" if a!=b else "·" for a,b in zip(OBFUSC_SECRET, noisy_obf)]
st.markdown("### 👁️ Noise Character Diff")
st.write(" ".join(diffs))

st.markdown("---")
st.subheader("📄 Session Metadata")
st.json(json.load(open(session_json)))

# ---------------------------------------------------------------------------
# 9️⃣  𝗠𝗼𝗱𝗲𝗹 𝗠𝗮𝗻𝗮𝗴𝗲𝗿 𝗧𝗮𝗯
# ---------------------------------------------------------------------------
tabs = st.tabs(["🔍 Demo", "🗂 Model Manager"])
with tabs[1]:
    st.header("🗂 Stored Micro-Models")
    model_files = sorted(f for f in os.listdir("models") if f.startswith("master_to_obf"))
    if not model_files:
        st.info("No models saved yet.")
    for f in model_files:
        sid = f.split("_")[-1].split(".")[0]              # original session id
        obf_match = f.replace("master_to_obf", "obf_to_master")
        cols = st.columns([1,3,3,2])
        cols[0].markdown(f"**{sid}**")
        cols[1].code(f)
        cols[2].code(obf_match)
        if cols[3].button("▶ Run", key=f"run_{f}"):
            NEW_OBF  = generate_secret()
            noisy    = inject_noise(NEW_OBF, NOISE_RATIO, VOCAB)
            st.success(f"New OBFUSC_SECRET for {sid}: {NEW_OBF}")

            m1 = train_secret_regenerator(
                    secret_str     = NEW_OBF,
                    input_override = MASTER_SECRET,
                    vocab          = VOCAB,
                    epochs         = 50)
            m2 = train_secret_regenerator(
                    secret_str     = MASTER_SECRET,
                    input_override = NEW_OBF,
                    vocab          = VOCAB,
                    epochs         = 50)
            new_master_path = f"models/master_to_obf_{sid}_next.pt"
            new_obf_path    = f"models/obf_to_master_{sid}_next.pt"
            torch.save(m1, new_master_path)
            torch.save(m2, new_obf_path)

            sess_file = f"sessions/{sid}.json"
            sess = json.load(open(sess_file)) if os.path.exists(sess_file) else {}
            sess.update({"obfuscated_secret": NEW_OBF,
                         "model_master_path": new_master_path,
                         "model_obf_path"   : new_obf_path,
                         "timestamp"        : time.strftime("%Y-%m-%d %H:%M:%S")})
            json.dump(sess, open(sess_file, "w"), indent=2)
            st.balloons()
            st.rerun()
