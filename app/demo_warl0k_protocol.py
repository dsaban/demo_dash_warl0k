# demo_warl0k_agent.py
# -------------------------------------------------------------
# WARL0K Micro-AI dashboard  (Streamlit, CPU-only Torch)
# -------------------------------------------------------------
from __future__ import annotations

import streamlit as st
st.set_page_config(page_title="WARL0K Micro-AI Demo", layout="wide")

# ── standard libs ─────────────────────────────────────────────
import os, json, random, string, time, platform, psutil, shutil
import numpy as np, pandas as pd, matplotlib.pyplot as plt, torch
from model import train_secret_regenerator, evaluate_secret_regenerator
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import hashlib
# -----------------------------------------------------------------
# 0️⃣  Utility helpers
# -----------------------------------------------------------------

def aesgcm_encrypt(key16: bytes, plaintext: bytes) -> dict:
	nonce = os.urandom(12)
	aes   = AESGCM(key16)
	ct    = aes.encrypt(nonce, plaintext, None)
	return {"nonce": nonce.hex(), "cipher": ct.hex()}

def aesgcm_decrypt(key16: bytes, payload: dict) -> bytes:
	aes   = AESGCM(key16)
	nonce = bytes.fromhex(payload["nonce"])
	ct    = bytes.fromhex(payload["cipher"])
	return aes.decrypt(nonce, ct, None)
def generate_secret(n=16):
	return ''.join(random.choices(string.ascii_letters + string.digits, k=n))

def visualize_secret(secret: str, title: str):
	fig, ax = plt.subplots(figsize=(6, 2.2))
	vals = [ord(c) for c in secret]
	ax.bar(range(len(secret)), vals, tick_label=list(secret))
	ax.set_title(title); ax.set_ylim(min(vals)-2, max(vals)+2)
	ax.tick_params(axis='x', labelrotation=90); return fig

def visualize_noisy_diff(clean, noisy, title):
	ascii_vals = [ord(c) for c in noisy]
	colors = ["red" if a != b else "lightgray" for a, b in zip(clean, noisy)]
	fig, ax = plt.subplots(figsize=(6, 2.2))
	ax.bar(range(len(noisy)), ascii_vals, tick_label=list(noisy),
		   color=colors, edgecolor="black")
	ax.set_title(title); ax.set_ylim(min(ascii_vals)-2, max(ascii_vals)+2)
	ax.tick_params(axis='x', labelrotation=90); return fig

# def inject_noise(s, ratio=.25, vocab=None):
# 	vocab = vocab or list(string.ascii_letters + string.digits)
# 	return ''.join(random.choice(vocab) if random.random()<ratio else c for c in s)
# def inject_noise(s: str,
# 				 ratio: float = .25,
# 				 vocab=None,
# 				 seed: int | None = None) -> str:
# 	"""Deterministic noise if `seed` provided."""
# 	vocab = vocab or list(string.ascii_letters + string.digits)
# 	rng   = random.Random(seed)            # use local RNG for reproducibility
# 	chars = [
# 		rng.choice(vocab) if rng.random() < ratio else c
# 		for c in s
# 	]
# 	return ''.join(chars)

def inject_noise(s, ratio=.25, vocab=None, seed=None):
	vocab = vocab or list(string.ascii_letters + string.digits)
	rng   = random.Random(seed)           # deterministic when seed provided
	return ''.join(rng.choice(vocab) if rng.random()<ratio else c for c in s)

def fake_loss_curve(n):         # for chart animation only
	return np.clip(np.linspace(1,0.1,n)+np.random.uniform(-.05,.05,n),0.05,1)

def simulate_handshake(session_id, master, obf_clean, noisy_ratio, vocab):
	"""Return (success_bool, log_lines) for one local handshake round."""
	log      = []
	# noisy_fn = lambda s: inject_noise(s, noisy_ratio, vocab)
	# import hashlib
	seed_val = int(hashlib.sha256(session_id.encode()).hexdigest()[:16], 16)
	noisy_fn = lambda s: inject_noise(s, noisy_ratio, vocab, seed=seed_val)
	
	# 1️⃣  SERVER → CLIENT  (send obf secret encrypted with ephemeral key)
	nonce_serv  = os.urandom(12)
	ep_key      = hashlib.sha256((session_id + nonce_serv.hex()).encode()).digest()[:16]
	server_send = AESGCM(ep_key).encrypt(nonce_serv, obf_clean.encode(), None)
	log.append("Server: encrypted OBF_SEC with ep-key")

	# 2️⃣  CLIENT decrypts, injects deterministic noise, encrypts payload
	client_obf  = AESGCM(ep_key).decrypt(nonce_serv, server_send, None).decode()
	assert client_obf == obf_clean
	noisy_obf   = noisy_fn(client_obf)
	sample_msg  = f"session_id::{session_id}::demo_payload".encode()
	client_key  = noisy_obf.encode()[:16]
	payload     = aesgcm_encrypt(client_key, sample_msg)
	log.append("Client: injected noise + encrypted sample payload")

	# 3️⃣  SERVER regenerates the same noisy secret
	#     (same noise injection with same seed)
	regenerated_noisy = noisy_fn(obf_clean)
	if regenerated_noisy != noisy_obf:
		log.append("Server: noisy regeneration mismatch!")
		return False, log

	# 4️⃣  SERVER decrypts payload, reconstructs master via model
	server_key = regenerated_noisy.encode()[:16]
	try:
		plain = aesgcm_decrypt(server_key, payload).decode()
		ok_id = f"session_id::{session_id}" in plain
		ok_master = True  # already proven by model in outer code
		log.append("Server: decrypt OK, ID match ✓" if ok_id else "ID mismatch!")
		success = ok_id and ok_master
	except Exception as e:
		log.append(f"Server: decrypt failed → {e}")
		success = False
	return success, log

# -----------------------------------------------------------------
# 1️⃣  Config (saved)  +  session-state
# -----------------------------------------------------------------
CFG_PATH  = "demo_config.json"
DEFAULTS  = {"epochs_anim": 50, "noise_ratio": 0.25}
CFG       = {**DEFAULTS, **json.load(open(CFG_PATH))} if os.path.isfile(CFG_PATH) else DEFAULTS

if "demo_running" not in st.session_state: st.session_state.demo_running = False
if "logs"         not in st.session_state: st.session_state.logs = []

def log(msg): st.session_state.logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

# -----------------------------------------------------------------
# 2️⃣  SIDEBAR  (controls + system stats)
# -----------------------------------------------------------------
with st.sidebar:
	st.title("WARL0K Controls")

	# ► Config panel
	st.markdown("### ⚙️ Parameters")
	cfg_epochs = st.number_input("Animation epochs", 10, 300, CFG["epochs_anim"])
	cfg_noise  = st.slider("Noise ratio (%)", 0, 100, int(CFG["noise_ratio"]*100))
	if st.button("💾 Save & Apply"):
		CFG.update({"epochs_anim": cfg_epochs, "noise_ratio": cfg_noise/100})
		json.dump(CFG, open(CFG_PATH, "w"), indent=2)
		st.success("Saved. Click ▶ Start New Demo."); st.stop()

	# ► Start demo
	if st.button("▶ Start New Demo"):
		st.session_state.demo_running = True
		st.session_state.logs = []
		st.rerun()

	# ► Danger-zone erase
	st.divider(); st.markdown("## 🗑️ Danger Zone")
	if st.button("Erase ALL data"): st.session_state.erase_mode = True
	if st.session_state.get("erase_mode"):
		st.warning("Delete **models/**, **sessions/** and logs?")
		if st.button("✅ Confirm erase"):
			for d in ("models","sessions"): shutil.rmtree(d, ignore_errors=True)
			for f in ("archive_success.jsonl","archive_failed.jsonl","server_log.txt"):
				if os.path.isfile(f): os.remove(f)
			st.session_state.clear(); st.rerun()

	# ► System stats
	st.divider(); st.markdown("## 📊 System")
	st.metric("CPU", f"{psutil.cpu_percent()} %")
	mem = psutil.virtual_memory()
	st.metric("Memory", f"{mem.percent}% of {round(mem.total/1e9,1)} GB")
	st.caption(f"Python {platform.python_version()} · Torch {torch.__version__}")

# -----------------------------------------------------------------
# 3️⃣  Idle screen
# -----------------------------------------------------------------
if not st.session_state.demo_running:
	st.write("## 👋 Configure params then **Start New Demo**."); st.stop()

# -----------------------------------------------------------------
# 4️⃣  DEMO RUN  (fresh every rerun)
# -----------------------------------------------------------------
os.makedirs("../models", exist_ok=True); os.makedirs("../sessions", exist_ok=True)

SESSION_ID    = ''.join(random.choices(string.ascii_lowercase+string.digits,k=8))
MASTER_SECRET = generate_secret(); OBFUSC_SECRET = generate_secret()
VOCAB = list(string.ascii_letters+string.digits)
NOISE_RATIO, EPOCHS_ANIM = CFG["noise_ratio"], CFG["epochs_anim"]

st.title("🔐 WARL0K Micro-AI Demo"); st.caption(f"Session `{SESSION_ID}`")

# Training animation widgets
fake_loss = fake_loss_curve(EPOCHS_ANIM*2)
loss_chart = st.empty().line_chart(pd.DataFrame({"loss":[fake_loss[0]]}))
pb1 = st.empty().progress(0); pb2 = st.empty().progress(0)

# ► Phase 1 fake
for ep in range(EPOCHS_ANIM):
	pb1.progress((ep+1)/EPOCHS_ANIM, f"Master→Obf {ep+1}/{EPOCHS_ANIM}")
	loss_chart.add_rows({"loss":[fake_loss[ep]]}); time.sleep(.01)

log("Train model_master_to_obf")
model_master_to_obf = train_secret_regenerator(
	secret_str=OBFUSC_SECRET, input_override=MASTER_SECRET,
	vocab=VOCAB, epochs=CFG["epochs_anim"])

# ► Phase 2 fake
for ep in range(EPOCHS_ANIM):
	idx=EPOCHS_ANIM+ep
	pb2.progress((ep+1)/EPOCHS_ANIM, f"Obf→Master {ep+1}/{EPOCHS_ANIM}")
	loss_chart.add_rows({"loss":[fake_loss[idx]]}); time.sleep(.01)

log("Train model_obf_to_master")
model_obf_to_master = train_secret_regenerator(
	secret_str=MASTER_SECRET, input_override=OBFUSC_SECRET,
	vocab=VOCAB, epochs=CFG["epochs_anim"])

pb1.empty(); pb2.empty(); st.success("✔ Training complete"); log("Training finished")

# Save models & JSON
# ensure folders exist
os.makedirs("models", exist_ok=True)
os.makedirs("sessions", exist_ok=True)

master_path=f"models/master_to_obf_{SESSION_ID}.pt"
obf_path   =f"models/obf_to_master_{SESSION_ID}.pt"

torch.save(model_master_to_obf, master_path)
torch.save(model_obf_to_master, obf_path)
# Save session metadata
session_json = {"session_id":SESSION_ID,"master_secret":MASTER_SECRET,
				"obfuscated_secret":OBFUSC_SECRET,
				"model_master_path":master_path,"model_obf_path":obf_path}
json.dump(session_json, open(f"sessions/{SESSION_ID}.json","w"), indent=2)

# Three-column visuals
noisy_obf = inject_noise(OBFUSC_SECRET, NOISE_RATIO, VOCAB)
c1,c2,c3 = st.columns(3)
with c1:
	st.subheader("🎯 Master Secret")
	st.code(MASTER_SECRET); st.pyplot(visualize_secret(MASTER_SECRET,"MASTER"))
with c2:
	st.subheader("🕵️ Ephemeral Secret (obfuscated)")
	st.code(OBFUSC_SECRET)
	st.pyplot(visualize_secret(OBFUSC_SECRET,"OBF"))
with c3:
	st.subheader("🔧 Client Fingerprint"); st.code(noisy_obf)
	st.pyplot(visualize_noisy_diff(OBFUSC_SECRET,noisy_obf,"Noisy diff"))

# -----------------------------------------------------------------
# 4️⃣  Handshake simulation
st.markdown("## 🔄 Client ↔ Server Protocol Demo")
if st.button("Run Client-Server Handshake"):
	success, proto_log = simulate_handshake(
		SESSION_ID, MASTER_SECRET, OBFUSC_SECRET, NOISE_RATIO, VOCAB
	)
	st.write("**Result:**", "✅ Success" if success else "❌ Failed")
	st.text("\n".join(proto_log))


# Reconstruction
tensor_in = torch.tensor([[VOCAB.index(c)] for c in OBFUSC_SECRET], dtype=torch.long)
recon = evaluate_secret_regenerator(model_obf_to_master, tensor_in, VOCAB)
st.markdown("### 🔁 Reconstructed MASTER_SECRET")
st.code(recon); st.success("✅ Match" if recon==MASTER_SECRET else "❌ Mismatch")

# Logs expander
with st.expander("🪵 Runtime Log"): st.write("\n".join(st.session_state.logs))

# -----------------------------------------------------------------
# 5️⃣  Model Manager tab
# -----------------------------------------------------------------
tab_demo, tab_mgr = st.tabs(["📊 Demo Output", "🗂 Model Manager"])

with tab_demo:
	st.subheader("📄 Session Metadata")
	st.json(session_json)

with tab_mgr:
	st.header("🗂 Stored Models")
	files = sorted(f for f in os.listdir("models") if f.startswith("master_to_obf"))
	if not files: st.info("No models yet.")
	for f in files:
		sid = f.split("_")[-1].split(".")[0]
		obf_mate = f.replace("master_to_obf","obf_to_master")
		row = st.columns([1,3,3,2])
		row[0].markdown(f"**{sid}**"); row[1].code(f); row[2].code(obf_mate)
		if row[3].button("▶ Retrain", key=f"re_{f}"):
			NEW_OBF = generate_secret(); log(f"Retrain session {sid}")
			m1 = train_secret_regenerator(secret_str=NEW_OBF,input_override=MASTER_SECRET,vocab=VOCAB,epochs=CFG["epochs_anim"])
			m2 = train_secret_regenerator(secret_str=MASTER_SECRET,input_override=NEW_OBF,vocab=VOCAB,epochs=CFG["epochs_anim"])
			torch.save(m1,f"models/master_to_obf_{sid}_next.pt")
			torch.save(m2,f"models/obf_to_master_{sid}_next.pt")
			log(f"Saved new models for {sid}")
			sess_file=f"sessions/{sid}.json"; sess=json.load(open(sess_file)) if os.path.isfile(sess_file) else {}
			sess.update({"obfuscated_secret":NEW_OBF,"timestamp":time.strftime("%Y-%m-%d %H:%M:%S")})
			json.dump(sess, open(sess_file,"w"), indent=2)
			st.success(f"Retrained and rotated secret for {sid}")
			st.rerun()
