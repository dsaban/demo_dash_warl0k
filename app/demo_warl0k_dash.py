# demo_warl0k_storyboard.py
# -------------------------------------------------------------
# WARL0K Micro-AI Storyboard – client ↔ server fingerprint demo
# -------------------------------------------------------------
from __future__ import annotations
import streamlit as st

st.set_page_config(page_title="WARL0K Micro-AI Demo", layout="wide")

# ── std libs & deps ────────────────────────────────────────────
import os, json, random, string, time, hashlib, shutil, psutil, platform
import numpy as np, pandas as pd, matplotlib.pyplot as plt, torch
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from model import train_secret_regenerator, evaluate_secret_regenerator


# ── Helper utils ───────────────────────────────────────────────
def generate_secret(n=16) -> str:
	return ''.join(random.choices(string.ascii_letters + string.digits, k=n))


def visualize_secret(s: str, title: str, color="steelblue"):
	fig, ax = plt.subplots(figsize=(4, 1.6))
	vals = [ord(c) for c in s]
	ax.bar(range(len(s)), vals, tick_label=list(s), color=color)
	ax.set_ylim(min(vals) - 2, max(vals) + 2);
	ax.set_title(title, fontsize=10)
	ax.axis("off");
	return fig


def visualize_noisy(clean, noisy, title):
	vals = [ord(c) for c in noisy]
	colors = ["red" if a != b else "lightgray" for a, b in zip(clean, noisy)]
	fig, ax = plt.subplots(figsize=(4, 1.6))
	ax.bar(range(len(noisy)), vals, tick_label=list(noisy),
	       color=colors, edgecolor="black")
	ax.set_title(title, fontsize=10);
	ax.axis("off");
	return fig


def inject_noise(s, ratio=.25, vocab=None, seed=None):
	vocab = vocab or list(string.ascii_letters + string.digits)
	rng = random.Random(seed)
	return ''.join(rng.choice(vocab) if rng.random() < ratio else c for c in s)


def aesgcm_enc(key16, b):
	n = os.urandom(12);
	return n, AESGCM(key16).encrypt(n, b, None)
def aesgcm_dec(key16, n, ct):
	return AESGCM(key16).decrypt(n, ct, None)

def visualize_noisy_diff(clean, noisy, title):
	ascii_vals = [ord(c) for c in noisy]
	colors = ["red" if a != b else "lightgray" for a, b in zip(clean, noisy)]
	fig, ax = plt.subplots(figsize=(6, 2.2))
	ax.bar(range(len(noisy)), ascii_vals, tick_label=list(noisy),
		   color=colors, edgecolor="black")
	ax.set_title(title); ax.set_ylim(min(ascii_vals)-2, max(ascii_vals)+2)
	ax.tick_params(axis='x', labelrotation=90)
	return fig

# ── Config & session state ─────────────────────────────────────
CFG_FILE = "demo_cfg.json";
DEFAULTS = {"epochs": 50, "noise": 25, "speed": "⚡ Fast"}
cfg = {**DEFAULTS, **json.load(open(CFG_FILE))} if os.path.isfile(CFG_FILE) else DEFAULTS

if "stage" not in st.session_state: st.session_state.stage = 0  # 0-4
if "timeline" not in st.session_state: st.session_state.timeline = []
if "logs" not in st.session_state: st.session_state.logs = []


def log(msg): st.session_state.logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")


# ── Sidebar ----------------------------------------------------
with st.sidebar:
	st.title("WARL0K Control")
	st.markdown("### Parameters")
	epochs = st.selectbox("Demo speed", ["⚡ Fast", "🐢 Detailed"], index=0)
	noise = st.slider("Noise (%)", 0, 100, cfg["noise"])
	if st.button("💾 Save"):
		json.dump({"epochs": 50 if epochs == "⚡ Fast" else 120,
		           "noise": noise, "speed": epochs}, open(CFG_FILE, "w"))
		st.success("Saved – reload to apply")
	
	st.divider();
	st.metric("CPU", f"{psutil.cpu_percent()}%")
	mem = psutil.virtual_memory();
	st.metric("Mem", f"{mem.percent}% of {round(mem.total / 1e9, 1)} GB")
	st.caption(f"Py {platform.python_version()} · Torch {torch.__version__}")

# ── If first visit show welcome screen -------------------------
if st.session_state.stage == 0:
	st.title("🔐 WARL0K Micro-AI Storyboard")
	st.write("""
    This interactive storyboard walks through WARL0K’s **noise-fingerprint
    authentication** in four comic-strip steps:

    1. Generate secrets
    2. Train micro-models
    3. Client injects deterministic noise & encrypts data
    4. Server regenerates noise, decrypts, verifies, rotates secrets
    """)
	if st.button("▶ Begin"):
		st.session_state.stage = 1
		st.rerun()
	st.stop()

# ---------------------------------------------------------------
# Build one session’s artefacts once per run
# ---------------------------------------------------------------
if "built" not in st.session_state:
	SEED_ID = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
	MASTER = generate_secret()
	OBF = generate_secret()
	VOCAB = list(string.ascii_letters + string.digits)
	NOISE_R = noise / 100
	EPOCHS = 50 if epochs == "⚡ Fast" else 150
	st.session_state.update(dict(
		sid=SEED_ID, master=MASTER, obf=OBF, vocab=VOCAB,
		noise_r=NOISE_R, epochs=EPOCHS))
	st.session_state.built = True

sid = st.session_state.sid
MASTER = st.session_state.master
OBF = st.session_state.obf
NOISE_R = st.session_state.noise_r
EPOCHS = st.session_state.epochs
VOCAB = st.session_state.vocab

# ---------------------------------------------------------------
# ① Generate secrets (stage 1)
# ---------------------------------------------------------------
if st.session_state.stage == 1:
	st.header("① Generate Secrets")
	col1, col2 = st.columns(2)
	col1.subheader("Master Secret");
	col1.code(MASTER)
	col1.pyplot(visualize_secret(MASTER, "Master"))
	col2.subheader("Ephemeral Secret");
	col2.code(OBF)
	col2.pyplot(visualize_secret(OBF, "Ephemeral"))
	if st.button("Next ▸"):
		st.session_state.stage = 2;
		st.rerun()
	st.stop()

# ---------------------------------------------------------------
# ② Train Micro-Models
# ---------------------------------------------------------------
if st.session_state.stage == 2:
	st.header("② Train Micro-Models")
	ph_bar = st.empty();
	ph_loss = st.empty()
	losses = np.linspace(1, 0.1, EPOCHS) + np.random.uniform(-.05, .05, EPOCHS)
	for i in range(EPOCHS):
		ph_bar.progress((i + 1) / EPOCHS, text=f"Training {i + 1}/{EPOCHS}")
		ph_loss.line_chart(pd.DataFrame({"loss": [losses[i]]}))
		time.sleep(0.01)
	# model_M2O = train_secret_regenerator(OBF, MASTER, VOCAB)
	# model_O2M = train_secret_regenerator(MASTER, OBF, VOCAB)
	model_M2O = train_secret_regenerator(
		secret_str=OBF,  # target string to learn
		input_override=MASTER,  # input the model sees
		vocab=VOCAB,
		epochs=150  # or EPOCHS if you prefer
	)
	
	model_O2M = train_secret_regenerator(
		secret_str=MASTER,
		input_override=OBF,
		vocab=VOCAB,
		epochs=150
	)
	
	st.success("Models trained.")
	st.session_state.update(dict(model_M2O=model_M2O, model_O2M=model_O2M))
	if st.button("Next ▸"):
		st.session_state.stage = 3;
		st.rerun()
	st.stop()

# ---------------------------------------------------------------
# ③ Client injects noise & encrypts
# ---------------------------------------------------------------
if st.session_state.stage == 3:
	st.header("③ Client Fingerprint & Encrypt")
	seed = int(hashlib.sha256(sid.encode()).hexdigest()[:16], 16)
	noisy = inject_noise(OBF, NOISE_R, VOCAB, seed)
	colA, colB = st.columns(2)
	colA.pyplot(visualize_noisy_diff(OBF, noisy, "Deterministic noise"))
	colA.caption("Red = flipped char")
	sample = f"session_id::{sid}::demo_payload"
	key16 = noisy.encode()[:16]
	payload = aesgcm_enc(key16, sample.encode())
	colB.markdown("**Encrypted payload (hex, trimmed)**")
	colB.code(str(payload)[:90] + "…")
	if st.button("Next ▸"):
		st.session_state.noisy = noisy
		st.session_state.payload = payload
		st.session_state.stage = 4;
		st.rerun()
	st.stop()

# ---------------------------------------------------------------
# ④ Server verify, decrypt & rotate
# ---------------------------------------------------------------
if st.session_state.stage == 4:
	st.header("④ Server Verification & Rotation")
	seed = int(hashlib.sha256(sid.encode()).hexdigest()[:16], 16)
	regen_noisy = inject_noise(OBF, NOISE_R, VOCAB, seed)
	ok_noise = regen_noisy == st.session_state.noisy
	# try:
	# 	plain = aesgcm_dec(regen_noisy.encode()[:16],
	# 	                   bytes.fromhex(st.session_state.payload["nonce"]),
	# 	                   bytes.fromhex(st.session_state.payload["cipher"])
	# 	                   ).decode()
	# 	ok_id = f"session_id::{sid}" in plain
	# except Exception:
	# 	ok_id = False
	# try:
	# 	plain = aesgcm_dec(
	# 		regen_noisy.encode()[:16],
	# 		st.session_state.payload  # pass the dict directly
	# 	).decode()
	# 	ok_id = f"session_id::{sid}" in plain
	# except Exception:
	# 	ok_id = False
	# try:
	# 	# decrypt payload ...................................
	# 	plain = aesgcm_dec(regen_noisy.encode()[:16],st.session_state.payload).decode()
	#
	# 	ok_id = f"session_id::{sid}" in plain
	#
	# 	# --- NEW MASTER RECONSTRUCTION CHECK ---------------
	# 	# use the Obf→Master model trained in Stage-2
	# 	tensor_in = torch.tensor(
	# 		[[VOCAB.index(c)] for c in OBF], dtype=torch.long
	# 	)
	# 	reconstructed = evaluate_secret_regenerator(
	# 		st.session_state.model_O2M, tensor_in, VOCAB
	# 	)
	# 	ok_master = reconstructed == MASTER
	# except Exception as e:
	# 	ok_id = ok_master = False
	try:
		nonce, ct = st.session_state.payload  # ← unpack
		plain = aesgcm_dec(
			regen_noisy.encode()[:16],  # key
			nonce,  # nonce
			ct  # ciphertext
		).decode()
		
		ok_id = f"session_id::{sid}" in plain
		
		# Master-secret reconstruction check
		tensor_in = torch.tensor([[VOCAB.index(c)] for c in OBF], dtype=torch.long)
		reconstructed = evaluate_secret_regenerator(
			st.session_state.model_O2M, tensor_in, VOCAB
		)
		ok_master = reconstructed == MASTER
	except Exception as e:
		ok_id = ok_master = False
	
	# -------------------------------------------------------
	
	result = ok_noise and ok_id and ok_master
	# result = ok_noise and ok_id
	tile = st.empty()
	# tile.metric(label="", value="✅ Authentication PASSED" if result else "❌ Authentication FAILED")
	# tile.metric(label="result",
	#             value="✅ Authentication PASSED",
	#             label_visibility="collapsed")  # hides visually, keeps aria-label
	status_txt = "✅ Authentication PASSED" if result else "❌ Authentication FAILED"
	
	tile.metric(
		label="Auth Status",  # ← non-empty
		value=status_txt
		# or: label_visibility="collapsed"  # keep aria-label, hide visually
	)
	
	# st.write("• Noise match:", ok_noise);
	# st.write("• Session-ID match:", ok_id)
	st.write("• Noise match:", ok_noise)
	st.write("• Session-ID match:", ok_id)
	st.write("• Master Secret reconstructed:", ok_master)
	
	st.subheader("Log")
	st.text("Noise regenerated   ✓" if ok_noise else "Noise mismatch")
	if st.button("Restart demo"):
		st.session_state.clear()
		st.rerun()
	st.stop()
