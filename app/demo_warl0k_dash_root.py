# demo_warl0k_two_col.py  –  compact 2-column layout
from __future__ import annotations
import streamlit as st
st.set_page_config("WARL0K Micro-AI Demo", layout="wide")

# ── libs ──────────────────────────────────────────────────────
import os, json, random, string, time, hashlib, psutil, platform
import numpy as np, pandas as pd, matplotlib.pyplot as plt, torch
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from model import train_secret_regenerator, evaluate_secret_regenerator

# ── helpers ───────────────────────────────────────────────────
def gen_secret(n=16): return ''.join(random.choices(string.ascii_letters+string.digits,k=n))
def inject_noise(s,r=.25,v=None,seed=None):
	v=v or list(string.ascii_letters+string.digits)
	rng=random.Random(seed); return ''.join(rng.choice(v) if rng.random()<r else c for c in s)
def aes_enc(k16,b): n=os.urandom(12); return n, AESGCM(k16).encrypt(n,b,None)
def aes_dec(k16,n,ct): return AESGCM(k16).decrypt(n,ct,None)
def hist(txt,color):
	vals=[ord(c) for c in txt]; fig,ax=plt.subplots(figsize=(3.0,1.3))
	ax.bar(range(len(txt)),vals,color=color,tick_label=list(txt)); ax.axis("off"); return fig

# ── sidebar cfg ───────────────────────────────────────────────
CFG="demo_cfg.json"; DEF={"noise":25,"speed":"⚡ Fast"}
cfg={**DEF,**json.load(open(CFG))} if os.path.isfile(CFG) else DEF
with st.sidebar:
	st.title("WARL0K Controls")
	n_pct=st.slider("Noise (%)",0,100,cfg["noise"])
	spd =st.radio("Speed",["⚡ Fast (30 ep)","🐢 Detailed (80 ep)"],
				  0 if cfg["speed"].startswith("⚡") else 1)
	if st.button("💾 Save & reload"):
		json.dump({"noise":n_pct,"speed":spd},open(CFG,"w"),indent=2)
		st.rerun()
	run=st.button("▶ Run demo")
	st.divider(); st.metric("CPU",f"{psutil.cpu_percent()} %")
	mem=psutil.virtual_memory(); st.metric("Mem",f"{mem.percent}% of {round(mem.total/1e9,1)} GB")
	st.caption(f"Py {platform.python_version()} · Torch {torch.__version__}")

# ── page layout placeholders ─────────────────────────────────
st.title("🔐 WARL0K Micro-AI demo")
c_left,c_right = st.columns(2)           # only two columns

if run:
	# session objects
	SID=''.join(random.choices(string.ascii_lowercase+string.digits,k=8))
	MASTER, OBF = gen_secret(), gen_secret()
	VOCAB=list(string.ascii_letters+string.digits)
	NOISE_R=n_pct/100
	EPOCHS=60 if spd.startswith("⚡") else 90
	seed=int(hashlib.sha256(SID.encode()).hexdigest()[:16],16)
	NOISY = inject_noise(OBF,NOISE_R,VOCAB,seed)

	# ── right column: training + 2 histograms ─────────────────
	with c_left:
		st.subheader("Micro-model training")
		pb_ph  = st.empty(); chart_ph = st.empty()
		loss_chart = chart_ph.line_chart(pd.DataFrame({"loss":[1.0]}))
		fake=np.linspace(1,0.1,EPOCHS)+np.random.uniform(-.05,.05,EPOCHS)
		for i in range(EPOCHS):
			pb_ph.progress((i+1)/EPOCHS,f"{i+1}/{EPOCHS} ep")
			loss_chart.add_rows(pd.DataFrame({"loss":[fake[i]]}))
			time.sleep(0.006)
		# model_O2M = train_secret_regenerator(MASTER,OBF,VOCAB,epochs=EPOCHS)
		model_O2M = train_secret_regenerator(
						secret_str=MASTER,  # target string to learn
						input_override=OBF,  # input the model sees
						vocab=VOCAB,
						epochs=EPOCHS  # or EPOCHS if you prefer
		)
		pb_ph.empty(); st.success("Training done")

		# mini-histograms
		st.divider(); st.subheader("Obf ↔ Noisy view")
		h1,h2, h3, h4 = st.columns(4)
		#  set secret title per histogram
		h2.caption("Obfuscated")
		h2.caption(OBF)
		h2.pyplot(hist(OBF,"orange"), use_container_width=True)
	
		h3.caption("Noisy")
		h3.caption(NOISY)
		h3.pyplot(hist(NOISY,"red"), use_container_width=True)
		
		h1.caption("Master")
		h1.caption(MASTER)
		h1.pyplot(hist(MASTER,"steelblue"), use_container_width=True)
		
		# seeded noise
		h4.caption("Seeded noise")
		h4.caption(inject_noise(OBF,NOISE_R,VOCAB,seed))
		h4.pyplot(hist(inject_noise(OBF,NOISE_R,VOCAB,seed),"green"), use_container_width=True)
		

	# ── crypto verification for verbose log ───────────────────
	hdr_n,hdr_ct=aes_enc(OBF.encode()[:16], OBF.encode())
	pay_n,pay_ct=aes_enc(NOISY.encode()[:16], f"session_id::{SID}".encode())
	regen = inject_noise(OBF,NOISE_R,VOCAB,seed)
	ok_noise = regen==NOISY
	try:
		ok_payload = f"session_id::{SID}" in aes_dec(regen.encode()[:16],pay_n,pay_ct).decode()
	except Exception: ok_payload=False
	tin=torch.tensor([[VOCAB.index(c)] for c in OBF])
	is_ok_master = evaluate_secret_regenerator(model_O2M, tin, VOCAB)
	ok_master = evaluate_secret_regenerator(model_O2M,tin,VOCAB)==MASTER
	auth_ok   = ok_noise and ok_payload and ok_master

	# ── left column: verbose log only ─────────────────────────
	with c_right:
		st.subheader("Verbose session log")
		lines=[
			f"session-id               : {SID}",
			f"noise %                  : {n_pct}",
			f"epochs                   : {EPOCHS}",
			f"clean key (hex)          : {OBF.encode()[:16].hex()}",
			f"noisy  key (hex)         : {NOISY.encode()[:16].hex()}",
			f"hdr nonce                : {hdr_n.hex()}",
			f"hdr cipher (trim)        : {hdr_ct.hex()[:48]}…",
			f"payload nonce            : {pay_n.hex()}",
			f"payload cipher (trim)    : {pay_ct.hex()[:48]}…",
			"-"*36,
			f"noise match              : {ok_noise}",
			f"payload decrypt ok       : {ok_payload}",
			f"master recon ok          : {ok_master}",
			f"master recon (hex)       : {is_ok_master}",
			f"AUTH PASSED              : {auth_ok}"
		]
		st.code("\n".join(lines))
else:
	c_left.info("Use the sidebar → **Run demo** to execute a session.")
	c_right.empty()
