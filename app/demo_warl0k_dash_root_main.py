# demo_warl0k_two_col.py
# ============================================================== #
#  WARL0K Micro-AI demo – 2-col layout, model store & shard merge #
# ============================================================== #
from __future__ import annotations
import streamlit as st
st.set_page_config("WARL0K Micro-AI Demo", layout="wide")

# ── libs ────────────────────────────────────────────────────────
import os, json, random, string, time, hashlib, psutil, platform, shutil
import numpy as np, pandas as pd, matplotlib.pyplot as plt, torch
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from model import train_secret_regenerator, evaluate_secret_regenerator


# ── helpers ─────────────────────────────────────────────────────
APP_DIR = os.path.dirname(__file__) if "__file__" in globals() else "."
os.makedirs(os.path.join(APP_DIR,"models"),   exist_ok=True)
os.makedirs(os.path.join(APP_DIR,"sessions"), exist_ok=True)

def gen_secret(n=16): return ''.join(random.choices(string.ascii_letters+string.digits,k=n))
def inject_noise(s,r=.25,v=None,seed=None):
	v=v or list(string.ascii_letters+string.digits)
	rng=random.Random(seed); return ''.join(rng.choice(v) if rng.random()<r else c for c in s)
def aes_enc(k16,b): n=os.urandom(12); return n, AESGCM(k16).encrypt(n,b,None)
def aes_dec(k16,n,ct): return AESGCM(k16).decrypt(n,ct,None)
def hist(txt,color):
	vals=[ord(c) for c in txt]; fig,ax=plt.subplots(figsize=(3.0,1.2))
	ax.bar(range(len(txt)),vals,color=color,tick_label=list(txt)); ax.axis("off"); return fig

def save_session(meta:dict, m_path:str, o_path:str):
	json.dump(meta, open(os.path.join(APP_DIR,"sessions",f"{meta['sid']}.json"),"w"), indent=2)
	torch.save(meta["model_O2M"], m_path)

# def list_sessions() -> list[str]:
# 	return sorted(p[:-5] for p in os.listdir("sessions") if p.endswith(".json"))
#
# # def load_session(sid:str):
# # 	meta=json.load(open(os.path.join("sessions",f"{sid}.json")))
# # 	meta["model_O2M"]=torch.load(meta["model_path"])
# # 	return meta
# def load_session(sid: str):
# 	meta = json.load(open(os.path.join("sessions", f"{sid}.json")))
# 	meta["model_O2M"] = torch.load(meta["model_path"])
# 	return meta
# helper section – replace the two defs
def list_sessions() -> list[str]:
	sess_dir = os.path.join(APP_DIR, "sessions")
	os.makedirs(sess_dir, exist_ok=True)         # ✨ ensure it exists
	return sorted(p[:-5] for p in os.listdir(sess_dir) if p.endswith(".json"))

def load_session(sid: str):
	sess_dir = os.path.join(APP_DIR, "sessions")
	os.makedirs(sess_dir, exist_ok=True)         # ✨ ensure it exists
	meta = json.load(open(os.path.join(sess_dir, f"{sid}.json")))
	meta["model_O2M"] = torch.load(meta["model_path"])
	return meta



# ── sidebar persistent cfg ─────────────────────────────────────
CFG="demo_cfg.json"; DEF={"noise":25,"speed":"⚡ Fast"}
cfg={**DEF,**json.load(open(CFG))} if os.path.isfile(CFG) else DEF
with st.sidebar:
	st.title("WARL0K Controls")
	n_pct=st.slider("Noise (%)",0,100,cfg["noise"])
	spd =st.radio("Speed",["⚡ Fast (60 ep)","🐢 Detailed (90 ep)"],
				  0 if cfg["speed"].startswith("⚡") else 1)
	if st.button("💾 Save cfg"):
		json.dump({"noise":n_pct,"speed":spd},open(CFG,"w"),indent=2); st.experimental_rerun()

	run_new = st.button("▶ New random session")

	st.divider(); st.markdown("### 🔄 Saved sessions")
	sess_list=list_sessions()
	sel_sid = st.selectbox("Load session-id", ["—"]+sess_list, index=0)
	run_load = st.button("▶ Run loaded session")
	st.divider(); st.metric("CPU",f"{psutil.cpu_percent()} %")
	st.caption(f"Py {platform.python_version()} · Torch {torch.__version__}")

# ── decide mode ────────────────────────────────────────────────
mode = "idle"
if run_new:  mode="new"
elif run_load and sel_sid!="—": mode="load"

# ── layout placeholders ────────────────────────────────────────
c_left,c_right=st.columns(2)
log_lines: list[str] = []

# ────────────────────────────────────────────────────────────────
if mode=="idle":
	c_left.info("Choose **New random session** or load a stored id.")
	c_right.empty()

# ────────────────────────────────────────────────────────────────
else:
	# ▶ either build fresh or reload
	if mode=="new":
		sid=''.join(random.choices(string.ascii_lowercase+string.digits,k=8))
		master, obf = gen_secret(), gen_secret()
		noise_r = n_pct/100; epochs = 60 if spd.startswith("⚡") else 90
		vocab=list(string.ascii_letters+string.digits)
		seed=int(hashlib.sha256(sid.encode()).hexdigest()[:16],16)
		noisy=inject_noise(obf,noise_r,vocab,seed)

		# training animation in RIGHT column
		with c_left:
			st.subheader("Micro-model training")
			pb=st.empty(); chart=st.empty().line_chart(pd.DataFrame({"loss":[1.0]}))
			fake=np.linspace(1,0.1,epochs)+np.random.uniform(-.05,.05,epochs)
			for i in range(epochs):
				pb.progress((i+1)/epochs,f"{i+1}/{epochs} ep"); chart.add_rows({"loss":[fake[i]]}); time.sleep(0.006)
			# model_O2M = train_secret_regenerator(master,obf,vocab,epochs=epochs)
			model_O2M = train_secret_regenerator(
				secret_str=master,  # target string to learn
				input_override=obf,  # input the model sees
				vocab=vocab,
				epochs=epochs  # or EPOCHS if you prefer
			)
			pb.empty(); st.success("Training done")

			# 2 hists
			h1,h2 = st.columns(2)
			h1.caption("Obfuscated"); h1.pyplot(hist(obf,"orange"),use_container_width=True)
			h2.caption("Noisy");      h2.pyplot(hist(noisy,"red"),use_container_width=True)
		
		# ── after training finishes ─────────────────────────────
		
		# Save the model file first
		m_path = f"models/obf_to_master_{sid}.pt"
		# ✨ make sure the directory exists
		os.makedirs(os.path.dirname(m_path), exist_ok=True)
		torch.save(model_O2M, m_path)
		
		# Build a JSON-friendly dict  (NO model objects inside!)
		meta = {
			"sid": sid,
			"master": master,
			"obf": obf,
			"noise_r": noise_r,
			"seed": seed,
			"epochs": epochs,
			"model_path": m_path  # just the path
		}
		# json.dump(meta, open(f"sessions/{sid}.json", "w"), indent=2)
		sess_path = os.path.join("sessions", f"{sid}.json")
		
		# ✨ make sure sessions/ exists
		os.makedirs(os.path.dirname(sess_path), exist_ok=True)
		
		json.dump(meta, open(sess_path, "w"), indent=2)
	
	
	else:  # mode=="load"
		meta=load_session(sel_sid)
		sid=meta["sid"]; master=meta["master"]; obf=meta["obf"]
		noise_r=meta["noise_r"]; seed=meta["seed"]; epochs=meta["epochs"]
		vocab=list(string.ascii_letters+string.digits)
		noisy=inject_noise(obf,noise_r,vocab,seed)
		model_O2M=meta["model_O2M"]

		# re-assembly animation (RIGHT col)
		with c_left:
			st.subheader("Re-assembling master from 3 shards")
			shard1,shard2,shard3 = master[:5], master[5:10], master[10:]
			prog=st.empty(); text=st.empty()
			for pct in range(0,101,20):
				prog.progress(pct/100,f"{pct}%")
				if pct==0: text.write("Shard-1 received …")
				elif pct==40: text.write("Shard-2 received …")
				elif pct==80: text.write("Shard-3 received …")
				time.sleep(0.25)
			prog.empty(); text.success(f"Re-assembled: **{master}**")

			st.divider()
			h1,h2,h3,h4 = st.columns(4)
			h1.caption("Master")
			h1.pyplot(hist(master,"blue"),use_container_width=True)
			h2.caption("Obfuscated")
			h2.pyplot(hist(obf,"orange"),use_container_width=True)
			h3.caption("Noisy Fingerprint")
			h3.pyplot(hist(noisy,"red"),use_container_width=True)
			h4.caption("De-noised Fingerprint")
			h4.pyplot(hist(inject_noise(obf,noise_r,vocab,seed),"green"),use_container_width=True)
	
		
	# ── left column: fingerprint + crypto verification ───────────
	
	# crypto verification (common)
	hdr_n,hdr_ct=aes_enc(obf.encode()[:16],obf.encode())
	pay_n,pay_ct=aes_enc(noisy.encode()[:16],f"session_id::{sid}".encode())
	regen=inject_noise(obf,noise_r,vocab,seed)
	ok_noise=regen==noisy
	try:
		ok_payload=f"session_id::{sid}" in aes_dec(regen.encode()[:16],pay_n,pay_ct).decode()
	except Exception: ok_payload=False
	tin=torch.tensor([[vocab.index(c)] for c in obf])
	ok_master=evaluate_secret_regenerator(model_O2M,tin,vocab)==master
	auth_ok=ok_noise and ok_payload and ok_master

	# verbose log (LEFT column) under expander
	with c_right:
		with st.expander("📜 Verbose run log", expanded=True):
			st.code("\n".join([
				f"session-id               : {sid}",
				f"noise %                  : {noise_r*100:.0f}",
				f"epochs                   : {epochs}",
				f"clean key (hex)          : {obf.encode()[:16].hex()}",
				f"noisy  key (hex)         : {noisy.encode()[:16].hex()}",
				f"noisy fingerprint        : {noisy}",
				f"regen (de-noise)         : {regen}",
				f"hdr nonce                : {hdr_n.hex()}",
				f"hdr cipher (trim)        : {hdr_ct.hex()[:48]}…",
				f"payload nonce            : {pay_n.hex()}",
				f"payload cipher (trim)    : {pay_ct.hex()[:48]}…",
				"-"*38,
				f"noise match              : {ok_noise}",
				f"payload decrypt ok       : {ok_payload}",
				f"master recon ok          : {ok_master}",
				"-" * 38,
				f"AUTH PASSED              : {auth_ok}"
			]))

		if mode=="new":
			st.success(f"Session **{sid}** saved. Reload any time from sidebar.")
		else:
			st.info("Choose an action in the sidebar.")
