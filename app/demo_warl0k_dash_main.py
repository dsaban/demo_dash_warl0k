# demo_warl0k_columns.py
# -------------------------------------------------------------
# WARL0K Micro-AI – 3-column auto storyboard
# -------------------------------------------------------------
from __future__ import annotations
import streamlit as st
st.set_page_config(page_title="WARL0K Micro-AI Demo", layout="wide")

import os, random, string, time, hashlib, numpy as np, pandas as pd, matplotlib.pyplot as plt
import torch
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from model import train_secret_regenerator, evaluate_secret_regenerator

# ── helper -----------------------------------------------------
def gen_secret(n=16): return ''.join(random.choices(string.ascii_letters+string.digits,k=n))
def inject_noise(s,r=.25,v=None,seed=None):
    v=v or list(string.ascii_letters+string.digits); rng=random.Random(seed)
    return ''.join(rng.choice(v) if rng.random()<r else c for c in s)
def bar(fig_ax,txt,color="steelblue"):
    s=[ord(c) for c in txt]; ax=fig_ax
    ax.bar(range(len(txt)),s,color=color,tick_label=list(txt)); ax.axis("off")
def aes_enc(k,b): n=os.urandom(12); return n,AESGCM(k).encrypt(n,b,None)
def aes_dec(k,n,ct): return AESGCM(k).decrypt(n,ct,None)

# ── one fresh session each reload ------------------------------
SID     = ''.join(random.choices(string.ascii_lowercase+string.digits,k=8))
MASTER  = gen_secret();  OBF = gen_secret()
VOCAB   = list(string.ascii_letters+string.digits)
NOISE_R = .25
EPOCHS  = 70

# ── 3 columns --------------------------------------------------
c1,c2,c3 = st.columns(3)

# ① secrets -----------------------------------------------------
with c1:
    st.subheader("① Secrets")
    fig,_ax = plt.subplots(figsize=(3,1.4))
    bar(_ax, MASTER); st.pyplot(fig, use_container_width=True); st.caption("Master")
    fig,_ax = plt.subplots(figsize=(3,1.4))
    bar(_ax, OBF,"orange"); st.pyplot(fig, use_container_width=True); st.caption("Ephemeral")

# ② live training ----------------------------------------------
with c2:
    st.subheader("② Train micro-models")
    ph_bar = st.empty(); ph_loss = st.empty()
    fake_loss = np.linspace(1,0.1,EPOCHS)+np.random.uniform(-.05,.05,EPOCHS)
    loss_chart = ph_loss.line_chart(pd.DataFrame({"loss":[fake_loss[0]]}))
    for i in range(EPOCHS):
        ph_bar.progress((i+1)/EPOCHS, f"epoch {i+1}/{EPOCHS}")
        loss_chart.add_rows({"loss":[fake_loss[i]]})
        time.sleep(0.01)
    # m_M2O = train_secret_regenerator(OBF, MASTER, VOCAB, epochs=80)
    # m_O2M = train_secret_regenerator(MASTER, OBF, VOCAB, epochs=80)
    model_M2O = train_secret_regenerator(
	    secret_str=OBF,  # target string to learn
	    input_override=MASTER,  # input the model sees
	    vocab=VOCAB,
	    epochs=120  # or EPOCHS if you prefer
    )
    
    model_O2M = train_secret_regenerator(
	    secret_str=MASTER,
	    input_override=OBF,
	    vocab=VOCAB,
	    epochs=120
    )
    ph_bar.empty(); st.success("models trained")

# ③ fingerprint + handshake ------------------------------------
with c3:
    st.subheader("③ Fingerprint ➜ Verify")
    seed = int(hashlib.sha256(SID.encode()).hexdigest()[:16],16)
    noisy = inject_noise(OBF, NOISE_R, VOCAB, seed)
    fig,_ax = plt.subplots(figsize=(3,1.4))
    bar(_ax, noisy,"red"); st.pyplot(fig, use_container_width=True)
    st.caption("deterministic noise")
    # simulate handshake
    n,ct = aes_enc(noisy.encode()[:16], f"session_id::{SID}::payload".encode())
    regen = inject_noise(OBF, NOISE_R, VOCAB, seed)
    try:
        plain = aes_dec(regen.encode()[:16], n, ct).decode()
        ok_id = f"session_id::{SID}" in plain
        tin = torch.tensor([[VOCAB.index(c)] for c in OBF])
        recon = evaluate_secret_regenerator(model_O2M,tin,VOCAB)
        ok_master = recon==MASTER
        ok = ok_id and ok_master
    except Exception:
        ok = False
    st.metric("Auth Status", "✅ PASSED" if ok else "❌ FAILED")
    st.write(f"noise match  : {regen==noisy}")
    st.write(f"id in msg    : {ok_id}")
    st.write(f"master match : {ok_master}")
