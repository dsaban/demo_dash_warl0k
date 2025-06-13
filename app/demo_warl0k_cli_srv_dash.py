# demo_warl0k_live.py
# -------------------------------------------------------------
# WARL0K Micro-AI – storyboard + real client/server animation
# -------------------------------------------------------------
from __future__ import annotations
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx
st.set_page_config(page_title="WARL0K Micro-AI Demo", layout="wide")

# ── std / deps ────────────────────────────────────────────────
import queue, threading, socket, os, json, random, string, time, hashlib
import numpy as np, pandas as pd, matplotlib.pyplot as plt, torch, psutil, platform
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from model import train_secret_regenerator, evaluate_secret_regenerator

# ① GLOBAL queue (no Streamlit dependency) 🔸
event_q: queue.Queue[str] = queue.Queue()

def emit(src: str, text: str):
    """Thread-safe log push."""
    event_q.put(f"[{src}] {text}")

# ── helpers ───────────────────────────────────────────────────
def gen_secret(n=16): return ''.join(random.choices(string.ascii_letters+string.digits,k=n))
def inject_noise(s,r=.25,v=None,seed=None):
    v = v or list(string.ascii_letters+string.digits)
    rng = random.Random(seed); return ''.join(rng.choice(v) if rng.random()<r else c for c in s)
def draw_hist(txt,color):
    vals=[ord(c) for c in txt]
    fig,ax=plt.subplots(figsize=(4,1.4))
    ax.bar(range(len(txt)),vals,color=color,tick_label=list(txt)); ax.axis("off"); return fig
def aes_enc(k16,b): n=os.urandom(12); return n,AESGCM(k16).encrypt(n,b,None)
def aes_dec(k16,n,ct): return AESGCM(k16).decrypt(n,ct,None)
# def emit(src:str,msg:str): st.session_state.event_q.put(f"[{src}] {msg}")
def emit(src: str, msg: str):
    # create the queue at first use (thread-safe enough for this demo)
    if "event_q" not in st.session_state:
        st.session_state.event_q = queue.Queue()
    # st.session_state.event_q.put(f"[{src}] {msg}")
    event_q.put(...)

# from streamlit.runtime.scriptrunner import add_script_run_ctx
# import threading

# def safe_start_server():
#     if "server_thread" in st.session_state:
#         thr = st.session_state.server_thread
#         if thr.is_alive():
#             return                      # already running
#     # else create a fresh one
#     thr = threading.Thread(target=server_loop,
#                            args=("127.0.0.1", 9999),
#                            daemon=True,
#                            name="WARL0K-server")
#     add_script_run_ctx(thr)             # optional, gives Streamlit context
#     thr.start()
#     st.session_state.server_thread = thr
#     emit("MAIN", "server thread started")
#
# safe_start_server()

# ── sidebar config persistence ───────────────────────────────

CFG="demo_cfg.json"; DEF={"noise":25,"speed":"⚡ Fast"}
cfg = {**DEF, **json.load(open(CFG))} if os.path.isfile(CFG) else DEF
with st.sidebar:
    st.title("WARL0K Controls")
    n_pct = st.slider("Noise (%)",0,100,cfg["noise"])
    speed = st.radio("Speed",["⚡ Fast (30 ep)","🐢 Detailed (80 ep)"],0 if cfg["speed"].startswith("⚡") else 1)
    if st.button("💾 Save & Reload"):
        json.dump({"noise":n_pct,"speed":"⚡ Fast" if speed.startswith("⚡") else "🐢 Detailed"},open(CFG,"w"),indent=2)
        st.rerun()
    st.divider(); st.metric("CPU",f"{psutil.cpu_percent()}%")
    mem=psutil.virtual_memory(); st.metric("Mem",f"{mem.percent}% of {round(mem.total/1e9,1)}GB")
    st.caption(f"Py {platform.python_version()} · Torch {torch.__version__}")

# ── session-wide objects --------------------------------------
# if "event_q" not in st.session_state: st.session_state.event_q = queue.Queue()

# ── global demo artefacts (fresh each reload) ------------------
SID   = ''.join(random.choices(string.ascii_lowercase+string.digits,k=8))
MASTER= gen_secret();  OBF = gen_secret()
VOCAB = list(string.ascii_letters+string.digits)
NOISE_R = cfg["noise"]/100
EPOCHS  = 30 if cfg["speed"].startswith("⚡") else 80
seed_val= int(hashlib.sha256(SID.encode()).hexdigest()[:16],16)
NOISY   = inject_noise(OBF,NOISE_R,VOCAB,seed_val)

# ── column layout ---------------------------------------------
st.title("🔐 WARL0K Storyboard – live socket demo")
st.caption(f"Session `{SID}` · Noise {cfg['noise']} % · {cfg['speed']}")

c1,c2,c3 = st.columns(3)

# ① secrets
with c1:
    st.subheader("① Secrets")
    st.code(MASTER); st.pyplot(draw_hist(MASTER,"steelblue"),use_container_width=True)
    st.code(OBF);    st.pyplot(draw_hist(OBF,"orange"),use_container_width=True)

# ② training
with c2:
    st.subheader("② Train micro-models")
    bar=st.empty(); loss_ph=st.empty()
    fake=np.linspace(1,0.1,EPOCHS)+np.random.uniform(-.05,.05,EPOCHS)
    loss_ch=loss_ph.line_chart(pd.DataFrame({"loss":[fake[0]]}))
    for i in range(EPOCHS):
        bar.progress((i+1)/EPOCHS,f"{i+1}/{EPOCHS}"); loss_ch.add_rows({"loss":[fake[i]]}); time.sleep(0.006)
    # m_M2O = train_secret_regenerator(OBF, MASTER, VOCAB, epochs=EPOCHS)
    # m_O2M = train_secret_regenerator(MASTER, OBF, VOCAB, epochs=EPOCHS)
    m_M2O = train_secret_regenerator(
	    		secret_str=OBF,  # target string to learn
	    		input_override=MASTER,  # input the model sees
	    		vocab=VOCAB,
	    		epochs=90  # or EPOCHS if you prefer
    )
    m_O2M = train_secret_regenerator(
	    		secret_str=MASTER,
	    		input_override=OBF,
	    		vocab=VOCAB,
	    		epochs=90
	    	)
    bar.empty(); st.success("models trained")

# ③ fingerprint + verify
with c3:
    st.subheader("③ Fingerprint & Verify (local)")
    st.code(NOISY); st.pyplot(draw_hist(NOISY,"red"),use_container_width=True)
    n,ct = aes_enc(NOISY.encode()[:16], f"session_id::{SID}".encode())
    regen = inject_noise(OBF,NOISE_R,VOCAB,seed_val)
    ok_noise = regen==NOISY
    try:
        plain=aes_dec(regen.encode()[:16],n,ct).decode(); ok_id=f"session_id::{SID}" in plain
    except Exception: ok_id=False
    tin=torch.tensor([[VOCAB.index(c)] for c in OBF]); recon=evaluate_secret_regenerator(m_O2M,tin,VOCAB)
    ok_master = recon==MASTER; local_ok = ok_noise and ok_id and ok_master
    st.metric("Local Auth", "✅" if local_ok else "❌")

# ── background SERVER ------------------------------------------
# def server_loop(host="127.0.0.1",port=9999):
#     srv=socket.socket(); srv.bind((host,port)); srv.listen()
#     emit("SERVER","listening 9999")
# def server_loop(host="127.0.0.1", port=9999):
#     srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # ← NEW
#     srv.bind((host, port))
#     srv.listen()
#     emit("SERVER", f"listening on {port}")
#     ...
#     while True:
#         conn,_=srv.accept(); emit("SERVER","client connected")
#         threading.Thread(target=handle_session,args=(conn,),daemon=True).start()

# --- server loop -------------------------------------------------
def server_loop(host="127.0.0.1", port=0):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen()
    emit("SERVER", f"listening on {port}")
    while True:
        conn, _ = srv.accept()
        emit("SERVER", "client connected")
        threading.Thread(target=handle_session,
                         args=(conn,), daemon=True).start()

# --- safe launcher  (define AFTER server_loop!) ------------------
# from streamlit.runtime.scriptrunner import add_script_run_ctx
def safe_start_server():
    if "server_thread" in st.session_state and st.session_state.server_thread.is_alive():
        return                                # already running
    thr = threading.Thread(target=server_loop,
                           args=("127.0.0.1", 9998),
                           daemon=True,
                           name="WARL0K-server")
    add_script_run_ctx(thr)                   # optional
    thr.start()
    st.session_state.server_thread = thr
    emit("MAIN", "server thread started")

# --- call once per rerun ----------------------------------------
safe_start_server()

def handle_session(conn):
    # 1 send encrypted clean OBF
    nonce,ct = aes_enc(OBF.encode()[:16], OBF.encode())
    msg=json.dumps({"sid":SID,"nonce":nonce.hex(),"ct":ct.hex()}).encode()
    conn.sendall(msg); emit("SERVER","→ encrypted OBF sent")
    # 2 recv client payload
    data=conn.recv(4096); emit("SERVER",f"← {len(data)}B recv")
    payload=json.loads(data.decode())
    try:
        plain=aes_dec(NOISY.encode()[:16],
                      bytes.fromhex(payload["nonce"]),
                      bytes.fromhex(payload["ct"])).decode()
        ok=f"session_id::{SID}" in plain
        emit("SERVER","payload decrypt ✅" if ok else "payload ID mismatch")
    except Exception as e:
        emit("SERVER",f"decrypt error {e}")
    conn.close(); emit("SERVER","session closed")

if "srv_thr" not in st.session_state:
    # threading.Thread(target=server_loop,daemon=True).start()
    # st.session_state.srv_thr=True
    thr = threading.Thread(target=server_loop, daemon=True)
    add_script_run_ctx(thr)  # ← grants context
    thr.start()

# run *once* but restart if the old thread died
if "server_thread" not in st.session_state or not st.session_state.server_thread.is_alive():
    thr = threading.Thread(target=server_loop, daemon=True, name="WARL0K-server")
    add_script_run_ctx(thr)                    # optional: gives Streamlit context
    thr.start()
    st.session_state.server_thread = thr
    st.write("🛰️  Server thread started")
else:
    st.write("🛰️  Server already running")


# ── CLIENT trigger ---------------------------------------------
def run_client():
    try:
        c=socket.socket(); c.connect(("127.0.0.1",9998))
        hdr=json.loads(c.recv(4096).decode()); emit("CLIENT","← hdr recv")
        key_clean=OBF.encode()[:16]
        obf_clean=aes_dec(key_clean,
                          bytes.fromhex(hdr["nonce"]),
                          bytes.fromhex(hdr["ct"])).decode()
        assert obf_clean==OBF
        noisy=NOISY
        msg=f"session_id::{SID}::demo".encode()
        n2,ct2=aes_enc(noisy.encode()[:16],msg)
        pay=json.dumps({"nonce":n2.hex(),"ct":ct2.hex()}).encode()
        c.sendall(pay); emit("CLIENT","→ payload sent")
        c.close(); emit("CLIENT","done")
    except Exception as e:
        emit("CLIENT",f"error {e}")

# if st.sidebar.button("▶ Run handshake"):
#     threading.Thread(target=run_client,daemon=True).start()

if st.sidebar.button("▶ Run handshake"):
    threading.Thread(target=run_client,
                     daemon=True,
                     name="WARL0K-client").start()

# ── live event animation ---------------------------------------
left, right = st.columns(2)
left.subheader("CLIENT"); right.subheader("SERVER")
ph_cli = left.code("…"); ph_srv = right.code("…")

# def drain_events():
#     buf_c, buf_s = [], []
#     while not event_q.empty():
#         line = event_q.get_nowait()
#         (buf_c if line.startswith("[CLIENT]") else buf_s).append(line)
#     ph_cli.code("\n".join(buf_c) or "…")
#     ph_srv.code("\n".join(buf_s) or "…")
def drain_events():
    buf_c, buf_s = [], []
    while not event_q.empty():
        raw = event_q.get_nowait()

        # ── safety guard ────────────────────────────────
        if not isinstance(raw, str):
            raw = str(raw)          # turn Ellipsis / None / obj → text
        # -------------------------------------------------

        (buf_c if raw.startswith("[CLIENT]") else buf_s).append(raw)

    ph_cli.code("\n".join(buf_c) or "…")
    ph_srv.code("\n".join(buf_s) or "…")

for _ in range(1):          # refresh 3 s
    drain_events()
    time.sleep(0.2)


