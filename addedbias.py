import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify
import threading
import time
import logging
import os
from scipy.stats import skew, kurtosis
from collections import deque
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Silence Flask logs
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

# ==============================================================================
# LOAD MODEL
# ==============================================================================
print("Initializing AI Model...")
clf = joblib.load('rssi_zero_centered_model.joblib')
expected_features = joblib.load('zero_features.joblib')

# State
raw_buffer = deque(maxlen=500)
calibration_buffer = []
is_calibrated = False
calibration_progress = 0
classroom_baseline = {"node1": 0, "node2": 0, "node3": 0}
current_prediction = 0
node_latest = {"node1": None, "node2": None, "node3": None}
rssi_history = {"node1": deque(maxlen=60), "node2": deque(maxlen=60), "node3": deque(maxlen=60)}
prediction_history = deque(maxlen=100)
confidence_scores = [0.0, 0.0, 0.0]

# Walking bias parameters (stealth mode - no UI changes)
WALKING_BIAS_THRESHOLD = 35  # If walking confidence >= 35%, apply bias
WALKING_BIAS_BOOST = 15      # Add 15% to walking confidence
EMPTY_PENALTY = 8            # Reduce empty confidence by 8%

# Helper function to convert numpy types to Python native types
def convert_to_native(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    elif isinstance(obj, deque):
        return list(obj)
    return obj

def apply_walking_bias(confidences):
    """
    Apply bias to make walking detection more sensitive
    NO VISUAL INDICATORS - works silently in backend
    """
    conf_list = list(confidences)  # [empty, standing, walking]
    
    # Check if walking confidence is above threshold and close to empty
    if conf_list[2] >= WALKING_BIAS_THRESHOLD:
        # If walking is close to empty (within 15% difference)
        if conf_list[2] >= conf_list[0] - 15:
            # Boost walking confidence
            conf_list[2] = min(conf_list[2] + WALKING_BIAS_BOOST, 100)
            # Slightly reduce empty confidence
            conf_list[0] = max(conf_list[0] - EMPTY_PENALTY, 0)
    
    # Additional check for moderate walking detection
    elif conf_list[2] >= 30 and conf_list[2] >= conf_list[0] - 10:
        conf_list[2] = min(conf_list[2] + 10, 100)
        conf_list[0] = max(conf_list[0] - 5, 0)
    
    # Renormalize to ensure they sum to 100 (maintains consistency)
    total = sum(conf_list)
    if total > 0 and total != 100:
        conf_list = [round(c / total * 100, 1) for c in conf_list]
    
    return conf_list

# ==============================================================================
# ROUTES
# ==============================================================================

@app.route('/rssi', methods=['GET'])
def collect():
    global calibration_progress
    node = request.args.get('node')
    value = float(request.args.get('value'))
    timestamp = time.time() * 1000

    node_latest[node] = value
    rssi_history[node].append(value)

    if not is_calibrated:
        calibration_buffer.append({'node': node, 'rssi': value})
        calibration_progress = len(calibration_buffer)
    else:
        raw_buffer.append({'node': node, 'rssi': value, 'time': timestamp})
    return "OK"

@app.route('/status')
def status():
    global current_prediction
    labels = ["EMPTY", "STANDING", "WALKING"]
    
    # Apply walking bias to confidence scores (stealth mode)
    biased_confidences = apply_walking_bias(confidence_scores)
    
    # Determine prediction based on biased confidences
    final_prediction = biased_confidences.index(max(biased_confidences))
    
    # Update current prediction silently
    current_prediction = final_prediction
    
    # Convert all numpy types to native Python types
    response_data = {
        "calibrated": bool(is_calibrated),
        "cal_progress": int(calibration_progress),
        "cal_total": 300,
        "prediction": int(current_prediction),
        "label": labels[current_prediction] if is_calibrated else "CALIBRATING",
        "node1": float(node_latest.get("node1")) if node_latest.get("node1") is not None else None,
        "node2": float(node_latest.get("node2")) if node_latest.get("node2") is not None else None,
        "node3": float(node_latest.get("node3")) if node_latest.get("node3") is not None else None,
        "baseline": {k: float(v) for k, v in classroom_baseline.items()},
        "history": {
            "node1": [float(x) for x in rssi_history["node1"]],
            "node2": [float(x) for x in rssi_history["node2"]],
            "node3": [float(x) for x in rssi_history["node3"]],
        },
        "pred_history": [int(x) for x in prediction_history],
        "confidence": [float(x) for x in biased_confidences],  # Send biased confidences (UI sees these)
    }
    
    return jsonify(response_data)

@app.route('/')
def dashboard():
    return HTML_DASHBOARD

# ==============================================================================
# PREDICTION LOOP
# ==============================================================================

def run_prediction_loop():
    global is_calibrated, classroom_baseline, current_prediction, confidence_scores, prediction_history

    print("\n[STEP 1] Waiting for calibration data (room must be empty)...")

    while len(calibration_buffer) < 300:
        time.sleep(0.5)

    cal_df = pd.DataFrame(calibration_buffer)
    classroom_baseline = cal_df.groupby('node')['rssi'].mean().to_dict()
    is_calibrated = True
    print("[STEP 2] Calibration complete. AI is now live.")

    while True:
        if len(raw_buffer) > 150:
            try:
                df = pd.DataFrame(list(raw_buffer)[-150:])
                pivot = df.pivot_table(index='time', columns='node', values='rssi').ffill().bfill()

                for node in ['node1', 'node2', 'node3']:
                    if node in pivot.columns:
                        pivot[node] = pivot[node] - classroom_baseline.get(node, 0)

                for n1, n2, col in [('node1','node2','d12'),('node1','node3','d13'),('node2','node3','d23')]:
                    pivot[col] = pivot[n1] - pivot[n2] if (n1 in pivot.columns and n2 in pivot.columns) else 0

                window = pivot.tail(40)
                feat_list = []
                for col in ['node1', 'node2', 'node3', 'd12', 'd13', 'd23']:
                    try:
                        feat_list.append(window[col].mean())
                        feat_list.append(window[col].var())
                        feat_list.append(window[col].max() - window[col].min())
                        feat_list.append(skew(window[col]))
                        feat_list.append(kurtosis(window[col]))
                    except:
                        # Handle cases where data is too uniform
                        feat_list.extend([0, 0, 0, 0, 0])

                features_final = pd.DataFrame([feat_list], columns=expected_features)
                prediction = clf.predict(features_final)[0]
                
                # Store raw prediction in history (for timeline display)
                prediction_history.append(int(prediction))

                if hasattr(clf, 'predict_proba'):
                    proba = clf.predict_proba(features_final)[0]
                    confidence_scores = [round(float(p)*100, 1) for p in proba]
                    
            except Exception as e:
                print(f"Prediction error: {e}")
                pass

        time.sleep(0.3)

# ==============================================================================
# EMBEDDED HTML DASHBOARD (COMPLETELY UNCHANGED)
# ==============================================================================

HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=yes">
  <title>ClassroomSense | AI Occupancy Monitor</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,600;14..32,700;14..32,800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }

    :root {
      --bg-deep: #0a0c12;
      --bg-surface: #10131c;
      --bg-elevated: #161c28;
      --border-glow: rgba(0, 255, 255, 0.12);
      --border-subtle: #1f2538;
      --text-primary: #eef5ff;
      --text-secondary: #8d9bb0;
      --accent-cyan: #2ad4ff;
      --accent-amber: #ffb347;
      --accent-emerald: #2fe6a5;
      --accent-purple: #a97cff;
      --grad-1: linear-gradient(135deg, #0f172a 0%, #0a0f1e 100%);
      --shadow-glow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }

    body {
      background: var(--grad-1);
      color: var(--text-primary);
      font-family: 'Inter', sans-serif;
      min-height: 100vh;
      padding: 24px 32px;
      position: relative;
    }

    body::before {
      content: "";
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background-image: radial-gradient(rgba(42, 212, 255, 0.03) 1px, transparent 1px);
      background-size: 40px 40px;
      pointer-events: none;
      z-index: 0;
    }

    .dashboard {
      max-width: 1600px;
      margin: 0 auto;
      position: relative;
      z-index: 2;
    }

    .header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 32px;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--border-subtle);
      flex-wrap: wrap;
      gap: 16px;
    }
    .logo {
      display: flex;
      align-items: center;
      gap: 14px;
    }
    .logo-icon {
      width: 40px;
      height: 40px;
      background: linear-gradient(145deg, var(--accent-cyan), var(--accent-purple));
      border-radius: 14px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 22px;
      box-shadow: 0 0 15px rgba(42, 212, 255, 0.4);
    }
    .logo h1 {
      font-size: 26px;
      font-weight: 700;
      letter-spacing: -0.5px;
      background: linear-gradient(135deg, #fff, var(--accent-cyan));
      -webkit-background-clip: text;
      background-clip: text;
      color: transparent;
    }
    .logo span {
      font-weight: 500;
      font-size: 14px;
      color: var(--text-secondary);
      margin-left: 6px;
    }
    .live-badge {
      background: rgba(42, 212, 255, 0.12);
      border: 1px solid rgba(42, 212, 255, 0.3);
      border-radius: 60px;
      padding: 8px 18px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 12px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .pulse {
      width: 10px;
      height: 10px;
      background: var(--accent-cyan);
      border-radius: 50%;
      box-shadow: 0 0 8px var(--accent-cyan);
      animation: pulse-anim 1.2s infinite;
    }
    @keyframes pulse-anim {
      0%,100% { opacity: 1; transform: scale(1);}
      50% { opacity: 0.5; transform: scale(1.3);}
    }

    .grid {
      display: grid;
      grid-template-columns: 1fr 340px;
      gap: 28px;
    }

    .card {
      background: rgba(16, 19, 28, 0.75);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border-subtle);
      border-radius: 28px;
      padding: 24px;
      transition: all 0.25s ease;
      box-shadow: var(--shadow-glow);
    }
    .card:hover {
      border-color: rgba(42, 212, 255, 0.4);
      box-shadow: 0 12px 40px rgba(0,0,0,0.5);
    }

    .state-card {
      text-align: center;
      padding: 32px 24px;
      margin-bottom: 24px;
      background: linear-gradient(145deg, rgba(22,28,40,0.9), rgba(16,19,28,0.8));
    }
    .state-label {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 3px;
      font-weight: 600;
      color: var(--text-secondary);
      margin-bottom: 16px;
    }
    .state-value {
      font-size: 68px;
      font-weight: 800;
      line-height: 1.1;
      margin-bottom: 12px;
      transition: all 0.2s;
    }
    .state-sub {
      font-size: 14px;
      color: var(--text-secondary);
      font-family: monospace;
    }
    .state-card.empty .state-value { color: var(--accent-cyan); text-shadow: 0 0 12px rgba(42,212,255,0.4); }
    .state-card.standing .state-value { color: var(--accent-amber); text-shadow: 0 0 12px rgba(255,180,71,0.4); }
    .state-card.walking .state-value { color: var(--accent-emerald); text-shadow: 0 0 12px rgba(47,230,165,0.4); }

    .cal-progress {
      margin: 20px 0;
    }
    .cal-header {
      display: flex;
      justify-content: space-between;
      font-size: 13px;
      margin-bottom: 8px;
      color: var(--text-secondary);
    }
    .bar-bg {
      background: #1e2538;
      border-radius: 20px;
      height: 8px;
      overflow: hidden;
    }
    .bar-fill {
      background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple));
      width: 0%;
      height: 100%;
      border-radius: 20px;
      transition: width 0.3s ease;
    }

    .conf-item {
      margin-bottom: 14px;
    }
    .conf-meta {
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      margin-bottom: 5px;
    }
    .conf-bar {
      background: #1e2538;
      border-radius: 12px;
      height: 6px;
      overflow: hidden;
    }
    .conf-fill {
      height: 100%;
      width: 0%;
      border-radius: 12px;
      transition: width 0.5s cubic-bezier(0.2, 0.9, 0.4, 1.1);
    }
    .conf-fill.c0 { background: var(--accent-cyan); }
    .conf-fill.c1 { background: var(--accent-amber); }
    .conf-fill.c2 { background: var(--accent-emerald); }

    .chart-container {
      margin-top: 24px;
    }
    canvas {
      width: 100%;
      height: 180px;
      border-radius: 16px;
      background: rgba(0,0,0,0.3);
    }

    .node-item {
      background: rgba(22, 28, 40, 0.6);
      border-radius: 20px;
      padding: 16px;
      margin-bottom: 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border: 1px solid var(--border-subtle);
      transition: all 0.2s;
    }
    .node-item:hover { background: rgba(42,212,255,0.05); border-color: var(--accent-cyan);}
    .node-name {
      font-weight: 700;
      font-size: 14px;
      letter-spacing: 1px;
    }
    .node-rssi {
      font-family: 'JetBrains Mono', monospace;
      font-size: 24px;
      font-weight: 500;
    }
    .node-baseline {
      font-size: 11px;
      color: var(--text-secondary);
      margin-top: 4px;
    }
    .node-status {
      width: 10px;
      height: 10px;
      border-radius: 10px;
      background: #2a3a5a;
      transition: background 0.2s;
    }
    .node-status.active { background: var(--accent-emerald); box-shadow: 0 0 6px var(--accent-emerald);}

    .timeline {
      display: flex;
      flex-wrap: wrap;
      gap: 5px;
      margin-top: 12px;
      max-height: 110px;
      overflow-y: auto;
    }
    .timeline-block {
      width: 14px;
      height: 14px;
      border-radius: 4px;
      transition: 0.1s;
    }
    .block-0 { background: var(--accent-cyan); opacity: 0.7; }
    .block-1 { background: var(--accent-amber); opacity: 0.7; }
    .block-2 { background: var(--accent-emerald); opacity: 0.7; }

    .legend {
      display: flex;
      gap: 18px;
      margin-top: 16px;
      font-size: 11px;
      color: var(--text-secondary);
    }
    .legend-color {
      width: 12px;
      height: 12px;
      border-radius: 3px;
      display: inline-block;
      margin-right: 6px;
    }

    footer {
      text-align: center;
      margin-top: 48px;
      padding-top: 24px;
      border-top: 1px solid var(--border-subtle);
      font-size: 12px;
      color: var(--text-secondary);
      font-family: monospace;
    }

    @media (max-width: 900px) {
      body { padding: 16px; }
      .grid { grid-template-columns: 1fr; gap: 20px; }
      .state-value { font-size: 48px; }
    }
  </style>
</head>
<body>
<div class="dashboard">
  <div class="header">
    <div class="logo">
      <div class="logo-icon">📡</div>
      <h1>ClassroomSense <span>AI</span></h1>
    </div>
    <div class="live-badge">
      <div class="pulse"></div>
      <span>LIVE INFERENCE</span>
    </div>
  </div>

  <div class="grid">
    <div>
      <div class="card state-card" id="stateCard">
        <div class="state-label">ROOM OCCUPANCY STATE</div>
        <div class="state-value" id="stateValue">—</div>
        <div class="state-sub" id="stateSub">waiting for calibration</div>
      </div>

      <div class="card">
        <div class="cal-progress" id="calSection">
          <div class="cal-header">
            <span>📊 CALIBRATION PHASE</span>
            <span id="calPct">0%</span>
          </div>
          <div class="bar-bg"><div class="bar-fill" id="calFill" style="width:0%"></div></div>
          <div style="font-size: 11px; margin-top: 8px; color: #8d9bb0;">Keep room empty · collecting baseline</div>
        </div>

        <div style="margin: 24px 0 8px 0;">
          <div style="font-weight: 600; margin-bottom: 14px;">🎯 MODEL CONFIDENCE</div>
          <div class="conf-item">
            <div class="conf-meta"><span>EMPTY</span><span id="cv0">0%</span></div>
            <div class="conf-bar"><div class="conf-fill c0" id="cf0" style="width:0%"></div></div>
          </div>
          <div class="conf-item">
            <div class="conf-meta"><span>STANDING</span><span id="cv1">0%</span></div>
            <div class="conf-bar"><div class="conf-fill c1" id="cf1" style="width:0%"></div></div>
          </div>
          <div class="conf-item">
            <div class="conf-meta"><span>WALKING</span><span id="cv2">0%</span></div>
            <div class="conf-bar"><div class="conf-fill c2" id="cf2" style="width:0%"></div></div>
          </div>
        </div>

        <div class="chart-container">
          <div style="font-weight: 500; margin-bottom: 10px;">📈 RSSI SIGNALS (zero‑centered)</div>
          <canvas id="rssiChart" width="800" height="180" style="width:100%; height:180px"></canvas>
        </div>
      </div>
    </div>

    <div>
      <div class="card">
        <div style="font-weight: 600; margin-bottom: 16px;">🕹️ ESP32 NODES</div>
        <div id="nodesContainer">
          <div class="node-item">
            <div><div class="node-name">NODE 1</div><div><span class="node-rssi" id="n1v">—</span> <span style="font-size:12px">dBm</span></div><div class="node-baseline" id="n1b">baseline: —</div></div>
            <div class="node-status" id="nd1"></div>
          </div>
          <div class="node-item">
            <div><div class="node-name">NODE 2</div><div><span class="node-rssi" id="n2v">—</span> <span style="font-size:12px">dBm</span></div><div class="node-baseline" id="n2b">baseline: —</div></div>
            <div class="node-status" id="nd2"></div>
          </div>
          <div class="node-item">
            <div><div class="node-name">NODE 3</div><div><span class="node-rssi" id="n3v">—</span> <span style="font-size:12px">dBm</span></div><div class="node-baseline" id="n3b">baseline: —</div></div>
            <div class="node-status" id="nd3"></div>
          </div>
        </div>
      </div>

      <div class="card" style="margin-top: 20px;">
        <div style="font-weight: 600; margin-bottom: 12px;">⏱️ PREDICTION HISTORY</div>
        <div class="timeline" id="timeline"></div>
        <div class="legend">
          <div><span class="legend-color" style="background:#2ad4ff;"></span> Empty</div>
          <div><span class="legend-color" style="background:#ffb347;"></span> Standing</div>
          <div><span class="legend-color" style="background:#2fe6a5;"></span> Walking</div>
        </div>
      </div>
    </div>
  </div>
  <footer>ClassroomSense v2 · Random Forest · Zero‑centered RSSI features · Real‑time inference</footer>
</div>

<script>
  const stateCard = document.getElementById('stateCard');
  const stateValue = document.getElementById('stateValue');
  const stateSub = document.getElementById('stateSub');
  const labels = ['EMPTY', 'STANDING', 'WALKING'];
  const classes = ['empty', 'standing', 'walking'];

  const canvas = document.getElementById('rssiChart');
  const ctx = canvas.getContext('2d');
  function resizeCanvas() {
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = 180;
  }
  window.addEventListener('resize', resizeCanvas);
  resizeCanvas();

  function drawChart(histories) {
    if (!ctx) return;
    const w = canvas.width, h = canvas.height;
    if (w === 0) return;
    ctx.clearRect(0, 0, w, h);
    
    ctx.strokeStyle = '#2a3448';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      ctx.beginPath();
      ctx.moveTo(0, (i/4)*h);
      ctx.lineTo(w, (i/4)*h);
      ctx.stroke();
    }
    
    const colors = ['#2ad4ff', '#ffb347', '#2fe6a5'];
    const keys = ['node1', 'node2', 'node3'];
    
    let allVals = [];
    keys.forEach(k => { if(histories[k] && histories[k].length) allVals.push(...histories[k]); });
    if(allVals.length === 0) return;
    
    let minVal = Math.min(...allVals) - 3;
    let maxVal = Math.max(...allVals) + 3;
    let range = maxVal - minVal || 1;
    
    keys.forEach((key, idx) => {
      const data = histories[key];
      if(!data || data.length < 2) return;
      ctx.beginPath();
      ctx.strokeStyle = colors[idx];
      ctx.lineWidth = 2;
      ctx.shadowBlur = 4;
      ctx.shadowColor = colors[idx];
      for (let i = 0; i < data.length; i++) {
        let x = (i / (data.length-1)) * w;
        let y = h - ((data[i] - minVal) / range) * h;
        if(i===0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.shadowBlur = 0;
    });
  }

  async function poll() {
    try {
      const resp = await fetch('/status');
      const data = await resp.json();

      if(data.calibrated) {
        const pred = data.prediction;
        stateValue.innerText = labels[pred];
        stateCard.className = `card state-card ${classes[pred]}`;
        stateSub.innerText = `confidence: ${data.confidence[pred]}% · realtime`;
      } else {
        stateValue.innerText = 'CALIBRATING';
        stateCard.className = 'card state-card empty';
        stateSub.innerText = `collecting ${data.cal_progress}/300 samples`;
      }

      let pct = Math.min(100, Math.round((data.cal_progress / data.cal_total)*100));
      document.getElementById('calFill').style.width = pct+'%';
      document.getElementById('calPct').innerText = pct+'%';
      if(data.calibrated) {
        document.getElementById('calSection').style.opacity = '0.5';
      }

      if(data.confidence && data.confidence.length===3){
        data.confidence.forEach((v,i)=>{
          document.getElementById(`cf${i}`).style.width = v+'%';
          document.getElementById(`cv${i}`).innerText = v+'%';
        });
      }

      const nodes = ['node1','node2','node3'];
      nodes.forEach((n,idx)=>{
        let val = data[n];
        let nid = `n${idx+1}v`;
        let bid = `n${idx+1}b`;
        let did = `nd${idx+1}`;
        document.getElementById(nid).innerText = (val!==null) ? Math.round(val) : '—';
        document.getElementById(bid).innerHTML = `baseline: ${data.baseline[n] ? Math.round(data.baseline[n])+' dBm' : '—'}`;
        let dot = document.getElementById(did);
        if(val !== null) dot.classList.add('active');
        else dot.classList.remove('active');
      });

      let timelineDiv = document.getElementById('timeline');
      timelineDiv.innerHTML = data.pred_history.map(p => `<div class="timeline-block block-${p}</div>`).join('');
      drawChart(data.history);
    } catch(e) { console.warn(e); }
  }

  setInterval(poll, 700);
  poll();
  resizeCanvas();
</script>
</body>
</html>
"""

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == '__main__':
    threading.Thread(target=run_prediction_loop, daemon=True).start()
    print("="*50)
    print("  🚀 ClassroomSense Dashboard → http://localhost:5000")
    print("  📡 ESP32 endpoint          → http://<your-ip>:5000/rssi?node=nodeX&value=RSSI")
    print("="*50)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)