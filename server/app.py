"""
app.py — FastAPI server for Supply Chain Inventory Rebalancer.

Endpoints:
  POST /reset          — Reset environment for a task
  GET  /state          — Get current observation
  POST /step           — Execute an action
  GET  /grade          — Get current task grade
  GET  /history        — Full step-by-step log
  GET  /tasks          — List all tasks with metadata
  GET  /health         — Health check
  GET  /               — Serve dashboard HTML
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from server.environment import SupplyChainEnv
from server.models import Action, StepResult

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_env: Optional[SupplyChainEnv] = None
_current_task_id: int = 1
_history: list[dict] = []
_episode_start_time: float = 0.0
_total_reward: float = 0.0
_server_start: float = time.time()

TASK_META = {
    1: {
        "id": 1,
        "name": "Simple Restock",
        "difficulty": "Easy",
        "horizon": 5,
        "max_steps": 5,
        "success_threshold": 0.99,
        "description": "West warehouse has 0 SKU_A. Transfer ≥50 units from Central before advancing time.",
        "color": "#22c55e",
    },
    2: {
        "id": 2,
        "name": "Cost-Optimized Rebalancing",
        "difficulty": "Medium",
        "horizon": 5,
        "max_steps": 15,
        "success_threshold": 0.7,
        "description": "Mismatched inventory across 3 warehouses over 5 days. Fulfill demand, minimize costs.",
        "color": "#f59e0b",
    },
    3: {
        "id": 3,
        "name": "Supply Shock",
        "difficulty": "Hard",
        "horizon": 7,
        "max_steps": 25,
        "success_threshold": 0.6,
        "description": "On Day 3, Supplier 1 goes offline. Triage stock and maximize fulfilled demand.",
        "color": "#ef4444",
    },
}

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: int = 1
    seed: int = 42

class StepRequest(BaseModel):
    action: Action

# ---------------------------------------------------------------------------
# Lifespan — initialise env on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _env, _current_task_id, _episode_start_time
    _env = SupplyChainEnv(task_id=1, seed=42)
    _current_task_id = 1
    _episode_start_time = time.time()
    yield

app = FastAPI(
    title="Supply Chain Inventory Rebalancer",
    description="OpenEnv-compliant RL environment API for benchmarking LLM planning agents.",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_env() -> SupplyChainEnv:
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call POST /reset first.")
    return _env

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health():
    """Health check — always 200 if server is alive."""
    day = None
    if _env is not None:
        try:
            day = _env.state().current_day
        except Exception:
            pass
    return {
        "status": "ok",
        "env_initialized": _env is not None,
        "current_task_id": _current_task_id if _env else None,
        "current_day": day,
        "steps_taken": len(_history),
        "total_reward": round(_total_reward, 4),
        "uptime_seconds": round(time.time() - _server_start, 1),
    }


@app.get("/tasks", tags=["Environment"])
def list_tasks():
    """List all available tasks with metadata."""
    return {"tasks": list(TASK_META.values())}


@app.post("/reset", tags=["Environment"])
def reset(body: Optional[ResetRequest] = None):
    """Reset the environment for a given task_id and seed."""
    global _env, _current_task_id, _history, _episode_start_time, _total_reward

    if body is None:
        body = ResetRequest()

    if body.task_id not in TASK_META:
        raise HTTPException(
            status_code=422,
            detail=f"task_id must be 1, 2, or 3. Got {body.task_id}",
        )

    _env = SupplyChainEnv(task_id=body.task_id, seed=body.seed)
    obs = _env.reset()
    _current_task_id = body.task_id
    _history = []
    _episode_start_time = time.time()
    _total_reward = 0.0

    return {
        "task_id": body.task_id,
        "seed": body.seed,
        "task_name": TASK_META[body.task_id]["name"],
        "observation": obs.model_dump(),
        "message": f"Environment reset for Task {body.task_id}: {TASK_META[body.task_id]['name']}",
    }


@app.get("/state", tags=["Environment"])
def get_state():
    """Return current observation without mutating state."""
    env = _require_env()
    obs = env.state()
    return {
        "task_id": _current_task_id,
        "task_name": TASK_META[_current_task_id]["name"],
        "observation": obs.model_dump(),
        "step_number": len(_history),
        "total_reward": round(_total_reward, 4),
    }


@app.post("/step", tags=["Environment"])
def step(body: StepRequest):
    """Execute one action. Returns next observation, reward, done flag."""
    global _total_reward

    env = _require_env()
    obs_before = env.state()
    day_before = obs_before.current_day

    try:
        result: StepResult = env.step(body.action)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Step failed: {str(e)}")

    reward_total = result.reward.total
    _total_reward += reward_total

    step_record = {
        "step": len(_history) + 1,
        "day": day_before,
        "action": body.action.model_dump(),
        "reward": result.reward.model_dump(),
        "done": result.done,
        "info": result.info,
        "cumulative_reward": round(_total_reward, 4),
        "elapsed_seconds": round(time.time() - _episode_start_time, 2),
    }
    _history.append(step_record)

    return {
        "observation": result.observation.model_dump(),
        "reward": result.reward.model_dump(),
        "done": result.done,
        "info": result.info,
        "step_number": len(_history),
        "total_reward": round(_total_reward, 4),
    }


@app.get("/grade", tags=["Grading"])
def grade():
    """Return the current grade for the active task (0.0 – 1.0)."""
    env = _require_env()

    graders = {
        1: env.grade_task_1,
        2: env.grade_task_2,
        3: env.grade_task_3,
    }
    grade_result = graders[_current_task_id]()

    return {
        "task_id": _current_task_id,
        "task_name": TASK_META[_current_task_id]["name"],
        "difficulty": TASK_META[_current_task_id]["difficulty"],
        "success_threshold": TASK_META[_current_task_id]["success_threshold"],
        "score": grade_result["score"],
        "passed": grade_result["score"] >= TASK_META[_current_task_id]["success_threshold"],
        "details": grade_result,
        "total_reward": round(_total_reward, 4),
        "steps_taken": len(_history),
    }


@app.get("/history", tags=["Grading"])
def get_history():
    """Full step-by-step log of the current episode."""
    return {
        "task_id": _current_task_id,
        "task_name": TASK_META[_current_task_id]["name"],
        "steps_taken": len(_history),
        "total_reward": round(_total_reward, 4),
        "history": _history,
    }


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Supply Chain Rebalancer</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
:root{--bg:#0a0a0f;--surface:#111118;--surface2:#1a1a24;--border:#2a2a3a;--accent:#6ee7f7;--accent2:#a78bfa;--green:#22c55e;--amber:#f59e0b;--red:#ef4444;--text:#e2e8f0;--muted:#64748b;--mono:'Space Mono',monospace;--sans:'Syne',sans-serif}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:var(--sans);min-height:100vh;overflow-x:hidden}
body::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(110,231,247,.03) 1px,transparent 1px),linear-gradient(90deg,rgba(110,231,247,.03) 1px,transparent 1px);background-size:40px 40px;pointer-events:none;z-index:0}
.wrap{max-width:1400px;margin:0 auto;padding:24px;position:relative;z-index:1}
header{display:flex;align-items:center;justify-content:space-between;margin-bottom:32px;padding-bottom:20px;border-bottom:1px solid var(--border)}
.logo{display:flex;align-items:center;gap:12px}
.logo-icon{width:40px;height:40px;border-radius:10px;background:linear-gradient(135deg,var(--accent),var(--accent2));display:flex;align-items:center;justify-content:center;font-size:18px}
.logo h1{font-size:1.3rem;font-weight:800;letter-spacing:-.5px}
.logo p{font-size:.72rem;color:var(--muted);font-family:var(--mono)}
.pill{display:flex;align-items:center;gap:8px;padding:6px 14px;border-radius:999px;border:1px solid var(--border);background:var(--surface);font-size:.72rem;font-family:var(--mono)}
.dot{width:7px;height:7px;border-radius:50%;background:var(--green);animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.g3{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:24px}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:20px;transition:border-color .2s}
.card:hover{border-color:rgba(110,231,247,.3)}
.card-lbl{font-size:.68rem;font-family:var(--mono);color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:8px}
.card-val{font-size:2rem;font-weight:800;line-height:1}
.card-sub{font-size:.72rem;color:var(--muted);margin-top:4px;font-family:var(--mono)}
.tabs{display:flex;gap:10px;margin-bottom:24px}
.tab{flex:1;padding:14px 16px;border-radius:12px;border:1px solid var(--border);background:var(--surface);cursor:pointer;transition:all .2s;text-align:left}
.tab:hover{border-color:var(--accent)}
.tab.active{border-color:var(--accent);background:rgba(110,231,247,.06)}
.badge{display:inline-block;font-size:.62rem;font-family:var(--mono);padding:2px 8px;border-radius:4px;margin-bottom:6px;font-weight:700}
.easy{background:rgba(34,197,94,.15);color:var(--green)}
.med{background:rgba(245,158,11,.15);color:var(--amber)}
.hard{background:rgba(239,68,68,.15);color:var(--red)}
.tab h3{font-size:.88rem;font-weight:700;margin-bottom:4px}
.tab p{font-size:.7rem;color:var(--muted);line-height:1.4}
.panel{background:var(--surface);border:1px solid var(--border);border-radius:14px;overflow:hidden;margin-bottom:24px}
.ph{padding:14px 20px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between}
.pt{font-size:.82rem;font-weight:700;display:flex;align-items:center;gap:8px}
.pb{padding:20px}
.inv-g{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.wh{background:var(--surface2);border-radius:10px;padding:14px;border:1px solid var(--border)}
.wh-name{font-size:.68rem;font-family:var(--mono);color:var(--accent);text-transform:uppercase;letter-spacing:1px;margin-bottom:10px}
.sr{display:flex;justify-content:space-between;align-items:center;padding:4px 0;border-bottom:1px solid rgba(255,255,255,.04)}
.sr:last-child{border-bottom:none}
.sn{font-size:.7rem;font-family:var(--mono);color:var(--muted)}
.sq{font-size:.78rem;font-weight:700;font-family:var(--mono)}
.sq.lo{color:var(--red)}.sq.ok{color:var(--green)}
.bar{height:2px;background:var(--border);border-radius:2px;margin-top:2px}
.bar-f{height:100%;border-radius:2px;background:linear-gradient(90deg,var(--accent),var(--accent2));transition:width .4s}
.fg{margin-bottom:14px}
.fl{font-size:.7rem;font-family:var(--mono);color:var(--muted);margin-bottom:6px;display:block;text-transform:uppercase;letter-spacing:.5px}
select,input[type=number]{width:100%;padding:9px 12px;background:var(--surface2);border:1px solid var(--border);border-radius:8px;color:var(--text);font-family:var(--mono);font-size:.8rem;outline:none;transition:border-color .2s}
select:focus,input:focus{border-color:var(--accent)}
.tr-row{display:grid;grid-template-columns:1fr 1fr 1fr 1fr auto;gap:8px;align-items:end;margin-bottom:8px}
.btn{padding:9px 16px;border-radius:8px;border:none;font-family:var(--mono);font-size:.78rem;font-weight:700;cursor:pointer;transition:all .15s}
.btn-p{background:var(--accent);color:#000}.btn-p:hover{background:#a5f3fc;transform:translateY(-1px)}
.btn-g{background:transparent;border:1px solid var(--border);color:var(--text)}.btn-g:hover{border-color:var(--accent);color:var(--accent)}
.btn-d{background:rgba(239,68,68,.12);border:1px solid var(--red);color:var(--red)}.btn-d:hover{background:rgba(239,68,68,.22)}
.btn-sm{padding:5px 10px;font-size:.68rem}
.brow{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.tog{display:flex;align-items:center;gap:10px;padding:9px 14px;background:var(--surface2);border-radius:8px;border:1px solid var(--border);cursor:pointer}
.sw{width:34px;height:19px;background:var(--border);border-radius:9px;position:relative;transition:background .2s}
.sw.on{background:var(--accent)}
.kn{position:absolute;top:3px;left:3px;width:13px;height:13px;border-radius:50%;background:#fff;transition:left .2s}
.sw.on .kn{left:18px}
.log{max-height:320px;overflow-y:auto;font-family:var(--mono);font-size:.73rem}
.log::-webkit-scrollbar{width:3px}
.log::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
.le{padding:7px 10px;border-radius:7px;margin-bottom:5px;background:var(--surface2);border-left:3px solid var(--border);display:grid;grid-template-columns:55px 50px 1fr auto;gap:10px;align-items:center}
.le.rp{border-left-color:var(--green)}.le.rn{border-left-color:var(--red)}
.ls{color:var(--accent);font-weight:700}
.ld{color:var(--muted)}
.la{color:var(--text);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.lrw{font-weight:700;text-align:right}
.ring{width:120px;height:120px;position:relative;margin:0 auto 16px}
.ring svg{transform:rotate(-90deg)}
.ring-txt{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
.rn2{font-size:1.8rem;font-weight:800;font-family:var(--mono)}
.rl{font-size:.62rem;color:var(--muted);font-family:var(--mono)}
.rb-g{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.rb-i{background:var(--surface2);border-radius:8px;padding:10px 12px}
.rb-l{font-size:.65rem;color:var(--muted);font-family:var(--mono)}
.rb-v{font-size:.95rem;font-weight:700;font-family:var(--mono);margin-top:2px}
.alert{padding:9px 13px;border-radius:8px;border:1px solid;font-size:.75rem;font-family:var(--mono);margin-bottom:10px;display:flex;align-items:center;gap:8px}
.aw{background:rgba(245,158,11,.08);border-color:var(--amber);color:var(--amber)}
.ae{background:rgba(239,68,68,.08);border-color:var(--red);color:var(--red)}
.ai{background:rgba(110,231,247,.08);border-color:var(--accent);color:var(--accent)}
.ft{width:100%;border-collapse:collapse;font-size:.73rem;font-family:var(--mono)}
.ft th{padding:8px 10px;text-align:left;color:var(--muted);font-weight:400;border-bottom:1px solid var(--border)}
.ft td{padding:6px 10px;border-bottom:1px solid rgba(255,255,255,.03)}
.ft tr:last-child td{border-bottom:none}
.sep{height:1px;background:var(--border);margin:18px 0}
.tc{color:var(--accent)}.tg{color:var(--green)}.tr2{color:var(--red)}.ta{color:var(--amber)}.tm{color:var(--muted)}
#notif{position:fixed;bottom:24px;right:24px;padding:12px 18px;border-radius:10px;background:var(--surface);border:1px solid var(--accent);color:var(--accent);font-family:var(--mono);font-size:.75rem;z-index:100;transition:opacity .3s;opacity:0;pointer-events:none;max-width:300px}
#notif.show{opacity:1}
@media(max-width:900px){.g3,.g2,.inv-g,.tabs{grid-template-columns:1fr;flex-direction:column}.tr-row{grid-template-columns:1fr 1fr}}
</style>
</head>
<body>
<div class="wrap">

<header>
  <div class="logo">
    <div class="logo-icon">⚙</div>
    <div>
      <h1>Supply Chain Rebalancer</h1>
      <p>OpenEnv RL Benchmark &nbsp;·&nbsp; v1.0.0</p>
    </div>
  </div>
  <div class="pill">
    <div class="dot" id="sDot"></div>
    <span id="sTxt">Connecting...</span>
  </div>
</header>

<!-- KPI row -->
<div class="g3">
  <div class="card">
    <div class="card-lbl">Current Day</div>
    <div class="card-val tc" id="kDay">—</div>
    <div class="card-sub" id="kHorizon">/ — horizon</div>
  </div>
  <div class="card">
    <div class="card-lbl">Total Reward</div>
    <div class="card-val" id="kReward">—</div>
    <div class="card-sub" id="kSteps">0 steps taken</div>
  </div>
  <div class="card">
    <div class="card-lbl">Task Score</div>
    <div class="card-val tg" id="kScore">—</div>
    <div class="card-sub" id="kThreshold">threshold: —</div>
  </div>
</div>

<!-- Task tabs -->
<div class="tabs" id="taskTabs">
  <div class="tab active" data-id="1" onclick="selTask(1)">
    <div class="badge easy">EASY</div>
    <h3>Simple Restock</h3>
    <p>West=0 SKU_A. Transfer ≥50 from Central.</p>
  </div>
  <div class="tab" data-id="2" onclick="selTask(2)">
    <div class="badge med">MEDIUM</div>
    <h3>Cost Optimization</h3>
    <p>Minimize costs across mismatched inventory.</p>
  </div>
  <div class="tab" data-id="3" onclick="selTask(3)">
    <div class="badge hard">HARD</div>
    <h3>Supply Shock</h3>
    <p>Supplier 1 offline on Day 3. Survive 7 days.</p>
  </div>
</div>

<!-- Alerts -->
<div id="alertsRow"></div>

<!-- Inventory + Score -->
<div class="g2">
  <div class="panel">
    <div class="ph">
      <div class="pt">📦 Warehouse Inventory</div>
      <button class="btn btn-g btn-sm" onclick="doRefresh()">↻ Refresh</button>
    </div>
    <div class="pb">
      <div class="inv-g" id="invGrid">
        <div class="tm" style="font-size:.78rem;font-family:var(--mono)">Reset environment to begin.</div>
      </div>
    </div>
  </div>

  <div class="panel">
    <div class="ph">
      <div class="pt">🏆 Score &amp; Last Reward</div>
      <button class="btn btn-g btn-sm" onclick="doGrade()">↻ Grade</button>
    </div>
    <div class="pb">
      <div class="ring">
        <svg width="120" height="120" viewBox="0 0 120 120">
          <circle cx="60" cy="60" r="50" fill="none" stroke="#2a2a3a" stroke-width="10"/>
          <circle id="arc" cx="60" cy="60" r="50" fill="none" stroke="#22c55e" stroke-width="10"
            stroke-dasharray="314" stroke-dashoffset="314" stroke-linecap="round" style="transition:stroke-dashoffset .5s,stroke .3s"/>
        </svg>
        <div class="ring-txt">
          <div class="rn2" id="scoreNum">—</div>
          <div class="rl">SCORE</div>
        </div>
      </div>
      <div class="rb-g">
        <div class="rb-i"><div class="rb-l">Fulfilled</div><div class="rb-v tm" id="rb0">—</div></div>
        <div class="rb-i"><div class="rb-l">Stockout</div><div class="rb-v tm" id="rb1">—</div></div>
        <div class="rb-i"><div class="rb-l">Holding Cost</div><div class="rb-v tm" id="rb2">—</div></div>
        <div class="rb-i"><div class="rb-l">Shipping Cost</div><div class="rb-v tm" id="rb3">—</div></div>
        <div class="rb-i"><div class="rb-l">Balance</div><div class="rb-v tm" id="rb4">—</div></div>
        <div class="rb-i"><div class="rb-l">Forecast Align</div><div class="rb-v tm" id="rb5">—</div></div>
        <div class="rb-i"><div class="rb-l">Prevention</div><div class="rb-v tm" id="rb6">—</div></div>
        <div class="rb-i"><div class="rb-l">Invalid Pen.</div><div class="rb-v tm" id="rb7">—</div></div>
        <div class="rb-i" style="grid-column:span 2"><div class="rb-l">Step Total</div><div class="rb-v tm" id="rb8">—</div></div>
      </div>
    </div>
  </div>
</div>

<!-- Action panel -->
<div class="panel">
  <div class="ph">
    <div class="pt">⚡ Execute Action <span style="font-size:.68rem;color:var(--muted);font-family:var(--mono);margin-left:6px">POST /step</span></div>
  </div>
  <div class="pb">
    <div id="trList"></div>
    <button class="btn btn-g btn-sm" onclick="addTr()" style="margin-bottom:16px">+ Add Transfer</button>
    <div class="sep"></div>
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px">
      <div class="tog" onclick="toggleAdv()" id="togEl">
        <div class="sw on" id="swEl"><div class="kn"></div></div>
        <span style="font-family:var(--mono);font-size:.78rem">advance_time = <span id="advLbl">true</span></span>
      </div>
      <div class="brow">
        <button class="btn btn-g" onclick="clearTr()">Clear</button>
        <button class="btn btn-p" onclick="doStep()">▶ Execute Step</button>
      </div>
    </div>
  </div>
</div>

<!-- Forecast + History -->
<div class="g2">
  <div class="panel">
    <div class="ph">
      <div class="pt">📈 Demand Forecast</div>
      <span style="font-size:.68rem;font-family:var(--mono);color:var(--muted)">Next 3 days</span>
    </div>
    <div class="pb" style="padding:0">
      <div style="overflow-x:auto">
        <table class="ft">
          <thead><tr><th>Day</th><th>Warehouse</th><th>SKU</th><th>Qty</th></tr></thead>
          <tbody id="fcBody"><tr><td colspan="4" style="padding:14px;color:var(--muted)">No data yet.</td></tr></tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="panel">
    <div class="ph">
      <div class="pt">📋 Step History</div>
      <button class="btn btn-g btn-sm" onclick="doHistory()">↻ Load</button>
    </div>
    <div class="pb" style="padding:8px">
      <div class="log" id="histLog">
        <div class="tm" style="font-size:.75rem;font-family:var(--mono);padding:4px">No steps yet.</div>
      </div>
    </div>
  </div>
</div>

<!-- Environment control -->
<div class="panel">
  <div class="ph">
    <div class="pt">🔄 Environment Control <span style="font-size:.68rem;color:var(--muted);font-family:var(--mono);margin-left:6px">POST /reset</span></div>
  </div>
  <div class="pb">
    <div style="display:flex;gap:14px;align-items:flex-end;flex-wrap:wrap">
      <div class="fg" style="margin:0;flex:1;min-width:140px">
        <label class="fl">Task ID</label>
        <select id="rTask">
          <option value="1">1 — Simple Restock (Easy)</option>
          <option value="2">2 — Cost Optimization (Medium)</option>
          <option value="3">3 — Supply Shock (Hard)</option>
        </select>
      </div>
      <div class="fg" style="margin:0;min-width:110px">
        <label class="fl">Seed</label>
        <input type="number" id="rSeed" value="42" min="0" max="9999">
      </div>
      <button class="btn btn-d" onclick="doReset()">⟳ Reset Environment</button>
      <button class="btn btn-g" onclick="doGrade()">📊 Get Grade</button>
    </div>
  </div>
</div>

</div><!-- /wrap -->
<div id="notif"></div>

<script>
const BASE = window.location.origin;
const WHS = ['east','central','west'];
const SKUS = ['SKU_A','SKU_B','SKU_C','SKU_D','SKU_E'];
let adv = true, trCount = 0;

function notify(msg, type='info'){
  const el=document.getElementById('notif');
  el.textContent=msg;
  const c={info:'var(--accent)',success:'var(--green)',error:'var(--red)'};
  el.style.borderColor=el.style.color=c[type]||c.info;
  el.classList.add('show');
  setTimeout(()=>el.classList.remove('show'),3000);
}

async function api(method,path,body=null){
  const o={method,headers:{'Content-Type':'application/json'}};
  if(body)o.body=JSON.stringify(body);
  const r=await fetch(BASE+path,o);
  if(!r.ok){const e=await r.json().catch(()=>({detail:r.statusText}));throw new Error(e.detail||r.statusText);}
  return r.json();
}

async function pollHealth(){
  try{
    const h=await api('GET','/health');
    document.getElementById('sDot').style.background='var(--green)';
    document.getElementById('sTxt').textContent=`Online · Day ${h.current_day??'—'} · ${h.steps_taken} steps`;
  }catch{
    document.getElementById('sDot').style.background='var(--red)';
    document.getElementById('sTxt').textContent='Offline';
  }
}

function selTask(id){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelector(`[data-id="${id}"]`).classList.add('active');
  document.getElementById('rTask').value=id;
}

function renderInv(whs){
  const g=document.getElementById('invGrid');
  if(!whs?.length){g.innerHTML='<div class="tm">No data.</div>';return;}
  const mx=Math.max(...whs.flatMap(w=>Object.values(w.current_inventory)));
  g.innerHTML=whs.map(w=>`
    <div class="wh">
      <div class="wh-name">${w.warehouse_id}</div>
      ${SKUS.map(s=>{
        const q=w.current_inventory[s]??0,p=mx>0?q/mx*100:0,c=q<20?'lo':'ok';
        return`<div class="sr"><span class="sn">${s}</span><div style="text-align:right">
          <span class="sq ${c}">${q}</span>
          <div class="bar"><div class="bar-f" style="width:${p}%"></div></div>
        </div></div>`;
      }).join('')}
      ${w.incoming_shipments.length?`<div style="margin-top:8px;font-size:.63rem;color:var(--accent);font-family:var(--mono)">▲ ${w.incoming_shipments.length} inbound</div>`:''}
    </div>`).join('');
}

function renderFc(fc){
  const b=document.getElementById('fcBody');
  if(!fc?.length){b.innerHTML='<tr><td colspan="4" style="padding:12px;color:var(--muted)">No data.</td></tr>';return;}
  b.innerHTML=fc.slice(0,15).map(f=>`<tr><td class="tc">${f.day}</td><td>${f.warehouse_id}</td><td class="tm">${f.sku}</td><td style="font-weight:700">${f.quantity}</td></tr>`).join('');
}

function renderAlerts(al){
  const r=document.getElementById('alertsRow');
  r.innerHTML=al?.length?al.map(a=>`<div class="alert ae">🚨 ${a}</div>`).join(''):'';
}

function renderReward(rw){
  const f=v=>v>=0?`+${v.toFixed(2)}`:v.toFixed(2);
  const s=(id,v)=>{const e=document.getElementById(id);if(e){e.textContent=f(v);e.className='rb-v '+(v>=0?'tg':'tr2');}};
  s('rb0',rw.fulfilled_demand_reward??0);
  s('rb1',rw.stockout_penalty??0);
  s('rb2',rw.holding_cost_penalty??0);
  s('rb3',rw.shipping_cost_penalty??0);
  s('rb4',rw.inventory_balance_reward??0);
  s('rb5',rw.demand_forecast_accuracy_reward??0);
  s('rb6',rw.proactive_restocking_bonus??0);
  s('rb7',rw.invalid_action_penalty??0);
  const tot=document.getElementById('rb8');
  if(tot){const v=rw.total??0;tot.textContent=f(v);tot.className='rb-v '+(v>=0?'tg':'tr2');}
}

function setScore(s){
  const arc=document.getElementById('arc');
  arc.style.strokeDashoffset=314*(1-s);
  arc.style.stroke=s>=.7?'var(--green)':s>=.4?'var(--amber)':'var(--red)';
  document.getElementById('scoreNum').textContent=s.toFixed(2);
  document.getElementById('kScore').textContent=s.toFixed(2);
}

const TASKS={1:{horizon:5,threshold:1.0},2:{horizon:5,threshold:0.7},3:{horizon:7,threshold:0.6}};

function updateHorizon(taskId){
  const t=TASKS[taskId];
  if(t){
    document.getElementById('kHorizon').textContent=`/ ${t.horizon} horizon`;
    document.getElementById('kThreshold').textContent=`threshold: ${t.threshold}`;
  }
}

async function doRefresh(){
  try{
    const d=await api('GET','/state');
    const obs=d.observation;
    renderInv(obs.warehouses);
    renderFc(obs.demand_forecast);
    renderAlerts(obs.active_alerts);
    document.getElementById('kDay').textContent=obs.current_day;
    document.getElementById('kReward').textContent=d.total_reward.toFixed(2);
    document.getElementById('kSteps').textContent=`${d.step_number} steps taken`;
    updateHorizon(d.task_id);
  }catch(e){notify('Refresh failed: '+e.message,'error');}
}

async function doGrade(){
  try{
    const g=await api('GET','/grade');
    setScore(g.score);
    const msg=`Score ${g.score.toFixed(2)} — ${g.passed?'✓ PASSED':'✗ not yet passed'}`;
    notify(msg,g.passed?'success':'info');
  }catch(e){notify('Grade failed: '+e.message,'error');}
}

async function doHistory(){
  try{
    const d=await api('GET','/history');
    const log=document.getElementById('histLog');
    if(!d.history.length){log.innerHTML='<div class="tm" style="font-size:.75rem;font-family:var(--mono);padding:4px">No steps yet.</div>';return;}
    log.innerHTML=[...d.history].reverse().map(h=>{
      const tr=h.action.transfers.length?h.action.transfers.map(t=>`${t.quantity}×${t.sku}`).join(', '):'no transfer';
      const r=h.reward.total,cls=r>=0?'rp':'rn';
      return`<div class="le ${cls}"><span class="ls">S${h.step}</span><span class="ld">D${h.day}</span><span class="la">${tr}${h.action.advance_time?' +adv':''}</span><span class="lrw ${r>=0?'tg':'tr2'}">${r>=0?'+':''}${r.toFixed(1)}</span></div>`;
    }).join('');
    document.getElementById('kSteps').textContent=`${d.steps_taken} steps taken`;
    document.getElementById('kReward').textContent=d.total_reward.toFixed(2);
  }catch(e){notify('History failed: '+e.message,'error');}
}

function addTr(){
  trCount++;
  const id=`tr${trCount}`;
  const d=document.createElement('div');
  d.className='tr-row';d.id=id;
  d.innerHTML=`
    <div class="fg" style="margin:0"><label class="fl">Source</label><select name="src">${WHS.map(w=>`<option>${w}</option>`).join('')}</select></div>
    <div class="fg" style="margin:0"><label class="fl">Destination</label><select name="dst">${WHS.map((w,i)=>`<option ${i===1?'selected':''}>${w}</option>`).join('')}</select></div>
    <div class="fg" style="margin:0"><label class="fl">SKU</label><select name="sku">${SKUS.map(s=>`<option>${s}</option>`).join('')}</select></div>
    <div class="fg" style="margin:0"><label class="fl">Qty</label><input type="number" name="qty" value="50" min="1" onfocus="this.select()"></div>
    <button class="btn btn-d btn-sm" onclick="document.getElementById('${id}').remove()" style="margin-top:20px">✕</button>`;
  document.getElementById('trList').appendChild(d);
}

function clearTr(){document.getElementById('trList').innerHTML='';}

function toggleAdv(){
  adv=!adv;
  document.getElementById('swEl').classList.toggle('on',adv);
  document.getElementById('advLbl').textContent=adv?'true':'false';
}

async function doStep(){
  const rows=document.querySelectorAll('#trList .tr-row');
  const transfers=[];let ok=true;
  rows.forEach(row=>{
    const src=row.querySelector('[name=src]').value;
    const dst=row.querySelector('[name=dst]').value;
    const sku=row.querySelector('[name=sku]').value;
    const qty=parseInt(row.querySelector('[name=qty]').value);
    if(src===dst){notify('Source and destination must differ','error');ok=false;return;}
    if(!qty||qty<1){notify('Quantity must be ≥ 1','error');ok=false;return;}
    transfers.push({source_node:src,destination_node:dst,sku,quantity:qty});
  });
  if(!ok)return;
  try{
    const res=await api('POST','/step',{action:{transfers,advance_time:adv}});
    renderInv(res.observation.warehouses);
    renderFc(res.observation.demand_forecast);
    renderAlerts(res.observation.active_alerts);
    renderReward(res.reward);
    document.getElementById('kDay').textContent=res.observation.current_day;
    document.getElementById('kReward').textContent=res.total_reward.toFixed(2);
    document.getElementById('kSteps').textContent=`${res.step_number} steps taken`;
    if(res.info.violations?.length){notify('⚠ Invalid: '+res.info.violations[0],'error');}
    else{const r=res.reward.total;notify(`Step ${res.step_number} — reward: ${r>=0?'+':''}${r.toFixed(2)}`,r>=0?'success':'error');}
    if(res.done){await doGrade();notify('🏁 Episode complete! Check grade.','success');}
    await doHistory();
    clearTr();
  }catch(e){notify('Step failed: '+e.message,'error');}
}

async function doReset(){
  const task_id=parseInt(document.getElementById('rTask').value);
  const seed=parseInt(document.getElementById('rSeed').value)||42;
  try{
    const d=await api('POST','/reset',{task_id,seed});
    document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
    document.querySelector(`[data-id="${task_id}"]`).classList.add('active');
    renderInv(d.observation.warehouses);
    renderFc(d.observation.demand_forecast);
    renderAlerts(d.observation.active_alerts||[]);
    document.getElementById('kDay').textContent=d.observation.current_day;
    document.getElementById('kReward').textContent='0.00';
    document.getElementById('kSteps').textContent='0 steps taken';
    document.getElementById('scoreNum').textContent='—';
    document.getElementById('kScore').textContent='—';
    document.getElementById('arc').style.strokeDashoffset=314;
    document.getElementById('histLog').innerHTML='<div class="tm" style="font-size:.75rem;font-family:var(--mono);padding:4px">No steps yet.</div>';
    // Reset reward breakdown boxes
    ['rb0','rb1','rb2','rb3','rb4','rb5','rb6','rb7','rb8'].forEach(id=>{
      const el=document.getElementById(id);if(el){el.textContent='—';el.className='rb-v tm';}
    });
    updateHorizon(task_id);
    clearTr();
    notify(`Task ${task_id} reset — ${d.message}`,'success');
  }catch(e){notify('Reset failed: '+e.message,'error');}
}

(async()=>{
  await pollHealth();
  setInterval(pollHealth,10000);
  try{await doRefresh();}catch{}
})();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
def dashboard():
    """Serve the interactive control dashboard."""
    return HTMLResponse(content=DASHBOARD_HTML)

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()

