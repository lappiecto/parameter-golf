#!/usr/bin/env python3
"""
Parameter Golf Lab — Experiment Dashboard Server
A lightweight local server for running and tracking parameter-golf experiments.
"""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_DIR = DASHBOARD_DIR.parent
EXPERIMENTS_FILE = DASHBOARD_DIR / "experiments.json"
VENV_PYTHON = PROJECT_DIR / ".venv" / "bin" / "python3"
TRAIN_SCRIPT = PROJECT_DIR / "train_gpt_mlx.py"

# Track running processes
active_runs: dict[str, dict] = {}


def load_experiments() -> list[dict]:
    if EXPERIMENTS_FILE.exists():
        return json.loads(EXPERIMENTS_FILE.read_text())
    return []


def save_experiments(experiments: list[dict]):
    EXPERIMENTS_FILE.write_text(json.dumps(experiments, indent=2))


def run_experiment(exp_id: str, config: dict):
    """Run a training experiment in a subprocess."""
    env = os.environ.copy()
    env["RUN_ID"] = exp_id

    # Map config keys to env vars
    env_map = {
        "iterations": "ITERATIONS",
        "num_layers": "NUM_LAYERS",
        "model_dim": "MODEL_DIM",
        "num_heads": "NUM_HEADS",
        "num_kv_heads": "NUM_KV_HEADS",
        "mlp_mult": "MLP_MULT",
        "vocab_size": "VOCAB_SIZE",
        "train_seq_len": "TRAIN_SEQ_LEN",
        "train_batch_tokens": "TRAIN_BATCH_TOKENS",
        "grad_accum_steps": "GRAD_ACCUM_STEPS",
        "tied_embed_lr": "TIED_EMBED_LR",
        "matrix_lr": "MATRIX_LR",
        "scalar_lr": "SCALAR_LR",
        "muon_momentum": "MUON_MOMENTUM",
        "muon_backend_steps": "MUON_BACKEND_STEPS",
        "warmup_steps": "WARMUP_STEPS",
        "warmdown_iters": "WARMDOWN_ITERS",
        "tie_embeddings": "TIE_EMBEDDINGS",
        "logit_softcap": "LOGIT_SOFTCAP",
        "rope_base": "ROPE_BASE",
        "qk_gain_init": "QK_GAIN_INIT",
        "seed": "SEED",
        "max_wallclock_seconds": "MAX_WALLCLOCK_SECONDS",
        "val_loss_every": "VAL_LOSS_EVERY",
        "val_batch_size": "VAL_BATCH_SIZE",
        "train_log_every": "TRAIN_LOG_EVERY",
        "grad_clip_norm": "GRAD_CLIP_NORM",
        "use_smeargate": "USE_SMEARGATE",
        "bigram_vocab_size": "BIGRAM_VOCAB_SIZE",
        "bigram_dim": "BIGRAM_DIM",
        "muon_weight_decay": "MUON_WEIGHT_DECAY",
        "eval_stride": "EVAL_STRIDE",
        "swa_enabled": "SWA_ENABLED",
        "swa_start_frac": "SWA_START_FRAC",
        "swa_every": "SWA_EVERY",
        "quant_bits_mlp": "QUANT_BITS_MLP",
        "quant_bits_attn": "QUANT_BITS_ATTN",
        "trigram_vocab_size": "TRIGRAM_VOCAB_SIZE",
        "trigram_dim": "TRIGRAM_DIM",
        "ema_enabled": "EMA_ENABLED",
        "ema_decay": "EMA_DECAY",
    }

    for key, env_key in env_map.items():
        if key in config:
            env[env_key] = str(config[key])

    log_lines = []
    start_time = time.time()

    try:
        # PYTHONUNBUFFERED ensures we get real-time output from the training script
        env["PYTHONUNBUFFERED"] = "1"

        proc = subprocess.Popen(
            [str(VENV_PYTHON), "-u", str(TRAIN_SCRIPT)],
            cwd=str(PROJECT_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # line-buffered
        )

        active_runs[exp_id]["pid"] = proc.pid
        active_runs[exp_id]["status"] = "running"

        for line in proc.stdout:
            line = line.rstrip()

            # Track validation progress separately — don't flood the log
            if line.startswith("val_progress:"):
                parts = line.split(":")[1].split("/")
                active_runs[exp_id]["val_progress"] = {
                    "current": int(parts[0]),
                    "total": int(parts[1]),
                }
                continue

            # Track warmup separately
            if line.startswith("warmup_step:"):
                active_runs[exp_id]["warmup"] = line.split(":")[1]
                continue

            log_lines.append(line)
            active_runs[exp_id]["log"] = log_lines[-50:]

            # Parse step logs for live progress
            if line.startswith("step:"):
                parts = line.split()
                step_info = parts[0].split(":")[1]
                current, total = step_info.split("/")
                active_runs[exp_id]["progress"] = {
                    "step": int(current),
                    "total": int(total),
                }
                for p in parts:
                    if p.startswith("train_loss:"):
                        active_runs[exp_id]["latest_loss"] = float(p.split(":")[1])
                    if p.startswith("val_bpb:"):
                        active_runs[exp_id]["latest_bpb"] = float(p.split(":")[1])
                    if p.startswith("tok_s:"):
                        active_runs[exp_id]["tok_s"] = int(p.split(":")[1])

            # Parse final results
            if line.startswith("final_int8_zlib_roundtrip "):
                for p in line.split():
                    if p.startswith("val_loss:"):
                        active_runs[exp_id]["final_val_loss"] = float(p.split(":")[1])
                    if p.startswith("val_bpb:"):
                        active_runs[exp_id]["final_bpb"] = float(p.split(":")[1])

        proc.wait()
        elapsed = time.time() - start_time

        # Save to experiments file
        experiments = load_experiments()
        result = {
            "id": exp_id,
            "config": config,
            "final_bpb": active_runs[exp_id].get("final_bpb"),
            "final_val_loss": active_runs[exp_id].get("final_val_loss"),
            "elapsed_seconds": round(elapsed, 1),
            "timestamp": datetime.now().isoformat(),
            "status": "completed" if proc.returncode == 0 else "failed",
            "exit_code": proc.returncode,
        }
        experiments.append(result)
        save_experiments(experiments)

        active_runs[exp_id]["status"] = result["status"]
        active_runs[exp_id]["elapsed"] = result["elapsed_seconds"]

    except Exception as e:
        active_runs[exp_id]["status"] = "error"
        active_runs[exp_id]["error"] = str(e)


def generate_advice(experiments: list[dict]) -> dict:
    """Analyse experiment history and suggest what to try next."""
    completed = [e for e in experiments if e.get("status") == "completed" and e.get("final_bpb")]
    completed.sort(key=lambda e: e["final_bpb"])

    # Current best config
    best = completed[0] if completed else None
    best_cfg = best["config"] if best else {}
    best_bpb = best["final_bpb"] if best else None

    # What the leaderboard winners use (known-good targets)
    LEADER_SETTINGS = {
        "mlp_mult": {"target": 3, "impact": "~0.01 BPB", "why": "Wider MLPs store more knowledge per layer. Every leaderboard winner uses 3x. This is the single biggest architectural improvement available."},
        "muon_momentum": {"target": 0.99, "impact": "measurable", "why": "Higher momentum means the optimiser carries more memory of past gradients, leading to smoother and more stable weight updates."},
        "matrix_lr": {"target": 0.02, "impact": "stability", "why": "Lower matrix learning rate prevents the big weight matrices from overshooting. Leaders all dropped from 0.04 to 0.02."},
        "num_layers": {"target": 10, "impact": "~0.003 BPB", "why": "An extra layer gives the model more depth to reason through. Going from 9 to 10 layers adds about 2.6M parameters."},
        "train_seq_len": {"target": 2048, "impact": "~0.02 BPB", "why": "Longer context lets the model see more of the text when predicting. This was one of the easiest and biggest wins on the leaderboard."},
        "warmdown_iters": {"target": 3000, "impact": "convergence", "why": "A longer warmdown gives the model more time to settle into a good minimum. Leaders use 3000 iterations of learning rate decay."},
        "grad_clip_norm": {"target": 0.3, "impact": "stability", "why": "Gradient clipping prevents rare large gradients from destabilising training. The current leader uses 0.3."},
        "iterations": {"target": 2000, "impact": "significant", "why": "More training iterations let the model learn more. 200 is a quick smoke test; 2000+ gives much more realistic results."},
    }

    suggestions = []
    analysis = []
    tried_params = {}  # Track what settings have been explored

    # Analyse what's been tried across all experiments
    for exp in completed:
        cfg = exp.get("config", {})
        for key in LEADER_SETTINGS:
            if key not in tried_params:
                tried_params[key] = set()
            tried_params[key].add(cfg.get(key))

    # Pairwise analysis: what changed between experiments and did it help?
    if len(completed) >= 2:
        analysis.append({
            "type": "summary",
            "title": f"You've run {len(completed)} experiment{'s' if len(completed) != 1 else ''}",
            "body": f"Best BPB so far: {best_bpb:.4f} (experiment '{best_cfg.get('name', best['id'])}')."
        })

        # Compare each experiment to the best
        for exp in completed[1:]:
            cfg = exp.get("config", {})
            bpb = exp["final_bpb"]
            diffs = []
            for key in LEADER_SETTINGS:
                if cfg.get(key) != best_cfg.get(key):
                    diffs.append(f"{key}: {cfg.get(key)} vs {best_cfg.get(key)} (best)")
            if diffs:
                delta = bpb - best_bpb
                direction = "worse" if delta > 0 else "better"
                analysis.append({
                    "type": "comparison",
                    "title": f"'{cfg.get('name', exp['id'])}' was {abs(delta):.4f} BPB {direction}",
                    "body": "Differences from your best run: " + "; ".join(diffs[:4]),
                })

    elif len(completed) == 1:
        analysis.append({
            "type": "summary",
            "title": "First experiment complete!",
            "body": f"BPB: {best_bpb:.4f}. This is your starting point. The suggestions below are ordered by expected impact — start from the top."
        })
    else:
        analysis.append({
            "type": "summary",
            "title": "No experiments yet",
            "body": "Run your first experiment to get personalised suggestions. Try the 'Quick Smoke' preset for a fast baseline."
        })

    # Generate suggestions based on what hasn't been tried or what differs from leaders
    if best_cfg:
        # Priority-ordered suggestions
        suggestion_defs = [
            {
                "param": "iterations",
                "condition": lambda c: c.get("iterations", 200) < 500,
                "priority": 1,
                "title": "Increase iterations to 500+",
                "preset": {"iterations": 500, "train_log_every": 100},
                "body": f"You're at {best_cfg.get('iterations', 200)} iterations. At this level, the model barely learns anything — it's like trying to learn a language in one afternoon. Bump to 500+ to see meaningful patterns emerge. Even 500 will take about 2-3 minutes locally.",
            },
            {
                "param": "mlp_mult",
                "condition": lambda c: c.get("mlp_mult", 2) < 3,
                "priority": 2,
                "title": "Widen the MLP to 3x",
                "preset": {"mlp_mult": 3},
                "body": LEADER_SETTINGS["mlp_mult"]["why"] + f" You're currently at {best_cfg.get('mlp_mult', 2)}x.",
                "impact": LEADER_SETTINGS["mlp_mult"]["impact"],
            },
            {
                "param": "train_seq_len",
                "condition": lambda c: c.get("train_seq_len", 1024) < 2048,
                "priority": 3,
                "title": "Double the sequence length to 2048",
                "preset": {"train_seq_len": 2048, "train_batch_tokens": max(int(best_cfg.get("train_batch_tokens", 8192)), 16384)},
                "body": LEADER_SETTINGS["train_seq_len"]["why"] + f" You're at {best_cfg.get('train_seq_len', 1024)}. (Batch tokens also increased to fit the longer sequences.)",
                "impact": LEADER_SETTINGS["train_seq_len"]["impact"],
            },
            {
                "param": "muon_momentum",
                "condition": lambda c: c.get("muon_momentum", 0.95) < 0.99,
                "priority": 4,
                "title": "Bump Muon momentum to 0.99",
                "preset": {"muon_momentum": 0.99},
                "body": LEADER_SETTINGS["muon_momentum"]["why"] + f" You're at {best_cfg.get('muon_momentum', 0.95)}.",
                "impact": LEADER_SETTINGS["muon_momentum"]["impact"],
            },
            {
                "param": "matrix_lr",
                "condition": lambda c: c.get("matrix_lr", 0.04) > 0.025,
                "priority": 5,
                "title": "Lower the matrix learning rate to 0.02",
                "preset": {"matrix_lr": 0.02},
                "body": LEADER_SETTINGS["matrix_lr"]["why"] + f" You're at {best_cfg.get('matrix_lr', 0.04)}.",
                "impact": LEADER_SETTINGS["matrix_lr"]["impact"],
            },
            {
                "param": "num_layers",
                "condition": lambda c: c.get("num_layers", 9) < 10,
                "priority": 6,
                "title": "Add a 10th layer",
                "preset": {"num_layers": 10},
                "body": LEADER_SETTINGS["num_layers"]["why"] + f" You have {best_cfg.get('num_layers', 9)} layers.",
                "impact": LEADER_SETTINGS["num_layers"]["impact"],
            },
            {
                "param": "warmdown_iters",
                "condition": lambda c: c.get("warmdown_iters", 1200) < 2000,
                "priority": 7,
                "title": "Increase warmdown to 3000",
                "preset": {"warmdown_iters": 3000},
                "body": LEADER_SETTINGS["warmdown_iters"]["why"] + f" You're at {best_cfg.get('warmdown_iters', 1200)}.",
                "impact": LEADER_SETTINGS["warmdown_iters"]["impact"],
            },
            {
                "param": "grad_clip_norm",
                "condition": lambda c: c.get("grad_clip_norm", 0) < 0.1,
                "priority": 8,
                "title": "Enable gradient clipping at 0.3",
                "preset": {"grad_clip_norm": 0.3},
                "body": LEADER_SETTINGS["grad_clip_norm"]["why"],
                "impact": LEADER_SETTINGS["grad_clip_norm"]["impact"],
            },
        ]

        for s in suggestion_defs:
            if s["condition"](best_cfg):
                # Check if this specific change has been tried before
                param = s["param"]
                target = s["preset"][param]
                already_tried = target in tried_params.get(param, set())

                suggestions.append({
                    "priority": s["priority"],
                    "title": s["title"],
                    "body": s["body"],
                    "impact": s.get("impact", ""),
                    "preset": s["preset"],
                    "already_tried": already_tried,
                    "param": param,
                })

        # Safety: ensure batch sizes work with the sequence length
        for s in suggestions:
            preset = s.get("preset", {})
            seq = preset.get("train_seq_len", best_cfg.get("train_seq_len", 1024))
            grad_accum = int(best_cfg.get("grad_accum_steps", 8))

            # Train batch must fit at least one sequence per microbatch
            batch = preset.get("train_batch_tokens", best_cfg.get("train_batch_tokens", 8192))
            min_train_batch = seq * grad_accum
            if batch < min_train_batch:
                preset["train_batch_tokens"] = min_train_batch

            # Val batch must fit at least one sequence per grad_accum slice
            val_batch = preset.get("val_batch_size", best_cfg.get("val_batch_size", 8192))
            min_val_batch = seq * grad_accum
            if val_batch < min_val_batch:
                preset["val_batch_size"] = min_val_batch

        # If everything's been tried from the known playbook, suggest combos
        if not suggestions:
            suggestions.append({
                "priority": 99,
                "title": "You've tried all the known levers!",
                "body": "Your config matches the leaderboard winners. To go further, try: (1) stack multiple improvements together using the 'Aggressive' preset, (2) increase iterations to 2000+ for more training, or (3) experiment with unusual settings — what happens with mlp_mult=4? Or num_layers=11? The frontier is yours to explore.",
                "impact": "unknown",
                "preset": {},
                "already_tried": False,
                "param": "",
            })

    return {
        "analysis": analysis,
        "suggestions": suggestions[:6],  # Top 6 suggestions
        "best_config": best_cfg,
        "best_bpb": best_bpb,
        "total_experiments": len(completed),
    }


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/":
            self.path = "/index.html"
            return super().do_GET()

        if parsed.path == "/api/experiments":
            self._json_response(load_experiments())
            return

        if parsed.path == "/api/active":
            # Return sanitised active run info
            safe = {}
            for k, v in active_runs.items():
                safe[k] = {
                    "status": v.get("status"),
                    "progress": v.get("progress"),
                    "latest_loss": v.get("latest_loss"),
                    "latest_bpb": v.get("latest_bpb"),
                    "tok_s": v.get("tok_s"),
                    "final_bpb": v.get("final_bpb"),
                    "log": v.get("log", [])[-30:],
                    "elapsed": v.get("elapsed"),
                    "val_progress": v.get("val_progress"),
                    "warmup": v.get("warmup"),
                }
            self._json_response(safe)
            return

        if parsed.path == "/api/leaderboard":
            exps = load_experiments()
            completed = [e for e in exps if e.get("status") == "completed" and e.get("final_bpb")]
            completed.sort(key=lambda e: e["final_bpb"])
            self._json_response(completed)
            return

        if parsed.path == "/api/advisor":
            self._json_response(generate_advice(load_experiments()))
            return

        if parsed.path == "/api/external_run":
            # Monitor a run launched outside the dashboard (e.g. via nohup)
            import re
            log_files = sorted(Path("/tmp").glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
            log_files = [f for f in log_files if any(kw in f.name for kw in ("sota", "beast", "param", "golf"))]
            if not log_files:
                log_files = sorted(PROJECT_DIR.joinpath("logs").glob("*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not log_files:
                self._json_response({"status": "none"})
                return
            log_path = log_files[0]
            lines = log_path.read_text().splitlines()
            # Parse latest state
            # Parse all step losses for the chart
            loss_history = []
            for line in lines:
                if line.startswith("step:") and "train_loss:" in line:
                    for p in line.split():
                        if p.startswith("train_loss:"):
                            try:
                                loss_history.append(float(p.split(":")[1]))
                            except ValueError:
                                pass
            result = {"status": "running", "log": lines[-30:], "progress": None, "latest_loss": None, "tok_s": None, "final_bpb": None, "val_progress": None, "loss_history": loss_history}
            for line in reversed(lines):
                if line.startswith("step:") and result["progress"] is None:
                    parts = line.split()
                    step_info = parts[0].split(":")[1]
                    cur, total = step_info.split("/")
                    result["progress"] = {"step": int(cur), "total": int(total)}
                    for p in parts:
                        if p.startswith("train_loss:"):
                            result["latest_loss"] = float(p.split(":")[1])
                        if p.startswith("tok_s:"):
                            result["tok_s"] = int(float(p.split(":")[1]))
                if line.startswith("val_progress:") and result["val_progress"] is None:
                    vparts = line.split(":")[1].split("/")
                    result["val_progress"] = {"current": int(vparts[0]), "total": int(vparts[1])}
                if line.startswith("final_int8_zlib_roundtrip "):
                    for p in line.split():
                        if p.startswith("val_bpb:"):
                            result["final_bpb"] = float(p.split(":")[1])
                    result["status"] = "completed"
                if line.startswith("swa:"):
                    result["swa_info"] = line
            # Check if process is still alive
            import subprocess as sp
            ps = sp.run(["pgrep", "-f", "train_gpt_mlx"], capture_output=True, text=True)
            if not ps.stdout.strip() and result["status"] != "completed":
                if any("Traceback" in l for l in lines[-20:]):
                    result["status"] = "failed"
                else:
                    result["status"] = "finished_no_save"
            # Filter out val_progress spam from log
            result["log"] = [l for l in result["log"] if not l.startswith("val_progress:")][-30:]
            self._json_response(result)
            return

        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/run":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            config = json.loads(body)

            name = config.pop("name", "experiment")
            exp_id = f"{name}_{int(time.time())}"

            active_runs[exp_id] = {"status": "starting", "config": config}

            thread = threading.Thread(target=run_experiment, args=(exp_id, config), daemon=True)
            thread.start()

            self._json_response({"id": exp_id, "status": "started"})
            return

        if parsed.path == "/api/stop":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)
            exp_id = data.get("id")
            if exp_id in active_runs and "pid" in active_runs[exp_id]:
                try:
                    os.kill(active_runs[exp_id]["pid"], signal.SIGTERM)
                    active_runs[exp_id]["status"] = "stopped"
                except ProcessLookupError:
                    pass
            self._json_response({"status": "stopped"})
            return

        if parsed.path == "/api/delete":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)
            exp_id = data.get("id")
            experiments = load_experiments()
            experiments = [e for e in experiments if e["id"] != exp_id]
            save_experiments(experiments)
            self._json_response({"status": "deleted"})
            return

        self.send_error(404)

    def _json_response(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Quieten the request logs
        pass


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8888
    server = HTTPServer(("127.0.0.1", port), DashboardHandler)
    print(f"\n  Parameter Golf Lab")
    print(f"  Dashboard: http://localhost:{port}")
    print(f"  Press Ctrl+C to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
