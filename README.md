# UA-VLA-IL: Uncertainty-Aware Vision-Language-Action Imitation Learning

<p align="center">
  <a href="https://github.com/Saibhargav1208/UA_VLA_IL">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" />
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/built%20with-Python3-red.svg" />
  </a>
  <a href="https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct">
    <img src="https://img.shields.io/badge/VLM-Qwen2--VL--2B-blue.svg" />
  </a>
</p>

Built on [Uncertainty-Aware Deployment of Pre-trained Language-Conditioned Imitation Learning Policies](https://arxiv.org/abs/2403.18222) (Bucher et al., IROS 2024)
by **Rongali Sai Bhargav** — Honda R&D | IIT Bombay (B.Tech 2024)

---

## What is UA-VLA-IL?

The original paper calibrates pre-trained robot manipulation policies (PerAct, RVT, CLIPort) using temperature scaling and a fixed neighborhood threshold ω for uncertainty-aware action selection. Both T and ω are **fixed scalars** — the same value regardless of how cluttered the scene is or how precise the task requires the robot to be.

**UA-VLA-IL replaces both fixed values with VLM-predicted adaptive values**, conditioned on the current visual observation and language instruction — zero additional training required.

| | Original Paper | UA-VLA-IL |
|---|---|---|
| Temperature T | Single scalar, learned on 25 demos | `T_base + α × complexity(obs)` — adaptive per scene |
| Neighborhood ω | Fixed hand-designed threshold | `ω_base × (1 − precision(task) + ε)` — adaptive per task |
| VLM backbone | None | Qwen2-VL-2B-Instruct |
| Extra training | 25 demo calibration required | Zero-shot via VLM |
| Models supported | PerAct, RVT, CLIPort | PerAct, RVT, CLIPort |

---

## Why This Matters

**The problem with fixed T:**
A cluttered scene with many similar objects needs aggressive smoothing (high T) to avoid picking distractor spikes. A clean scene needs minimal smoothing (low T) to preserve action precision. One fixed T cannot be right for both.

**The problem with fixed ω:**
"Insert the peg into the hole" needs a tight neighborhood (small ω) — precision matters. "Put the block somewhere in the box" can use a wide neighborhood (large ω) — approximate placement is fine. One fixed ω cannot serve both.

**UA-VLA-IL asks Qwen2-VL-2B two questions per step:**
1. *"How visually complex is this scene? 0–1"* → sets T
2. *"How precisely must the robot act for this task? 0–1"* → sets ω

---

## Architecture

```
Current observation (RGB) + Language instruction
              |
              v
      +----------------+
      |  Qwen2-VL-2B   |  (runs every cache_steps steps)
      +-------+--------+
              |
    +---------+---------+
    |                   |
    v                   v
complexity(obs)    precision(task)
    |                   |
    v                   v
T = T_base           ω = ω_base ×
  + α × complexity     (1 − precision + ε)
    |                   |
    v                   v
Override              Override
calib_scaler.T     action_selection.τ
    |
    v
Model logits → scale by T → softmax → neighborhood(ω) → best action
```

---

## New Files

| File | Description |
|---|---|
| `vla_cal/vla_calibrator.py` | `VLACalibrator` — predicts adaptive T and ω. Core formulas, caching, reset logic. |
| `vla_cal/qwen_vl_client.py` | `QwenVLClient` (HTTP) + `QwenVLModel` (GPU). Complexity and precision prompts. |
| `vla_cal/qwen_vl_server.py` | HTTP server: `/health`, `/complexity`, `/precision` |
| `tests/test_ua_vla_il.py` | 30 unit tests — all mocked, no GPU needed |

## Modified Files

| File | Change |
|---|---|
| `uncertainty_quant_cliport/cliport/agents/transporter_lang_goal.py` | +27 lines: import VLACalibrator, instantiate in `_build_model()`, call `predict_both()` in `act()` before temperature scaling |
| `uncertainty_quant_peract/uncertainty_module/src/temperature_scaling/temperature_scaling.py` | +4 lines: `set_temperature(T)` method for adaptive override |

All original model files are untouched.

---

## Installation

### 1. Base environment
```bash
conda create -n ua_vla_il python=3.9 -y
conda activate ua_vla_il

# Install one of the three base models (choose what you need)
pip install -e uncertainty_quant_cliport/
pip install -e uncertainty_quant_peract/
pip install -e uncertainty_quant_rvt/
```

### 2. UA-VLA-IL dependencies
```bash
pip install transformers>=4.45.0 accelerate qwen-vl-utils requests Pillow
```

Qwen2-VL-2B weights (~4.5GB) download automatically on first launch.

### 3. GPU requirements
| Component | VRAM |
|---|---|
| Qwen2-VL-2B (float16) | ~5 GB |
| PerAct / RVT / CLIPort | ~3–6 GB |
| **Total** | **~8–11 GB** |

---

## Running UA-VLA-IL

### Step 1 — Start the Qwen2-VL-2B server
```bash
python -m vla_cal.qwen_vl_server --port 12190
```

Wait ~60 seconds for the model to load. Verify:
```bash
curl http://localhost:12190/health
# {"status": "ok"}
```

### Step 2 — Enable UA-VLA-IL in your config
Add this block to your existing CLIPort / PerAct YAML config:
```yaml
vla_cal:
  enabled: true
  qwen_port: 12190
  t_base: 1.0        # base temperature (original paper's T)
  alpha: 1.5         # sensitivity to visual complexity
  cache_steps: 5     # reuse VLM prediction for N steps
```

### Step 3 — Run evaluation (CLIPort example)
```bash
cd uncertainty_quant_cliport
python cliport/calib_eval.py
```

### Step 4 — Run tests (no GPU needed)
```bash
pytest tests/test_ua_vla_il.py -v
```

---

## Hyperparameters

All tunable via config YAML or environment variables:

| Parameter | Default | Formula | Description |
|---|---|---|---|
| `T_base` | 1.0 | `T = T_base + α×c` | Base temperature when scene is simple |
| `alpha` | 1.5 | `T = T_base + α×c` | How much complexity raises T |
| `omega_base` | 7.0 (CLIPort) / 5.0 (PerAct) | `ω = ω_base×(1−p+ε)` | Base neighborhood size |
| `eps` | 0.05 | `ω = ω_base×(1−p+ε)` | Prevents ω from reaching 0 |
| `cache_steps` | 5 | — | Steps between VLM queries (speed/accuracy tradeoff) |

Set via env vars: `VLA_CAL_T_BASE`, `VLA_CAL_ALPHA`, `VLA_CAL_OMEGA_BASE_CLIPORT`, `VLA_CAL_CACHE_STEPS`

---

## Ablation Table (Expected Results)

Evaluated on RLBench (PerAct) and Ravens (CLIPort):

| Method | CLIPort SR | PerAct SR |
|---|---|---|
| Baseline (no calibration) | 80.3% | 38.2% |
| Original paper (fixed T, fixed ω) | 83.3% | 41.4% |
| **UA-VLA-IL (adaptive T + ω)** | **TBD** | **TBD** |

*Results to be filled in after evaluation. Contribution is in the adaptive calibration mechanism.*

---

## Citation

```bibtex
@misc{rongali2025uavlail,
  title   = {UA-VLA-IL: VLM-Adaptive Calibration for Uncertainty-Aware Imitation Learning},
  author  = {Rongali, Sai Bhargav},
  year    = {2025},
  url     = {https://github.com/Saibhargav1208/UA_VLA_IL}
}

@inproceedings{wu2024uncertainty,
  title     = {Uncertainty-Aware Deployment of Pre-trained Language-Conditioned Imitation Learning Policies},
  author    = {Wu, Bob and Le Cleac'h, Simon and Matni, Nikolai and Bucher, Bernadette},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2024}
}

@inproceedings{shridhar2022peract,
  title     = {Perceiver-Actor: A Multi-Task Transformer for Robotic Manipulation},
  author    = {Shridhar, Mohit and Manuelli, Lucas and Fox, Dieter},
  booktitle = {Proceedings of the 6th Conference on Robot Learning (CoRL)},
  year      = {2022}
}

@article{goyal2023rvt,
  title   = {RVT: Robotic View Transformer for 3D Object Manipulation},
  author  = {Goyal, Ankit and Xu, Jie and Guo, Yijie and Blukis, Valts and Chao, Yu-Wei and Fox, Dieter},
  journal = {arXiv preprint arXiv:2306.14896},
  year    = {2023}
}

@inproceedings{shridhar2021cliport,
  title     = {CLIPort: What and Where Pathways for Robotic Manipulation},
  author    = {Shridhar, Mohit and Manuelli, Lucas and Fox, Dieter},
  booktitle = {Proceedings of the 5th Conference on Robot Learning (CoRL)},
  year      = {2021}
}
```

---

## Contact

**Rongali Sai Bhargav** — Honda R&D | IIT Bombay (B.Tech 2024)
[Google Scholar](https://scholar.google.com/citations?user=guiqu3wAAAAJ) · [GitHub](https://github.com/Saibhargav1208)

Interested in collaboration on robot manipulation, uncertainty quantification, or VLM-grounded control? Open an issue or reach out directly.

---

*UA-VLA-IL builds on the [original uncertainty quantification codebase](https://github.com/BobWu1998/uncertainty_quant_all) (MIT License) by Wu et al. All original model files are included as-is with no structural changes — only targeted additions.*
