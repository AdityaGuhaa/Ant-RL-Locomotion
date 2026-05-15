# CHANGELOG — Aditya Guha RL Quadruped Project

This changelog tracks the evolution of the Ant-based reinforcement learning environment and training logic used in Aditya Guha's DRDO internship project.

---

## Version V1 — Initial Setup

**Date:** Early June 2025

*  Installed Python dependencies: gymnasium, MuJoCo, numpy.
*  Created project folder structure.
*  Rendered `Ant-v4` simulation with basic manual control.

---

## Version V2 — Custom Environment Wrapper

**Date:** Mid June 2025

*  Created `AntStandEnv` wrapper around `Ant-v4`.
*  Added basic custom reward: torso height bonus (`> 0.5` → +5).
*  Reward logging to CSV: `ant_performance_log.csv`.

---

## Version V3 — PPO Integration with Stable-Baselines3

**Date:** Late June 2025

*  Installed `stable-baselines3`, `torch`.
*  Integrated `PPO` model for learning.
*  Used `Monitor` + `DummyVecEnv` for compatibility.
*  Model training initiated for `10000` steps.
*  Agent saved as `ppo_ant_stand.zip`.

---

## Version V4 — Multi-Agent Training & GUI Evaluation

**Date:** 1st July 2025

*  Switched to 8 parallel `AntStandEnv` instances using `DummyVecEnv`.
*  Trained agent using multiple parallel environments.
*  GUI restricted to evaluation only (1 Ant shown visually).
*  TensorBoard logging added at `./ppo_logs_tb/`.
*  Archived version as `AdityaGuhaV4`.

---

## Version V5 — Logging, Plotting & Video

**Date:** 3rd July 2025

*  Introduced `plot_logs.py` to visualize rewards and height.
*  Video recording added via `RecordVideo` wrapper.
*  Used `render_mode="rgb_array"` to enable off-screen video capture.
*  Videos saved to `/home/qrs/Aditya Guha/videos/`.
*  Video recording confirmed to be working.
*  Saved full snapshot as `AdityaGuhaV5`.

---

## Version V6 — Reward Shaping & Training Refinement

**Date:** 4th July 2025

*  Enhanced reward logic:
* `upright_bonus` ∝ height over threshold.
* `fall_penalty` if height < 0.3.
* `velocity_penalty` to discourage jitter.
* `angle_penalty` to promote uprightness.
* Switched all paths to `os.path.join`.
* Improved logging precision.
* TensorBoard visualizations confirmed.
* Evaluation loop confirmed functional with single GUI.
* Logging and rendering separated cleanly.
* Base for curriculum RL (CRL) considerations laid.

---

## 📍 NEXT STEPS (for V7)

* Reward function experimentation (try alternate penalties, balance bonuses).
* Try Curriculum Reinforcement Learning (CRL): easier → harder stages.
* Add training curves comparison.
* Set training targets (e.g., "hold stand >5 sec", "angle < 15°").
* Final visualization & result analysis.
