# A2W Plücker FOV55 ACT 50K Evaluation

Date: 2026-07-04

## Checkpoint

- Training run: `leroact_a2w_plucker_fov55_chunk100_exec50_bs16_0703_2332`
- Official checkpoint: `high-level/dp/logs/lerobot-train/leroact_a2w_plucker_fov55_chunk100_exec50_bs16_0703_2332/checkpoints/050000`
- Wrapped checkpoint: `high-level/dp/logs/door-auto-wrapped/leroact_a2w_plucker_fov55_chunk100_exec50_bs16_0703_2332/050000/model_latest.pt`
- Plücker FOV: `55.0 deg`

## Evaluation setup

- `num_envs=16`
- `total_trials=64`
- `steps=1000`
- `base_seed=62000`
- `success_metric=abs`
- `pass_open_angle_deg=80`
- depth noise disabled
- gaussian blur disabled
- robot pitch fixed at `0.0`

## Results

### Standard eval with camera randomization enabled

This matches the earlier camera-randomized evaluation setting:

| checkpoint | horizon | success |
|---|---:|---:|
| Plücker FOV55 50K | 10 | 9/64 = 14.06% |
| Plücker FOV55 50K | 12 | 11/64 = 17.19% |
| Plücker FOV55 50K | 14 | 9/64 = 14.06% |

Best under camera randomization: **11/64 = 17.19% at horizon 12**.

### Control eval with camera randomization disabled

The recorded Plücker dataset sidecar shows camera randomization was not enabled during data collection, so this setting better matches the training distribution:

| checkpoint | horizon | success |
|---|---:|---:|
| Plücker FOV55 50K | 12 | 48/64 = 75.00% |
| Plücker FOV55 50K | 14 | 38/64 = 59.38% |

Best with matched no-camera-randomization distribution: **48/64 = 75.00% at horizon 12**.

## Comparison to previous runs

| model / training variant | best known result |
|---|---:|
| No-keyframe baseline 50K | 42/64 = 65.62% |
| W3/R3 + sample30 + gating 50K | 39/64 = 60.94% |
| Phase200 W3 loss-only 50K | 35/64 = 54.69% |
| Per-timestep W3/R3 sample30 no-gating 50K | 33/64 = 51.56% |
| Plücker FOV55 50K, camera rand eval | 11/64 = 17.19% |
| Plücker FOV55 50K, no camera rand eval | 48/64 = 75.00% |

## Analysis

The Plücker model is not simply worse. It performs poorly only when camera randomization is enabled at evaluation time.

The current Plücker training data appears to have been collected without camera randomization, while the standard evaluation command enables camera pose randomization. Under that mismatch, both the visual depth pattern and the Plücker ray-map distribution shift. The model receives camera geometry, but it has not seen enough randomized camera poses during training to learn how to use that geometry robustly.

When evaluation disables camera randomization to match the training distribution, Plücker FOV55 50K reaches **48/64**, which is better than the previous best baseline result of **42/64**.

## Recommendation

For a fair Plücker-vs-baseline comparison under camera-randomized deployment, re-record or regenerate a Plücker dataset with:

```bash
--record_camera_pose
--enable_depth_camera_randomization
--depth_camera_pos_rand_m 0.02
--depth_camera_rot_rand_deg 5.0
```

Then train the same Plücker architecture again and evaluate with the same camera-randomized setting.
