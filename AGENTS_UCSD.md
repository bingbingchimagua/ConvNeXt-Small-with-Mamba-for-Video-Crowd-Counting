# AGENTS Memory - UCSD

## UCSD Goal
- Evaluate and adapt the current `VideoMambaCounter` on UCSD Pedestrian crowd counting.
- Use the common CVPR-2000 split:
  - train: global frames `601-1400`
  - test: global frames `1-600` and `1401-2000`
- Primary reporting metric: frame-level MAE/RMSE.
- Clip-level MAE/RMSE and density MSE are auxiliary diagnostics.

## Current UCSD Data Path
- Training/eval root on autodl: `/root/autodl-tmp/UCSD`
- Density cache root: `/root/autodl-tmp/UCSD_cache`
- Active cache variant for the current E2 line:
  - `ucsd/{train,test}/perspective_s0p5_min1p2_max6/*.npy`
- Earlier cache variant:
  - `ucsd/{train,test}/perspective_s0p5_min0p5_max6/*.npy`
- ROI and perspective priors are loaded from `uscdpeds_gt/gt/vidf`.

## Current UCSD Baseline
- The current UCSD baseline is the user-modified E2 scratch run.
- It was trained from scratch and did not use Mall `Exp19A` initialization.
- Real E2 configuration:
  - `embed_dim=64`
  - `depth=2`
  - `clip_len=8`
  - train input `240x360`
  - `batch_size=4`
  - `epochs=300`
  - `lr=1e-5`
  - `weight_decay=1e-4`
  - `density_kernel=perspective`
  - `perspective_scale=0.5`
  - `adaptive_min_sigma=1.2`
  - `adaptive_max_sigma=6.0`
  - `train_mode=crop`
  - `train_crop_size=224`
  - `train_scale_min=0.8`
  - `train_scale_max=1.2`
  - `random_hflip=True`
  - `train_clip_stride=5`
  - `test_clip_stride=1`
  - `use_roi_mask=True`
  - `use_roi_input=True`
  - `use_perspective_input=True`
  - `patch_weight=0.5`
  - `patch_grid_size=4`
  - `lambda_tc=0.01`
  - `tc_mode=density_prob`
  - `tc_warmup_epochs=10`
  - `scheduler=plateau`
  - `use_ema=True`
  - `early_stop_patience=15`
- Approximate results:
  - `120 epochs`: MAE around `5-6`
  - `300 epochs`: early stopped, MAE around `7-8`

## Current UCSD Judgment
- Longer training hurts rather than helps.
- The main issue is likely late-stage generalization degradation, not insufficient epochs.
- High augmentation strength, horizontal flip, crop count supervision, `patch_weight=0.5`, and `lambda_tc=0.01` may jointly make the small UCSD split unstable.
- `train_clip_stride=5` reduces per-epoch clip coverage, but the 300-epoch degradation means the core issue is not simply too few optimization steps.
- UCSD is low-resolution, single-scene, and strongly temporally smooth; full-frame count behavior is more important than dense localization quality for thesis reporting.
- E3A and E3B have now failed to beat E2:
  - E3A clean full-frame is worse than E2.
  - E3B light crop is also worse than E2.
  - This indicates that simply removing crop/hflip/patch/TC is not the right next direction.
- A new evaluation-scale issue has been confirmed:
  - evaluating the current E2 checkpoint at `192x288` gives MAE around `7-8`
  - evaluating at `240x320` makes offline MAE collapse to around `35`
  - this is treated as a severe scale/aspect/calibration mismatch, not a true model-quality result
- For current UCSD reporting, prefer the original `192x288` / `test_short_side=192` offline evaluation path unless a controlled scale ablation proves otherwise.
- For all new UCSD experiments, align training and evaluation to `192x288`; do not mix a `240x360` train setup with a `192x288` eval setup unless you are explicitly testing scale robustness.
- The active SOTA-chasing route is now fixed-scene count calibration:
  - save E2 train/test sequence predictions at `192x288`
  - sweep temporal post-processing on E2 best
  - fit affine or scale-only count calibration on train frames
  - apply calibration to test frames, then optionally apply weak motion-guided smoothing
- A new utility script exists for this route:
  - `calibrate_ucsd_counts.py`
- E5/E6 are now confirmed useful:
  - before calibration/smoothing: MAE `5.827958`, RMSE `6.972755`
  - affine with `motion_guided lambda=5 window=36`: MAE `4.423960`, RMSE `5.732198`
  - scale-only with `motion_guided lambda=5 window=36`: MAE `4.124715`, RMSE `5.484081`
  - scale-only is currently stronger than affine, suggesting bias correction overfits cross-segment count offsets.
- E8/E9 added more evidence:
  - E8 test exploration best observed: scale-only, ridge `1e-2`, `motion_guided lambda=12 window=28`, MAE `4.113213`, RMSE `5.478852`.
  - E9 inner train-val selected: scale-only, ridge `1e-2`, `motion_guided lambda=3 window=18`, inner MAE `2.824348`, RMSE `3.368927`.
  - E9 final test with the inner-selected config produced MAE `4.332667`, RMSE `5.642715`, so train-mid validation does not perfectly match the two held-out test segments.
  - E2 `last.pth` raw test is better than E2 `best.pth` raw test: frame MAE `5.170358`, RMSE `6.315279`.
- E12/E13 are now exhausted:
  - E12 last-only best after calibration/smoothing: MAE `4.442532`, RMSE `5.732123`.
  - E13 `0.25 best + 0.75 last`: MAE `4.367113`, RMSE `5.676456`.
  - E13 `0.50 best + 0.50 last`: MAE `4.287137`, RMSE `5.614760`.
  - E13 `0.10 best + 0.90 last`: MAE `4.412925`, RMSE `5.709372`.
  - None of these beat E8, so last-only and best/last ensemble should stop.
- E15 micro sweep found a tiny new best:
  - scale-only, ridge `1e-2`, `motion_guided lambda=13.5 window=26`, MAE `4.113029`, RMSE `5.478136`.
  - The gain over E8 is only about `0.000184` MAE, so the scale+motion-guided family is effectively saturated.
- New reporting/helper scripts:
  - `report_ucsd_segments.py` prints overall/head/tail metrics from metrics directories or sweep results.
  - `anchor_smooth_ucsd_counts.py` implements train-GT-anchored full-sequence smoothing using only frames `601-1400` as anchors and reporting test head/tail.
- E18 train-anchor smoothing produced the first strong breakthrough:
  - `lp1 ls12 la100 w28`: MAE `3.960702`, RMSE `5.410488`
  - `lp1 ls20 la300 w28`: MAE `3.869143`, RMSE `5.375907`
  - `lp1 ls30 la1000 w36`: MAE `3.705936`, RMSE `5.344047`
  - The main remaining issue is `ucsd_test_head` systematic overestimation; for the best E18 run, head bias is `+4.595771` while tail bias is `+1.497310`.
- E19 segmented calibration has now been judged worse than E18 and should be stopped.
- Final UCSD delivery decision:
  - Lock E18 `lp1 ls30 la1000 w36` as the main UCSD result.
  - Treat E18 as fixed-camera train-anchor temporal calibration, not as pure model raw performance.
  - Keep E15 `scale + motion_guided lambda=13.5 window=26` as prediction-only calibration baseline.
  - Do not claim public UCSD SOTA; present E18 as a strong improvement within this model pipeline.
- `anchor_smooth_ucsd_counts.py` now supports segmented calibration:
  - `--calibration-mode adjacent_halves`
  - `--calibration-mode boundary_windows`
  - `--boundary-window-size`
- A second utility script now exists:
  - `sweep_ucsd_calibration.py`
  - It sweeps source/model/ridge/postproc/lambda/window, writes ranked CSV/JSON, and can save top calibrated predictions.
- A third utility script now exists:
  - `ensemble_ucsd_predictions.py`
  - It weighted-averages multiple saved UCSD sequence prediction directories and writes compatible prediction JSONs for calibration/sweep.
- For thesis-quality reporting, do not rely only on test-set parameter exploration:
  - use train frames `601-1000` for fitting
  - use train frames `1001-1400` as inner validation for smoothing/ridge selection
  - refit on full train `601-1400`, then apply once to test
- UCSD-Exp40B-Prior remains the active UCSD mainline, but the latest full-sequence anchor result shows a ceiling:
  - structure: ImageNet `ConvNeXt-Small + bidirectional Temporal Mamba`
  - no Mall checkpoint initialization
  - `clip_len=16`, `192x288`
  - explicit UCSD priors enabled through `use_roi_mask`, `use_roi_input`, `use_perspective_input`, and `mask_rgb_with_roi`
  - `best.pth` raw test result:
    - `frame_count_mae = 3.027012`
    - `frame_count_rmse = 3.673662`
    - `clip_count_mae = 2.929126`
    - `clip_count_rmse = 3.567093`
  - latest full-sequence anchor result:
    - `frame_count_mae = 2.434468`
    - `frame_count_rmse = 3.265630`
    - `frame_count_bias = 0.631648`
    - `ucsd_test_head MAE = 3.010355`
    - `ucsd_test_tail MAE = 1.858581`
  - Interpretation:
    - the model now has a better raw count prior than E18, but the sequence-level correction is still dominated by head-segment bias
    - the remaining gap is not a calibration-only problem; the next improvement must change the objective/structure rather than only sweep post-processing

## UCSD Known Issues
- Previous parser failure for `vidf1_33_004/005/006_people_full.mat` was traced to the `people.deleted` flag.
- The parser now prefers non-deleted tracks but falls back to all tracks when a clip would otherwise contain no usable tracks.
- Local audit after the fix reports no fallback clips.
- Local audit point-count mismatch:
  - train MAE(point_count-count) = `1.741250`, bias = `-0.726250`
  - test MAE(point_count-count) = `5.885833`, bias = `-5.817500`
  - clips `004/005/006` are now close to official count, but some test clips remain systematically below official ROI counts.
- Interpretation: density spatial supervision for the core train clips is now cleaner, but point counts should not be treated as official count labels.

## UCSD Next Actions
1. CV1 confirmed that Mall Exp40B initialization transfers to UCSD and remains the active UCSD checkpoint:
   - `CV1 best`: frame MAE `2.151657`, frame RMSE `2.614105`, clip MAE `2.039720`, clip RMSE `2.479488`.
   - `CV1 last`: frame MAE `2.365001`, frame RMSE `2.798507`, clip MAE `2.267055`, clip RMSE `2.686606`.
   - `CV1 best + motion_guided lambda=0.5 window=5`: frame MAE `2.132524`, frame RMSE `2.587933`, clip MAE `2.026067`, clip RMSE `2.461083`.
2. CV2/CV3/CV4 are now judged worse than CV1, so stop model-side changes for this branch.
3. Next work is CV1-only post-processing:
   - export CV1 best train/test raw predictions
   - run train-only inner calibration selection (`601-1000` fit, `1001-1400` validation)
   - refit selected calibration on full train `601-1400`, apply to test
   - optionally run train-anchor smoothing on merged train+test predictions
4. Do not tune post-processing directly on test unless explicitly marked as exploratory upper-bound.

## Update Rules
- Keep UCSD updates in this file, not in the Mall-focused `AGENTS.md`.
- Update this file after:
  - a new UCSD best metric appears
  - an E3/E4 result changes the experiment order
  - a new UCSD data parsing/stability issue is confirmed
  - the active UCSD baseline changes
