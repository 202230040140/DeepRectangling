# Reproduce DeepRectangling on StitchBench General

Pipeline: **OpenCV Stitcher → DeepRectangling → MDR/NIQE**. Generated results stay under `outputs/`, ignored by git.

## Metrics (OBJ-GSP aligned)

| Metric | Definition |
|--------|------------|
| **MDR** | Overlap mapping RMSE (pixels) on final `rectangled.png`: SIFT matches between input pair, each warped into panorama coordinates; RMSE of correspondence distances. Same role as OBJ-GSP mesh RMSE for fair cross-method comparison. |
| **NIQE** | `pyiqa` NIQE on `rectangled.png`, same as `OBJ-GSP/tools/evaluate_stitchbench_ours.py`. |

PSNR / SSIM / FID / BIQUE are **not** computed (no rectangling GT / not aligned with other reproduced papers).

Dataset manifest: OBJ-GSP 100-pair StitchBench General list (`manifest.csv`).

## Environment

```powershell
pip install -r requirements-reproduce.txt
```

TensorFlow 2.x + `tf-slim` runs the legacy checkpoint via `deep_rectangling_py/tf_compat.py` (no separate TF 1.13 conda env required).

## Full Run

```powershell
cd c:\Users\22499\Documents\GitHub\DeepRectangling
$env:OPENCV_OPENCL_RUNTIME = "disabled"
py -3.13 -m deep_rectangling_py.run_stitchbench_general `
  --manifest C:\Users\22499\Documents\GitHub\Depth-GSP\experiments\phase1_depth_loss\runs\depth_gsp_v5_planarity035\manifest.csv `
  --out outputs\stitchbench_general `
  --device cuda `
  --gpu 0
```

## Compare with OBJ-GSP / Depth-GSP baseline

```powershell
py -3.13 -m deep_rectangling_py.evaluate_stitchbench_mdr_niqe `
  --candidate-root outputs\stitchbench_general `
  --baseline-root C:\Users\22499\Documents\GitHub\Depth-GSP\experiments\phase1_depth_loss\runs\depth_gsp_v5_planarity035 `
  --output-root outputs\deeprect_mdr_niqe
```

## Output Layout

```
outputs/stitchbench_general/
  per_pair.csv
  by_category.csv
  summary.json
  report.md
  SPHP-01_bridge/
    stitched.png
    mask.png
    rectangled.png
    metrics.json
```

## Smoke Test (single scene)

```powershell
py -3.13 -m deep_rectangling_py.run_stitchbench_general `
  --manifest C:\Users\22499\Documents\GitHub\Depth-GSP\experiments\phase1_depth_loss\runs\depth_gsp_v5_planarity035\manifest.csv `
  --out outputs\smoke `
  --scene SPHP-01_bridge `
  --limit 1 `
  --device cpu `
  --gpu -1
```

Delete `outputs/smoke/` after validating the full run.
