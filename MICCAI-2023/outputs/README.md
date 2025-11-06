# Training Outputs Folder

This folder contains all output files generated during MRE-PINN training runs.

## Contents

### Main Output Files
- `DEMO_*.nc` - NetCDF files containing training results (anatomy, elastogram, wave fields, etc.)
- `DEMO_train_*.png` - Training metric plots (norms, correlations, frequencies, regions)
- `DEMO_train_metrics.csv` - Detailed training metrics in CSV format

### Subdirectories

#### viewers/
Contains PNG snapshots of training visualizations saved at regular intervals:
- `DEMO_anatomy_*.png` - Anatomy visualizations
- `DEMO_direct_*.png` - Direct inversion results
- `DEMO_elastogram_*.png` - Elastogram visualizations
- `DEMO_laplacian_*.png` - Laplacian visualizations
- `DEMO_pde_*.png` - PDE residual visualizations
- `DEMO_wave_field_*.png` - Wave field visualizations

#### weights/
Contains model checkpoints saved during training:
- `DEMO_model-*.pt` - PyTorch model state dictionaries saved at specified intervals

## Notes

- This folder is automatically created by the training script
- All files in this folder are **ignored by git** (see `.gitignore`)
- Files are organized to prevent clutter in the main repository
- You can safely delete this folder to clean up training outputs
- New training runs will recreate this folder structure automatically

## Configuration

The output location is configured in the training script via the `save_prefix` parameter:

```python
test_eval = mre_pinn.testing.TestEvaluator(
    test_every=1000,
    save_every=10000,
    save_prefix='DEMO',  # Files will be saved to outputs/DEMO_*
    interact=True
)
```

The code automatically places all outputs in the `outputs/` directory to keep the repository clean.
