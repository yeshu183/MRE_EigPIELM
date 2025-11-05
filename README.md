# MRE-PINN

> **Note**: This repository is being modified with the following objectives:
> 1. Resolve various implementation issues to ensure smooth execution
> 2. Integrate EIG-PIELM methodology into the existing framework
> 3. Comprehensive Testing and Comparison:
>    - Test the EIG-PIELM implementation on simulation datasets
>    - Validate on real-world datasets available in the repository
>    - Compare performance against:
>      - Original MRE-PINN implementation
>      - Existing FEM-based approaches (using available code in this repository)

This repository contains code for the paper *Physics-informed neural networks for tissue elasticity reconstruction in magnetic resonance elastography* which is to be presented at MICCAI 2023.

![MRE-PINN examples](MICCAI-2023/images/patient_image_grid.png)

## Installation

### Local Installation (CPU/GPU)

Run the following to setup the conda environment and register it as a Jupyter notebook kernel:

```bash
mamba env create --file=environment.yml
mamba activate MRE-PINN
python -m ipykernel install --user --name=MRE-PINN
```

### Lightning AI (Recommended for GPU Training)

For faster training with free GPU credits, see [LIGHTNING_AI_GUIDE.md](LIGHTNING_AI_GUIDE.md) for detailed instructions.

Quick start on Lightning AI:
```bash
git clone https://github.com/yeshu183/MRE_EigPIELM.git
cd MRE_EigPIELM
./setup_lightning.sh
python download_data.py
python train_lightning.py --example_id 90 --frequency 90 --n_iters 100000
```

## Usage

### Jupyter Notebook

This [notebook](MICCAI-2023/MICCAI-2023-simulation-training.ipynb) downloads the BIOQIC simulation data set and trains PINNs to reconstruct a map of shear elasticity from the displacement field.

The notebook takes roughly 2.5 h to train for 100,000 iterations on an RTX 5000 and uses 2.5 GiB of GPU memory.

### Python Script

For training via command line (recommended on Lightning AI):

```bash
# Download data first
python download_data.py

# Train model
python train_lightning.py --example_id 90 --frequency 90 --n_iters 100000
```

See [LIGHTNING_AI_GUIDE.md](LIGHTNING_AI_GUIDE.md) for all training options and parameters.
