# MRE-PINN Documentation Index

Complete guide to navigating the MRE-PINN codebase documentation.

---

## 📚 Quick Navigation

### Getting Started

1. **[README.md](README.md)** - Start here! Installation and quick start
2. **[ARCHITECTURE.md](ARCHITECTURE.md)** - System overview and data flow
3. **[MICCAI-2023/EXPERIMENTS_GUIDE.md](MICCAI-2023/EXPERIMENTS_GUIDE.md)** - Experiment notebooks

### Core Documentation

| Topic | Location | Description |
|-------|----------|-------------|
| **Package Overview** | [mre_pinn/PACKAGE_OVERVIEW.md](mre_pinn/PACKAGE_OVERVIEW.md) | Main Python package |
| **Data Pipeline** | [mre_pinn/data/DATA_MODULE.md](mre_pinn/data/DATA_MODULE.md) | Data loading & preprocessing |
| **Neural Networks** | [mre_pinn/model/MODEL_ARCHITECTURES.md](mre_pinn/model/MODEL_ARCHITECTURES.md) | PINN architecture |
| **Training** | [mre_pinn/training/TRAINING_MODULE.md](mre_pinn/training/TRAINING_MODULE.md) | Training procedures |
| **Baselines** | [mre_pinn/baseline/BASELINE_METHODS.md](mre_pinn/baseline/BASELINE_METHODS.md) | AHI & FEM methods |
| **Evaluation** | [mre_pinn/testing/TESTING_MODULE.md](mre_pinn/testing/TESTING_MODULE.md) | Metrics & testing |
| **Datasets** | [data/DATASETS_GUIDE.md](data/DATASETS_GUIDE.md) | Data storage |

---

## 🎯 Documentation by Task

### I want to understand the overall system

1. Read **[ARCHITECTURE.md](ARCHITECTURE.md)** for high-level overview
2. Read **[mre_pinn/PACKAGE_OVERVIEW.md](mre_pinn/PACKAGE_OVERVIEW.md)** for package structure
3. Review workflow diagrams in ARCHITECTURE.md

### I want to load and preprocess data

1. **[mre_pinn/data/DATA_MODULE.md](mre_pinn/data/DATA_MODULE.md)** - Complete data pipeline
2. **[MICCAI-2023/EXPERIMENTS_GUIDE.md](MICCAI-2023/EXPERIMENTS_GUIDE.md)** - See preprocessing notebook
3. **[data/DATASETS_GUIDE.md](data/DATASETS_GUIDE.md)** - Dataset information

### I want to train a PINN model

1. **[mre_pinn/training/TRAINING_MODULE.md](mre_pinn/training/TRAINING_MODULE.md)** - Training details
2. **[mre_pinn/model/MODEL_ARCHITECTURES.md](mre_pinn/model/MODEL_ARCHITECTURES.md)** - Model architecture
3. **[MICCAI-2023/EXPERIMENTS_GUIDE.md](MICCAI-2023/EXPERIMENTS_GUIDE.md)** - Training notebooks

### I want to understand the physics

1. **[ARCHITECTURE.md](ARCHITECTURE.md)** - Section "Key Concepts"
2. **[mre_pinn/PACKAGE_OVERVIEW.md](mre_pinn/PACKAGE_OVERVIEW.md)** - Section "pde.py"
3. Review wave equation formulation

### I want to compare with baselines

1. **[mre_pinn/baseline/BASELINE_METHODS.md](mre_pinn/baseline/BASELINE_METHODS.md)** - Baseline methods
2. **[MICCAI-2023/EXPERIMENTS_GUIDE.md](MICCAI-2023/EXPERIMENTS_GUIDE.md)** - See FEM notebooks

### I want to reproduce the paper results

1. **[MICCAI-2023/EXPERIMENTS_GUIDE.md](MICCAI-2023/EXPERIMENTS_GUIDE.md)** - All experiment notebooks
2. **[data/DATASETS_GUIDE.md](data/DATASETS_GUIDE.md)** - Download datasets
3. Follow notebook execution order

---

## 📖 Documentation Structure

```
MRE-PINN/
│
├── README.md                     # ⭐ Start here
├── ARCHITECTURE.md               # ⭐ System overview
├── DOCUMENTATION_INDEX.md        # This file
│
├── mre_pinn/                     # Main package
│   ├── README.md                 # Package overview
│   ├── data/
│   │   └── README.md             # Data management
│   ├── model/
│   │   └── README.md             # Neural networks
│   ├── training/
│   │   └── README.md             # Training procedures
│   ├── baseline/
│   │   └── README.md             # Comparison methods
│   └── testing/
│       └── README.md             # Evaluation
│
├── MICCAI-2023/
│   └── README.md                 # Experiment notebooks
│
└── data/
    └── README.md                 # Dataset information
```

---

## 🔍 Find Information By Topic

### Concepts

| Topic | Where to Find |
|-------|--------------|
| What is MRE? | [ARCHITECTURE.md](ARCHITECTURE.md#key-concepts) |
| What is PINN? | [ARCHITECTURE.md](ARCHITECTURE.md#physics-informed-neural-networks) |
| Wave equation | [mre_pinn/PACKAGE_OVERVIEW.md](mre_pinn/PACKAGE_OVERVIEW.md#pde.py) |
| Ground truth | [ARCHITECTURE.md](ARCHITECTURE.md#ground-truth-vs-predictions) |
| Segmentation | [mre_pinn/data/DATA_MODULE.md](mre_pinn/data/DATA_MODULE.md#segment.py) |
| Registration | [mre_pinn/data/DATA_MODULE.md](mre_pinn/data/DATA_MODULE.md#register_image) |

### Implementation

| Topic | Where to Find |
|-------|--------------|
| Load data | [mre_pinn/data/DATA_MODULE.md](mre_pinn/data/DATA_MODULE.md#usage) |
| Create model | [mre_pinn/model/MODEL_ARCHITECTURES.md](mre_pinn/model/MODEL_ARCHITECTURES.md#usage-examples) |
| Train model | [mre_pinn/training/TRAINING_MODULE.md](mre_pinn/training/TRAINING_MODULE.md#training-loop) |
| Loss functions | [mre_pinn/training/TRAINING_MODULE.md](mre_pinn/training/TRAINING_MODULE.md#loss-functions) |
| Evaluate | [mre_pinn/testing/TESTING_MODULE.md](mre_pinn/testing/TESTING_MODULE.md#metrics) |
| Visualize | [mre_pinn/PACKAGE_OVERVIEW.md](mre_pinn/PACKAGE_OVERVIEW.md#visual.py) |

### Workflows

| Workflow | Where to Find |
|----------|--------------|
| Data preprocessing | [mre_pinn/data/DATA_MODULE.md](mre_pinn/data/DATA_MODULE.md#preprocessing-pipeline) |
| Training pipeline | [mre_pinn/training/TRAINING_MODULE.md](mre_pinn/training/TRAINING_MODULE.md#training-loop) |
| Full experiment | [MICCAI-2023/EXPERIMENTS_GUIDE.md](MICCAI-2023/EXPERIMENTS_GUIDE.md#running-experiments) |

---

## 🚀 Quick Links

### Most Important Files

1. **[ARCHITECTURE.md](ARCHITECTURE.md)** - Understand the whole system
2. **[mre_pinn/data/DATA_MODULE.md](mre_pinn/data/DATA_MODULE.md)** - Data pipeline (longest, most detailed)
3. **[mre_pinn/training/TRAINING_MODULE.md](mre_pinn/training/TRAINING_MODULE.md)** - How training works
4. **[MICCAI-2023/EXPERIMENTS_GUIDE.md](MICCAI-2023/EXPERIMENTS_GUIDE.md)** - Reproduce experiments

### Key Code Files

| File | Purpose | Documentation |
|------|---------|---------------|
| `imaging.py` | Patient data preprocessing | [data/DATASETS_GUIDE.md](mre_pinn/data/DATA_MODULE.md#imaging.py) |
| `pinn.py` | PINN architecture | [model/README.md](mre_pinn/model/MODEL_ARCHITECTURES.md#mrepinn) |
| `pinn_training.py` | Training loop | [training/README.md](mre_pinn/training/TRAINING_MODULE.md#mrepinnmodel) |
| `segment.py` | U-Net segmentation | [data/DATASETS_GUIDE.md](mre_pinn/data/DATA_MODULE.md#segment.py) |

---

## 📊 Documentation Statistics

| Document | Lines | Topics Covered |
|----------|-------|----------------|
| ARCHITECTURE.md | 600+ | System overview, data flow, workflows |
| mre_pinn/PACKAGE_OVERVIEW.md | 400+ | Package structure, utilities |
| mre_pinn/data/DATA_MODULE.md | 800+ | Data loading, preprocessing, segmentation |
| mre_pinn/model/MODEL_ARCHITECTURES.md | 300+ | Neural network architecture |
| mre_pinn/training/TRAINING_MODULE.md | 400+ | Training procedures, losses |
| mre_pinn/baseline/BASELINE_METHODS.md | 200+ | Baseline methods |
| mre_pinn/testing/TESTING_MODULE.md | 150+ | Evaluation metrics |
| MICCAI-2023/EXPERIMENTS_GUIDE.md | 300+ | Experiment notebooks |
| data/DATASETS_GUIDE.md | 200+ | Dataset information |

**Total**: ~3,350 lines of documentation!

---

## 🎓 Learning Path

### Beginner

1. Read [README.md](README.md) - Installation
2. Run `download_data.py`
3. Open [MICCAI-2023-simulation-training.ipynb](MICCAI-2023/MICCAI-2023-simulation-training.ipynb)
4. Read [ARCHITECTURE.md](ARCHITECTURE.md) while notebook runs
5. Explore visualization outputs

### Intermediate

1. Read [mre_pinn/data/DATA_MODULE.md](mre_pinn/data/DATA_MODULE.md)
2. Read [mre_pinn/model/MODEL_ARCHITECTURES.md](mre_pinn/model/MODEL_ARCHITECTURES.md)
3. Read [mre_pinn/training/TRAINING_MODULE.md](mre_pinn/training/TRAINING_MODULE.md)
4. Modify hyperparameters and retrain
5. Compare with baselines

### Advanced

1. Read all module documentation
2. Implement custom physics equations
3. Add new loss terms
4. Process new patient data
5. Contribute improvements

---

## 💡 Tips

### Finding Specific Information

1. **Use search** (Ctrl+F) within README files
2. **Follow cross-references** - all docs link to related docs
3. **Check diagrams** in ARCHITECTURE.md for visual understanding
4. **Read code comments** - well-documented inline

### Understanding Data Flow

1. Start with diagram in [ARCHITECTURE.md](ARCHITECTURE.md#data-flow-overview)
2. Follow preprocessing in [mre_pinn/data/DATA_MODULE.md](mre_pinn/data/DATA_MODULE.md#preprocessing-pipeline)
3. See training in [mre_pinn/training/TRAINING_MODULE.md](mre_pinn/training/TRAINING_MODULE.md#training-loop)

### Troubleshooting

Each README has a "Troubleshooting" section:
- [mre_pinn/data/DATA_MODULE.md](mre_pinn/data/DATA_MODULE.md#troubleshooting)
- [mre_pinn/training/TRAINING_MODULE.md](mre_pinn/training/TRAINING_MODULE.md#troubleshooting)

---

## 🔄 Documentation Updates

Documentation last updated: December 2024

To update documentation:
1. Edit relevant README.md file
2. Update cross-references if needed
3. Update this index if new files added

---

## 📝 Documentation Conventions

### File Naming
- `README.md` - In every directory
- `ARCHITECTURE.md` - Root-level overview
- `DOCUMENTATION_INDEX.md` - This file

### Markdown Structure
- Use headings (#, ##, ###) for hierarchy
- Use tables for comparisons
- Use code blocks with syntax highlighting
- Use relative links for cross-references

### Code Examples
- Include complete, runnable examples
- Show expected output
- Explain parameters

---

## ❓ Still Have Questions?

1. **Check the docs** using this index
2. **Search codebase** for examples
3. **Review notebooks** in MICCAI-2023/
4. **Open an issue** on GitHub
5. **Contact authors** for paper-related questions

---

## 🎉 You're Ready!

You now have comprehensive documentation for the entire MRE-PINN codebase. Happy coding!

**Quick Start Reminder**:
```bash
# 1. Install
mamba env create --file=environment.yml
mamba activate MRE-PINN

# 2. Download data
python download_data.py

# 3. Train (via notebook - recommended)
jupyter notebook MICCAI-2023/MICCAI-2023-simulation-training.ipynb

# Or via script
python train.py
```

Enjoy exploring Physics-Informed Neural Networks for MRE! 🚀
