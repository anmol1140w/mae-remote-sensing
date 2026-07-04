# Dataset Setup

This folder holds the **EuroSAT RGB** dataset used for MAE pre-training and fine-tuning.
The dataset is **not included** in this repository — download it separately.

## Download

1. Get EuroSAT from the official repository: [https://github.com/phelber/EuroSAT](https://github.com/phelber/EuroSAT)
2. Extract the RGB subset so the class folders sit directly under `datasets/EuroSAT/`

## Expected Layout

```
datasets/
└── EuroSAT/
    ├── AnnualCrop/
    ├── Forest/
    ├── HerbaceousVegetation/
    ├── Highway/
    ├── Industrial/
    ├── Pasture/
    ├── PermanentCrop/
    ├── Residential/
    ├── River/
    └── SeaLake/
```

Each class folder should contain `.jpg` images (64×64 RGB, 3 channels).

## About EuroSAT

- **27,000** satellite images across **10** land cover classes
- Used for both self-supervised pre-training (no labels needed) and supervised classification
- Train / val / test splits (80 / 10 / 10) are created automatically via stratified sampling — no manual split required

## Config

The dataset path is set in `configs/config.yaml`:

```yaml
dataset:
  path: "datasets/EuroSAT"
```

Change this only if you store EuroSAT elsewhere on disk.
