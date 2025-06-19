# MED: A Multi‐Encoder‐Decoder Architecture for Mixed‐frequency Data Forecasting

## Paper

This repository contains the official implementation for

> **“MED: A Multi-Encoder-Decoder Architecture for Mixed-Frequency Data Forecasting.”**  
> Xuebin Chen, Jiaxi Liu*, Silin Li, Xu Pan and Kai Ren. *Expert Systems with Applications*, 2025.  
> DOI: <https://doi.org/10.1016/j.eswa.2025.128546>  
> *Corresponding author: liujiaxi@stu.scu.edu.cn

## Citation
If you find this code useful, please cite the paper:
```bibtex
@article{Liu2025MED,
  title   = {A Multi-Encoder-Decoder Architecture for Mixed-Frequency Data Forecasting},
  author  = {Chen, Xuebin and Liu, Jiaxi and Li, Silin and Pan, Xu and Ren, Kai},
  journal = {Expert Systems with Applications},
  year    = {2025},
  doi     = {10.1016/j.eswa.2025.128546},
  url     = {https://doi.org/10.1016/j.eswa.2025.128546}
}
```
## Environment

| Item | Tested version |
|------|-------------|
| macOS | Sequoia 15.4.1 |
| Python | 3.9         |
| PyTorch | 2.2.0       |

### Quick start (Conda)
```bash
conda create -n med python=3.9
conda activate med
conda install pyyaml numpy pandas scikit-learn
conda install pytorch=2.2.0 
conda install openpyxl
```


# Repository Layout

```text
multi-encoder-decoder/
├── configs/          # YAML experiment configs
│   ├── 0_simu/       # synthetic data experiments
│   │   ├── defaults.yaml
│   │   └── med.yaml
│   ├── 1_fred/ …     # unemployment-rate experiments
│   │   ├── defaults.yaml
│   │   └── med.yaml
│   ├── 2_elec/ …     # electricity-load experiments
│   │   ├── defaults.yaml
│   │   └── med.yaml
│   └── 3_wea/  …     # CO₂ concentration experiments
│   │   ├── defaults.yaml
│   │   └── med.yaml
├── data/             # raw & synthetic datasets
│   ├── 0_simu/
│   │   ├── generate_simu_data.py
│   │   └── simu_data.xlsx
│   ├── 1_fred/
│   │   └── 202410.xlsx
│   ├── 2_elec/
│   │   └── elec.xlsx
│   ├── 1_wea/
│   │   └── wea.xlsx
├── models/           # core source code
│   ├── dual_encoder_decoder.py
│   ├── med_model.py
│   └── rnn_components.py
├── args_parser.py    # YAML + CLI merger
├── data_processor.py # Excel loader & splitter
├── trainer.py        # training / evaluation orchestrator
├── utils.py          # dataset builders, metrics, plotting
└── main.py           # entry point
```
# Experiments
## 0. Synthetic demo
```console
Synthetic data ready-made:
# python data/0_simu/generate_simu_data.py      # create ./data/0_simu/simu_data.xlsx
python main.py --configs configs/0_simu/med.yaml
```
## 1. Unemployment rate forecasting
```console
# dual-encoder-decoder, MFA fusion
python main.py --configs configs/1_fred/med.yaml --add_info run1
```
## 2. Electricity consumption forecasting
```console
python main.py --configs configs/2_elec/med.yaml
```

## 3. CO2 concentration forecasting
```console
python main.py --configs configs/3_wea/med.yaml
```
To explore other hyper-parameters, copy any YAML file and modify the values, e.g.:

```console
cp configs/1_fred/med.yaml configs/1_fred/med_h32.yaml
# edit hidden sizes, dropout, etc.
python main.py --configs configs/1_fred/med_h32.yaml
```

## Acknowledgement
The authors acknowledge financial support from the National Natural Science Foundation of China (No. 7237011356).

