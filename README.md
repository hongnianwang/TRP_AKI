# TRP_AKI

This repository supports the development and validation of the paper: "Interpreting Temporal Risk Patterns of Acute Kidney Injury Using Electronic Medical Records".

## Requirements

To set up the environment, ensure you have the following:

- Python 3.10
- Libraries:
  - scikit-learn 1.3
  - pycaret 3.0.2
  - shap 0.42.1

Additionally, this project utilizes the [S3M (Statistically Significant Shapelet Mining)](https://github.com/BorgwardtLab/S3M) method for identifying significant shapelets in time series data. Follow the S3M installation instructions to set up your environment. Below is a brief installation guide:

```shell
wget https://github.com/BorgwardtLab/S3M/releases/download/v1.0.0-alpha/s3m-1.0.0-alpha.deb
sudo apt-get update
sudo dpkg -i s3m-1.0.0-alpha.deb
sudo apt --fix-broken install
sudo dpkg -i s3m-1.0.0-alpha.deb
```

## Dataset

The clinical data used in this study, derived from the HERON dataset, is de-identified and subject to privacy restrictions. Access requires institutional IRB and ethical approvals.

A sample dataset is provided at `/data/synthetic/`. This dataset has a time series with one feature. It is formatted as `m x (t + 1)`, where `m` is the number of samples and `t` is the number of timestamps. The first column shows the label.

## Usage

1. Run `1_process_data.ipynb` to process and clean the raw time series data.

2. Execute `2_run_s3m.py` for statistically significant shapelet mining.

3. Use `3_postprocess_data.ipynb` to convert mined shapelets into feature vectors and remove redundancy.

4. Run `4_machine_learning.py` for hyperparameter tuning and model training using XGBoost. For other machine learning models' initialization and tuning, we use the third-party library `pycaret` and follow its steps.

Note: The `ShapeletEvaluation.py` script and `s3m_eval.sh` shell script have been modified from the original versions in the S3M repository to better fit machine learning needs.

## Acknowledgements

We thank [BorgwardtLab](https://github.com/BorgwardtLab) for developing S3M and making their code publicly available.

## Contact

For any questions or concerns, please contact hongnianwang@gmail.com or open an issue in the repository.
