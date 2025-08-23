# Risk Prediction for Cervical Lymph Node Metastasis in Esophageal Cancer

## 📌 Overview
This repository presents a machine learning (ML) framework developed to assess the risk of cervical lymph node metastasis in patients with **esophageal squamous cell carcinoma (ESCC)** using preoperative ultrasound and clinical information. The model aims to support personalized treatment planning and reduce unnecessary lymph node dissections. 

![Figure 1](/main/Figure%201.jpg)

## 🧬 Background
Cervical lymph node dissection in ESCC remains controversial due to its associated risks and uncertain survival benefits. Traditional imaging modalities like PET/CT are costly and often ineffective for occult metastasis. Ultrasound offers a non-invasive, accessible alternative, and when combined with ML, can significantly enhance diagnostic accuracy. 

## ⚙️ Usage

This repository serves two primary purposes:

1. **Reproduce Study Findings**: The source code enables full replication of our published methodology
2. **Apply Trained Models**: Pretrained models can be directly used for predictive inference on new patient data.

Additionally, this repository provides a platform to share our approach in detail, as described in a manuscript currently under peer review. Reference information will be updated upon publication.

## 1️⃣ Archive Code for Model Development

The folder `Archive Code` contains Python scripts used for model development, including data preprocessing, model training, and evaluation. These scripts allow full reproduction of the study’s methodology and results. 

Please note: the raw data used in this study is not publicly available due to institutional restrictions. For access, please contact the corresponding author.

## 2️⃣ Predictive Inference

You can use our trained models directly to predict cervical lymph node metastasis in new ESCC patient data. Follow the steps below to get started.

### 🧰 Prerequisites
Before you begin, ensure you have the following installed:

- Python 3.8 or above
- Jupyter Notebook
- Required Python libraries: pandas, numpy, scikit-learn, xgboost, lightgbm, joblib

You can install the required libraries using the following command:

```bash
pip install pandas numpy scikit-learn
pip install xgboost
pip install lightgbm
pip install joblib
```

### 🚀 Getting Started

1. Clone the repository or download the Jupyter Notebook `ECSS_ultrasound.ipynb` and the pre-trained models.
2. Launch Jupyter Notebook by running the following command in your terminal:
```bash
jupyter notebook
```
3. Open `ECSS_ultrasound.ipynb` from the directory.

### 📓 Notebook Structure

1. **Import Packages**: Load all required libraries.

2. **Data Input**: Enter patient data including age, sex, tumor size, and ultrasound findings. Select a pre-trained model.

3. **Preprocessing**: Automatically format input data for model compatibility.

4. **Model Loading**: Load the selected pre-trained ML model.

5. **Risk Prediction**: Generate and display the predicted risk of cervical lymph node metastasis.

### 🧪 Using the Notebook

Follow the step-by-step instructions in the notebook. Run each cell sequentially using `Shift + Enter`. Ensure all cells are executed in order for accurate results.


## 🔐 Data Privacy

Our model does not require any personally identifiable health information (PHI) to function. However, users are responsible for ensuring that all patient data is handled in compliance with applicable data privacy regulations. **Do not upload or share PHI in public repositories or forums.**

## 🤝 Support & Contributions

We are actively developing an automated pipeline to streamline the entire workflow—from data input to prediction output. Contributions are highly encouraged to improve usability, functionality, robustness, and clinical integration.

Feel free to fork the repository, submit pull requests, or open issues to share your ideas. Together, we can advance this tool to support precision oncology and improve patient care.

For questions or technical support, please contact the repository maintainer.

