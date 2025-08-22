# Risk Prediction for Cervical Lymph Node Metastasis in Esophageal Cancer

## 🧠 Project Overview
This repository presents a machine learning (ML) framework developed to predict cervical lymph node metastasis in patients with esophageal squamous cell carcinoma (ESCC) using preoperative ultrasound and clinical data. The model aims to support personalized surgical planning and reduce unnecessary lymph node dissections.

## 📚 Background
Cervical lymph node dissection in ESCC remains controversial due to its associated risks and uncertain survival benefits. Traditional imaging modalities like PET/CT are costly and often ineffective for small nodes. Ultrasound offers a non-invasive, accessible alternative, and when combined with ML, can significantly enhance diagnostic accuracy.


## Prerequisites
Before you begin, ensure you have the following installed:

-Python 3.8 or above
-Jupyter Notebook
-Required Python libraries: pandas, numpy, scikit-learn, matplotlib, xgboost, lightgbm, joblib

You can install the required libraries using the following command:

pip install pandas numpy scikit-learn matplotlib
pip install xgboost
pip install lightgbm
pip install joblib

## Getting Started

1. Clone the repository or download the Jupyter Notebook `Tutorial.ipynb` and the pre-trained models.
2. Launch Jupyter Notebook by running the following command in your terminal:

```bash
jupyter notebook
```

3. Navigate to the directory containing the downloaded notebook and open it.

## Notebook Structure

The notebook is structured as follows:

1. **Import**: Import all required packages.

2. **Data Input**: A section for inputting user data. Please enter all required clinical features, such as age, HDL-C levels, FBG levels, and creatinine levels, into the proper fields. You may also select from different pre-trained machine learning models.

3. **Data Preprocessing**: To automatically preprocess the above user inputs to the standard data format recognized by the model

4. **Model Loading**: Code to load the pre-trained machine learning model.

5. **Risk Prediction**: The notebook will use the input data to predict the risk of developing thyroid nodules and display the result.

## Using the Notebook

Follow the instructions within the notebook to input the required data. Each cell can be executed by selecting it and pressing `Shift + Enter`. Ensure that you run the cells in the order they appear.

## Support

For any issues or questions regarding the project, please contact the repository maintainer.

## Data Privacy

Please note that any input data you provide should be handled in accordance with relevant data privacy regulations. Do not share personal health information in public repositories or forums.
