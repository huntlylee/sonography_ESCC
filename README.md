# Risk Prediction for Cervical Lymph Node Metastasis in Esophageal Cancer

## 🧠 Project Overview
This repository presents a machine learning (ML) framework developed to assess the risk of cervical lymph node metastasis in patients with esophageal squamous cell carcinoma (ESCC) using preoperative ultrasound and clinical data. The model aims to support personalized treatment planning and reduce unnecessary lymph node dissections. 

![Figure 1](https://github.com/user-attachments/assets/d2fbf119-86b4-4fa3-a05c-4643e66edfbb)

## 📚 Background
Cervical lymph node dissection in ESCC remains controversial due to its associated risks and uncertain survival benefits. Traditional imaging modalities like PET/CT are costly and often ineffective for occult metastasis. Ultrasound offers a non-invasive, accessible alternative, and when combined with ML, can significantly enhance diagnostic accuracy. 

## Usage

The source code in this repository enables two things, 1. full reproduction of our study findings, including data preprocessing, model training, and evaluation. 2. to apply trained models directly to new patient data for predictive inference. This repo also serves as the platform to share our methodology with details described in a paper that is currently being reviewed in a journal. Futher reference will be updated once the paper is published.

## 1. archive code for model development

Please see the folder xx for Python scripts.

## 2. predictive inference

Please follow the following instructions to use our trained model directly for your data.

### Prerequisites
Before you begin, ensure you have the following installed:

-Python 3.8 or above
-Jupyter Notebook
-Required Python libraries: pandas, numpy, scikit-learn, matplotlib, xgboost, lightgbm, joblib

You can install the required libraries using the following command:

```bash
pip install pandas numpy scikit-learn matplotlib
pip install xgboost
pip install lightgbm
pip install joblib
```

### Getting Started

1. Clone the repository or download the Jupyter Notebook `Tutorial.ipynb` and the pre-trained models.
2. Launch Jupyter Notebook by running the following command in your terminal:

```bash
jupyter notebook
```

3. Navigate to the directory containing the downloaded notebook and open it.

### Notebook Structure

The notebook is structured as follows:

1. **Import**: Import all required packages.

2. **Data Input**: A section for inputting user data. Please enter all required clinical features, such as age, sex, tumor size, and ultrasound findings, into the proper fields. You may also select from different pre-trained machine learning models.

3. **Data Preprocessing**: To automatically preprocess the above user inputs to the standard data format recognized by the model

4. **Model Loading**: Code to load the pre-trained machine learning model.

5. **Risk Prediction**: The notebook will use the input data to predict the risk of developing thyroid nodules and display the result.

### Using the Notebook

Follow the instructions within the notebook to input the required data. Each cell can be executed by selecting it and pressing `Shift + Enter`. Ensure that you run the cells in the order they appear.


## Data Privacy

Our model does not require any PHI to function. Please note that any patient data should be handled in accordance with relevant data privacy regulations. Do not share personal health information in public repositories or forums.

## Support

We are actively developing an automated pipeline to streamline the entire workflow—from data input to prediction output. Contributions are warmly welcomed to enhance its easy-to-use, functionality, robustness, and clinical integration.

Feel free to fork the repository, submit pull requests, or open issues to share your ideas and help advance this tool for precision oncology.

For any issues or questions regarding the project, please contact the repository maintainer.

