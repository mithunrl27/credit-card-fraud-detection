# Credit Card Fraud Detection using Machine Learning

## Project Overview

Credit card fraud is a major challenge in the financial industry. This project uses machine learning techniques to detect fraudulent credit card transactions. The model analyzes transaction patterns and predicts whether a transaction is legitimate or fraudulent.

## Dataset

This project uses a credit card fraud dataset from Kaggle.

Dataset link:
https://www.kaggle.com/datasets/kartik2112/fraud-detection

Note: The dataset is not included in this repository because of its large size.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Flask

## Project Structure

```
credit-card-fraud-detection
│
├── src
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── visualize.py
│
├── model
│   └── fraud_model.pkl
│
├── data
│   └── (dataset not included)
│
├── app.py
├── requirements.txt
└── README.md
```

## Project Workflow

1. Data preprocessing
2. Feature scaling using StandardScaler
3. Model training using XGBoost
4. Model evaluation using accuracy and classification report
5. Visualization using Matplotlib
6. Model deployment using Flask API

## Model

The machine learning model used in this project is **XGBoost Classifier** for binary classification.

Target variable:

* 0 → Normal transaction
* 1 → Fraud transaction

## How to Run the Project

### 1 Install dependencies

```
pip install -r requirements.txt
```

### 2 Train the model

```
python src/train.py
```

### 3 Run visualization

```
python src/visualize.py
```

### 4 Start the Flask API

```
python app.py
```

Open browser:

```
http://127.0.0.1:5000
```

## API Endpoint

Prediction endpoint:

```
POST /predict
```

Example JSON input:

```
{
 "features":[0.3,0.2,0.5,0.7,0.1,0.6]
}
```

Example output:

```
{
 "prediction":"Fraud Transaction"
}
```

## Future Improvements

* Hyperparameter tuning
* Real-time fraud detection
* Deploy using Docker or cloud services
* Build a web interface for prediction

## Author

Mithun
Machine Learning Enthusiast
