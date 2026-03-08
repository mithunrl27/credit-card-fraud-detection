from xgboost import XGBClassifier
import joblib
from preprocess import load_data, preprocess

df=load_data()

x_train,x_test,y_train,y_test=preprocess(df)

model=XGBClassifier(
    n_estimators=200,
    reg_lambda=0.2,
    reg_alpha=0.9,
    learning_rate=0.1
        )
model.fit(x_train,y_train)
joblib.dump(model,"model/fraud_model.pkl")
print("Trained Model Saved ")