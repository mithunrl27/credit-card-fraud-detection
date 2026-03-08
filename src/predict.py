import joblib
from preprocess import load_data, preprocess
from sklearn.metrics import classification_report,accuracy_score

df=load_data()
x_train,x_test,y_train,y_test=preprocess(df)
model = joblib.load("model/fraud_model.pkl")
y_pred_train=model.fit(x_train)
y_pred_test=model.fit(x_test)

print("Training Accuracy:",accuracy_score(y_pred_train,y_train))
print("Training Accuracy:",accuracy_score(y_pred_train,y_train))

print("Classification Report:",classification_report(y_pred_test,y_test))