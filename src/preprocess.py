import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data():

    df = pd.read_csv("data/fraudTest.csv", low_memory=False)

    return df


def preprocess(df):

    drop_cols = [
        "first","last","street","trans_num",
        "cc_num","merchant","job","city"
    ]

    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # convert transaction date
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day"] = df["trans_date_trans_time"].dt.day
    df = df.drop("trans_date_trans_time", axis=1)

    # convert dob → age
    df["dob"] = pd.to_datetime(df["dob"])
    df["age"] = 2025 - df["dob"].dt.year
    df = df.drop("dob", axis=1)

    # encode categorical
    df = pd.get_dummies(df, columns=["category","gender","state"])

    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]

    numeric_cols = ["amt","lat","long","merch_lat","merch_long","city_pop"]

    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.2,random_state=42
    )

    return X_train,X_test,y_train,y_test