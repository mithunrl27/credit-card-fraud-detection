import pandas as pd
import matplotlib.pyplot as plt

# load dataset
df = pd.read_csv("data/fraudTrain.csv")

# 1️⃣ Fraud vs Normal transactions
fraud_counts = df["is_fraud"].value_counts()

plt.figure()
plt.bar(fraud_counts.index, fraud_counts.values)
plt.title("Fraud vs Normal Transactions")
plt.xlabel("Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Count")

plt.show()


# 2️⃣ Transaction Amount Distribution
plt.figure()

plt.hist(df["amt"], bins=50)

plt.title("Transaction Amount Distribution")
plt.xlabel("Amount")
plt.ylabel("Frequency")

plt.show()


# 3️⃣ Fraud transactions by category
fraud_data = df[df["is_fraud"] == 1]

category_counts = fraud_data["category"].value_counts().head(10)

plt.figure()

plt.bar(category_counts.index, category_counts.values)

plt.title("Top Fraud Categories")
plt.xlabel("Category")
plt.ylabel("Fraud Count")

plt.xticks(rotation=45)

plt.show()
