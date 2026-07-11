import os
import matplotlib.pyplot as plt
import seaborn as sns


def dataset_shape(df):
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

def dataset_info(df):
    df.info()


def missing_values(df):
    print(df.isnull().sum())


def class_distribution(df):

    counts = df["Class"].value_counts()

    print(f"Genuine Transactions : {counts[0]}")
    print(f"Fraud Transactions   : {counts[1]}")

    fraud_percentage = (counts[1] / len(df)) * 100

    print(f"Fraud Percentage     : {fraud_percentage:.4f}%")

    os.makedirs("../outputs", exist_ok=True)

    plt.figure(figsize=(6, 4))
    sns.countplot(x="Class", data=df)

    plt.title("Class Distribution")
    plt.savefig(
        "../outputs/class_distribution.png",
        bbox_inches="tight"
    )

    plt.show()
