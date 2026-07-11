from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def split_features_target(df):

    X = df.drop("Class", axis=1)
    y = df["Class"]

    return X, y

def scale_features(X):

    X = X.copy()

    scaler = StandardScaler()

    X["Time"] = scaler.fit_transform(X[["Time"]])
    X["Amount"] = scaler.fit_transform(X[["Amount"]])

    return X


def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train):

    smote = SMOTE(random_state=42)

    X_train_smote, y_train_smote = smote.fit_resample(
        X_train,
        y_train
    )

    print("\nBefore SMOTE:")
    print(y_train.value_counts())

    print("\nAfter SMOTE:")
    print(y_train_smote.value_counts())

    return X_train_smote, y_train_smote