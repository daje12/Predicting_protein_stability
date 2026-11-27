import pandas as pd
import numpy as np
import matplotlib as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

HYDROPATHY = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}
VOLUME = {
    "A": 88.6, "C": 108.5, "D": 111.1, "E": 138.4, "F": 189.9,
    "G": 60.1, "H": 153.2, "I": 166.7, "K": 168.6, "L": 166.7,
    "M": 162.9, "N": 114.1, "P": 112.7, "Q": 143.8, "R": 173.4,
    "S": 89.0, "T": 116.1, "V": 140.0, "W": 227.8, "Y": 193.6,
}


s2648_df = pd.read_csv("s2648.csv")

#create copy of original data to create feature columns
X = s2648_df.copy()
y = s2648_df["DDG"].copy()

#create feature columns
X["WT"] = X["MUT"].str[0]
X["MT"] = X["MUT"].str[-1]
X["Hydropathy change"] = X["MT"].map(HYDROPATHY) - X["WT"].map(HYDROPATHY)
X["Size change"] = X["MT"].map(VOLUME) - X["WT"].map(VOLUME)

#one hot encoding so the model can learn patterns from mutation types
X = pd.get_dummies(X, columns=["WT", "MT"], drop_first=True)

#drop non-numeric columns and label column
X = X.drop(["CLID", "PDB", "CHAIN", "MUT", "DDG"], axis=1)
#could use Group shuffle split evaluate the use of this
#have the model ignore the CLID, PDB, CHAIN columns, use one hot encoding for mutations
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#normalize features so large ranges don't skew smaller ones during regularisation
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

#scikit learn has a method: RidgeCV, which automatically performs ridge with cross validation
#usualy alpha values are chosen on a logarithmic scale because regularization controls penalty strength exponentially
alphas = [0.01, 0.1, 1, 10, 100, 1000]
#use this method to tune any hyper parameter
def ridge_hyperparameter(array):
    best_score = -float('inf')
    argmin_a = 0
    
    for a in array:
        #fit model
        model = Ridge(alpha=a)
        model.fit(X_train, y_train)

        #cross-validate using mean squared error score
        #evaluate use of this scoring method
        cv_scores = cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error")
        mean_score = np.mean(cv_scores)

        #find alpha which minimises error
        if mean_score > best_score:
            best_score = mean_score
            argmin_a = a

    print(f"best_score: {best_score}, argmin: {argmin_a}")

    return argmin_a

#use tuned hyper parameter to fit final model
a = ridge_hyperparameter(alphas)
ridge_model = Ridge(alpha=a)
ridge_model.fit(X_train, y_train)

#optimisation

y_pred_train = ridge_model.predict(X_train)
mse_training = mean_squared_error(y_train, y_pred_train)
rmse_training = root_mean_squared_error(y_train, y_pred_train)

y_pred_test = ridge_model.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred_test)
rmse_test = root_mean_squared_error(y_test, y_pred_test)

print(f"ridge_training error: \n MSE:{mse_training}\n RMSE:{rmse_training} ")
print(f"\nridge_test error: \n MSE:{mse_test}\n RMSE:{rmse_test} ")