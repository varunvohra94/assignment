import json
import pandas as pd
import os
import pickle
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report

import matplotlib.pyplot as plt

def read_json(path):
    with open(path, 'r') as f:
        file_dict = json.load(f)
    return file_dict


def read_csv(path: str):
    return pd.read_csv(path)


def separate_target_and_features(df: pd.DataFrame):
    df_X = df.drop("y", axis=1)
    df_label = df["y"]
    return df_X, df_label


def split_train_and_test(X: pd.DataFrame,
                         y: pd.DataFrame,
                         train_size: float,
                         random_state: int):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def separate_numerical_and_categorical_features(df: pd.DataFrame, numeric_features: list, categorical_features: list):
    numeric_features = df[numeric_features]
    categorical_features = df[categorical_features]
    return numeric_features, categorical_features


def fit_impute(df: pd.DataFrame, config):
    if config["imputer"]["type"] == "SimpleImpute":
        imputer = SimpleImputer(strategy=config["imputer"]["strategy"])
    elif config["imputer"]["type"] == "KNNImputer":
        imputer = KNNImputer(n_neighbors=config["imputer"]["n_neighbors"])
    else:
        raise("Imputer type not found")
    imputer.fit(df[config["numeric_features"]])
    return imputer


def impute_numeric_features(imputer, df: pd.DataFrame, config: dict):
    imputed_features = pd.DataFrame(imputer.transform(
        df[config["numeric_features"]]), columns=config["numeric_features"])
    return imputed_features


def fit_scalar(df: pd.DataFrame, config):
    scaler = StandardScaler()
    scaler.fit(df[config["numeric_features"]])
    return scaler


def scale_numeric_features(scaler, df: pd.DataFrame, config: dict):
    scaled_features = pd.DataFrame(scaler.transform(
        df), columns=config["numeric_features"])
    return scaled_features


def one_hot_encode(df, col):
    encoder = OneHotEncoder(handle_unknown="infrequent_if_exist")
    encoder.fit(df[[col]])
    encoded_df = pd.DataFrame(encoder.transform(df[[col]]))
    encoded_df = pd.DataFrame(encoder.transform(df[[col]]).toarray())
    encoded_df[col] = encoded_df[encoded_df.columns].values.tolist()
    return encoder, encoded_df[[col]]


def fit_ohencoders(df: pd.DataFrame(), config):
    encoders = []
    for col in config["categorical_features"]:
        encoder = OneHotEncoder(handle_unknown="infrequent_if_exist")
        encoder.fit(df[[col]])
        encoders.append(encoder)
    return encoders


def ohencode_features(encoders, df, config):
    encoded_features = []
    for encoder, col in zip(encoders, config["categorical_features"]):
        encoded_df = pd.DataFrame(encoder.transform(df[[col]]))
        encoded_df = pd.DataFrame(encoder.transform(df[[col]]).toarray())
        encoded_df.columns = [col + "_" + str(x) for x in encoded_df.columns]
        encoded_features.append(encoded_df)
    return encoded_features


def encode_categorical_features(df: pd.DataFrame, config: dict):
    encoders = []
    encoded_columns = []
    for col in config["categorical_features"]:
        encoder, encoded_column = one_hot_encode(df, col)
        encoders.append(encoder)
        encoded_columns.append(encoded_column)
    return encoders, encoded_columns


def concat_dfs(df_list: list):
    return pd.concat(df_list, axis=1)

def save_transformers(transformer,file_name):
    with open(file_name,"wb") as f:
        pickle.dump(transformer,f,protocol=pickle.HIGHEST_PROTOCOL)

def load_transformers(file_name):
    with open(file_name,"rb") as f:
        return pickle.load(f)

def train_model(X,y,config):
    if config["model"]["type"] == "LogisticRegression":
        clf = LogisticRegression(max_iter=config["model"]["max_iter"])
        clf.fit(X,y)
        return clf

def save_model(model,config):
    with open(os.path.join(config["model_dir"],"model.pkl"),"wb") as f:
        pickle.dump(model,f,protocol=pickle.HIGHEST_PROTOCOL)

def get_conf_matrix(model,X_test,y_test):
    return confusion_matrix(y_test, model.predict(X_test))

def confusion_matrix_fig(conf_matrix):
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    return fig

def get_roc_auc(model,X_test,y_test):
    tprobs = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, tprobs)

