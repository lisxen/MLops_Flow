import os
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

os.environ["MLFLOW_REGISTRY_URI"] = "/home/ml_srv_admin/PycharmProjects/MLops_Flow/mlflow/"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("data_preparation")

path = "/home/ml_srv_admin/PycharmProjects/MLops_Flow/"

with mlflow.start_run():

    df_X_train = pd.read_csv(path + "data/raw/X_train.csv")
    df_X_test = pd.read_csv(path + "data/raw/X_test.csv")
    df_Y_train = pd.read_csv(path + "data/raw/Y_train.csv")
    df_Y_test = pd.read_csv(path + "data/raw/Y_test.csv")

    mlflow.log_artifact(local_path="/home/ml_srv_admin/PycharmProjects/MLops_Flow/scripts/get_data.py",
                        artifact_path="get_data code")
    mlflow.end_run()

os.makedirs(path + "data/step_1", exist_ok=True)

df_X_train.to_csv(path + "data/step_1/X_train.csv", index=False)
df_X_test.to_csv(path + "data/step_1/X_test.csv", index=False)
df_Y_train.to_csv(path + "data/step_1/Y_train.csv", index=False)
df_Y_test.to_csv(path + "data/step_1/Y_test.csv", index=False)
