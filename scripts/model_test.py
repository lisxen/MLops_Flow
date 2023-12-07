import os
import yaml
import json
import pickle
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

os.environ["MLFLOW_REGISTRY_URI"] = "/home/ml_srv_admin/PycharmProjects/MLops_Flow/mlflow/"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("model_test")

path = "/home/ml_srv_admin/PycharmProjects/MLops_Flow/"

# параметры модели
params = yaml.safe_load(open(path + "params.yaml"))
p_batch_size = params["batch_size"]
p_epochs = params["epochs"]
p_validation_split = params["validation_split"]
p_verbose = params["verbose"]

df_X_test = pd.read_csv(path + "data/step_2/X_test_preprocessed.csv")
df_Y_test = pd.read_csv(path + "data/step_2/Y_test_preprocessed.csv")

with open(path + "models/simple_model.pickle", "rb") as f:
    model = pickle.load(f)

scores = model.evaluate(df_X_test, df_Y_test, verbose=1)
acc = scores[1]

results = {'test_loss':scores[0], 'test_accuracy':scores[1]}
with open(path + "results.json", 'w') as file:
	json.dump(results, file)
	
with mlflow.start_run():
    mlflow.log_metric("test_accuracy", acc)
    mlflow.log_artifact(local_path=path + "scripts/model_test.py",
    			artifact_path="model_test code")
    mlflow.end_run()
