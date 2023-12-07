import os
import pandas as pd
import pickle
import yaml
import mlflow
from mlflow.tracking import MlflowClient
from keras.models import Sequential
from keras.layers import Dense

os.environ["MLFLOW_REGISTRY_URI"] = "/home/ml_srv_admin/PycharmProjects/MLops_Flow/mlflow/"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("model_fit")

path = "/home/ml_srv_admin/PycharmProjects/MLops_Flow/"

df_X_train = pd.read_csv(path + "data/step_2/X_train_preprocessed.csv")
df_Y_train = pd.read_csv(path + "data/step_2/Y_train_preprocessed.csv")

# параметры модели
params = yaml.safe_load(open(path + "params.yaml"))
p_batch_size = params["batch_size"]
p_epochs = params["epochs"]
p_validation_split = params["validation_split"]
p_verbose = params["verbose"]

# создание последовательной модели
model = Sequential()
model.add(Dense(800, input_dim=784, activation='relu', kernel_initializer='normal'))
model.add(Dense(10, activation='softmax', kernel_initializer='normal'))

model.compile(loss='categorical_crossentropy',
              optimizer='SGD',
              metrics=['accuracy'])


with mlflow.start_run():
    mlflow.sklearn.log_model(model,
                            artifact_path="simple_model",
                            registered_model_name="simple_model")
    mlflow.log_artifact(local_path=path + "scripts/model_fit.py",
    			artifact_path="model_fit code")
    mlflow.end_run()

model.fit(df_X_train, df_Y_train,
	  batch_size=p_batch_size,
          epochs=p_epochs,
          validation_split=p_validation_split,
          verbose=p_verbose)

os.makedirs(path + "models", exist_ok=True)
with open(path + "models/simple_model.pickle", "wb") as f:
    pickle.dump(model, f)
