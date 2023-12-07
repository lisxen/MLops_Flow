import os
import pandas as pd
import numpy as np
import tensorflow as tf

path = "/home/ml_srv_admin/PycharmProjects/MLops_Flow/data/"

df_X_train = pd.read_csv(path + "step_1/X_train.csv")
df_X_test = pd.read_csv(path + "step_1/X_test.csv")
df_Y_train = pd.read_csv(path + "step_1/Y_train.csv")
df_Y_test = pd.read_csv(path + "step_1/Y_test.csv")

X_train = df_X_train.to_numpy()
X_test = df_X_test.to_numpy()
Y_train = df_Y_train.to_numpy()
Y_test = df_Y_test.to_numpy()

# нормализация данных
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# преобразование меток в формат one hot encoding
Y_train = tf.keras.utils.to_categorical(Y_train, 10)
Y_test = tf.keras.utils.to_categorical(Y_test, 10)

df_X_train = pd.DataFrame(X_train)
df_X_test = pd.DataFrame(X_test)
df_Y_train = pd.DataFrame(Y_train)
df_Y_test = pd.DataFrame(Y_test)

os.makedirs(path + "step_2", exist_ok=True)

df_X_train.to_csv(path + "step_2/X_train_preprocessed.csv", index=False)
df_X_test.to_csv(path + "step_2/X_test_preprocessed.csv", index=False)
df_Y_train.to_csv(path + "step_2/Y_train_preprocessed.csv", index=False)
df_Y_test.to_csv(path + "step_2/Y_test_preprocessed.csv", index=False)
