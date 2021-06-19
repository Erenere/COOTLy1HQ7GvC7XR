
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib
from sklearn.model_selection import train_test_split
import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow import keras

df=pd.read_csv("/content/ACME-HappinessSurvey2020.csv")

dftrain = df.copy()
dfeval = df.copy()

y_train = dftrain.pop('Y')
y_eval = dfeval.pop('Y')
for y in y_train:
  y=int(y)
for y in y_eval:
  y=int(y)


def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.Adam(0.001),
                metrics=["accuracy"])
  return model

normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(dftrain))

dnn_model = build_and_compile_model(normalizer)

history = dnn_model.fit(
    dftrain, y_train,
    validation_split=0.2,
    verbose=0, epochs=100)

dnn_model.save('dnn_model')
dnn_model.evaluate(dfeval, y_eval, verbose=1)

# Feature Extraction with RFE
from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='lbfgs')
rfe = RFE(model, 3)
fit = rfe.fit(dftrain, y_train)
print("\n\n\nNum Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)

#Top three features are X1 X3 and X5