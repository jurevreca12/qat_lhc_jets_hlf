from tensorflow.keras.utils import to_categorical
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

seed = 0
np.random.seed(seed)

# Based on https://github.com/fastmachinelearning/hls4ml-tutorial part1_getting_started.ipynb 
data = fetch_openml('hls4ml_lhc_jets_hlf')
X, y = data['data'], data['target']

le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y, 5)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_val = scaler.fit_transform(X_train_val)
X_test = scaler.transform(X_test)


#X_train_val = X_train_val * (2**6)
#X_test = X_test * (2**6)
#X_train_val = np.round(X_train_val)
#X_test = np.round(X_test)

np.save('X_train_val.npy', X_train_val)
np.save('X_test.npy', X_test)
np.save('y_train_val.npy', y_train_val)
np.save('y_test.npy', y_test)
np.save('classes.npy', le.classes_)
