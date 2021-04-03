from sklearn.preprocessing import StandardScaler
import numpy as np
X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])
scaler = StandardScaler().fit(X_train)
print(scaler.mean_)