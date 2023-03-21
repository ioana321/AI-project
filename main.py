import random

from sklearn import neural_network
import numpy as np
from sklearn.metrics import accuracy_score

nr_test = 60
nr_linii = 120

date_intrare = np.loadtxt("diagnosis.data", 'float')
date_intrare_test = np.zeros(shape=(nr_test, 8))

for i in range(nr_test):
    j = random.randrange(nr_linii)
    date_intrare_test[i] = date_intrare[j]
    date_intrare = np.delete(date_intrare, j, 0)
    nr_linii -= 1

etichete_train = date_intrare[:, 6:]
date_train = np.delete(date_intrare, [6, 7], 1)

etichete_test = date_intrare_test[:, 6:]
date_test = np.delete(date_intrare_test, [6, 7], 1)

clf = neural_network.MLPClassifier(hidden_layer_sizes=(3, ), learning_rate_init=0.1)
clf.fit(date_train, etichete_train)

predictii = clf.predict(date_test)

print('Accuracy:')
print(accuracy_score(etichete_test, predictii)*100, '%')