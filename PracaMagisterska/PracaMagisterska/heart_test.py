from SimilarityForest import SimilarityForest
from datasets.heart import HeartDataset as hds
from sklearn.model_selection import train_test_split
import numpy as np


vectors, labels = hds.convert_file("heart_scale")

X_train, X_test, y_train, y_test = train_test_split(vectors, labels, test_size=0.20)

print(" --- TEST 1 --- ")
sf_1 = SimilarityForest(5, 25, np.dot)

print(" --- FIT --- ")
sf_1.fit(X_train, y_train)

print (" --- PREDICT --- ")
sf_1.predict(X_test)

print (" ---- CONFUSION MATRIX ---- ")
matrix_1 = sf_1.get_confusion_matrix(y_test)
print(matrix_1)


print(" --- TEST 2 --- ")
sf_2 = SimilarityForest(15, 50, np.dot)

print(" --- FIT --- ")
sf_2.fit(X_train, y_train)

print (" --- PREDICT --- ")
sf_2.predict(X_test)

print (" ---- CONFUSION MATRIX ---- ")
matrix_2 = sf_2.get_confusion_matrix(y_test)
print(matrix_2)


print(" --- TEST 3 --- ")
sf_3 = SimilarityForest(30, 50, np.dot)

print(" --- FIT --- ")
sf_3.fit(X_train, y_train)

print (" --- PREDICT --- ")
sf_3.predict(X_test)

print (" ---- CONFUSION MATRIX ---- ")
matrix_3 = sf_3.get_confusion_matrix(y_test)
print(matrix_3)


print(" --- TEST 4 --- ")
sf_4 = SimilarityForest(50, 80, np.dot)

print(" --- FIT --- ")
sf_4.fit(X_train, y_train)

print (" --- PREDICT --- ")
sf_4.predict(X_test)

print (" ---- CONFUSION MATRIX ---- ")
matrix_4 = sf_4.get_confusion_matrix(y_test)
print(matrix_4)


print(" --- TEST 5 --- ")
sf_5 = SimilarityForest(100, 180, np.dot)

print(" --- FIT --- ")
sf_5.fit(X_train, y_train)

print (" --- PREDICT --- ")
sf_5.predict(X_test)

print (" ---- CONFUSION MATRIX ---- ")
matrix_5 = sf_5.get_confusion_matrix(y_test)
print(matrix_5)


print(" --- TEST 6 --- ")
sf_6 = SimilarityForest(10, 180, np.dot)

print(" --- FIT --- ")
sf_6.fit(X_train, y_train)

print (" --- PREDICT --- ")
sf_6.predict(X_test)

print (" ---- CONFUSION MATRIX ---- ")
matrix_6 = sf_6.get_confusion_matrix(y_test)
print(matrix_6)


print(" --- TEST 7 --- ")
sf_7 = SimilarityForest(15, 250, np.dot)

print(" --- FIT --- ")
sf_7.fit(X_train, y_train)

print (" --- PREDICT --- ")
sf_7.predict(X_test)

print (" ---- CONFUSION MATRIX ---- ")
matrix_7 = sf_7.get_confusion_matrix(y_test)
print(matrix_7)


print(" --- TEST 8 --- ")
sf_8 = SimilarityForest(25, 300, np.dot)

print(" --- FIT --- ")
sf_8.fit(X_train, y_train)

print (" --- PREDICT --- ")
sf_8.predict(X_test)

print (" ---- CONFUSION MATRIX ---- ")
matrix_8 = sf_8.get_confusion_matrix(y_test)
print(matrix_8)


print(" --- TEST 9 --- ")
sf_9 = SimilarityForest(45, 360, np.dot)

print(" --- FIT --- ")
sf_9.fit(X_train, y_train)

print (" --- PREDICT --- ")
sf_9.predict(X_test)

print (" ---- CONFUSION MATRIX ---- ")
matrix_9 = sf_9.get_confusion_matrix(y_test)
print(matrix_9)


print(" --- TEST 10 --- ")
sf_10 = SimilarityForest(45, 720, np.dot)

print(" --- FIT --- ")
sf_10.fit(X_train, y_train)

print (" --- PREDICT --- ")
sf_10.predict(X_test)

print (" ---- CONFUSION MATRIX ---- ")
matrix_10 = sf_10.get_confusion_matrix(y_test)
print(matrix_10)