import numpy as np
from sklearn.datasets import make_regression
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import math

# Generation des données
X, y = make_regression(n_samples=100, n_features=1, n_informative=1, noise=9, coef=False)

# Mise a l'echelle des données par rapport au cas souhaité
# x: nb de produits vendus : 1 à 100
# y: montant du chiffres d'affaire : 1 à 100_000
X = np.interp(X, (X.min(), X.max()), (1, 100))
y = np.interp(y, (y.min(), y.max()), (1, 10_000))

# Existence correlation lineaire entre X et Y
mean_x = X.mean()
mean_y = y.mean()
cov = sum((xi - mean_x)*(yi - mean_y) for xi, yi in zip(X, y))/(len(X)-1)

# Train / Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, random_state=42)


"""
Manuel methode
"""
# Création de la matrice
matrice_train = []
for i in range(0, len(X_train)):
    # 1 : Constante
    # 2 : Xt (1, 2, 3, 4, 5, 6...)
    matrice_train.append([1, X_train[i][0]])

# -- Betas estimations --
XT = np.transpose(matrice_train)
XTX = np.matmul(XT, matrice_train)

# Calcul the inverse of the matrice
# The linalg inv Compute the (Moore_penrose) pseudo-inverse of a matrix
INV = np.linalg.inv(XTX)
INVXT = np.matmul(INV, XT)
betas = np.matmul(INVXT, y_train)
# print(betas)


"""
Automatique methode
"""
# Formule plus rapide ...
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(np.array(X_train).reshape(-1, 1), np.array(y_train).reshape(-1, 1))
# print("beta 0 :", reg.intercept_[0])
# print("beta 1 :", reg.coef_[0][0])
beta_0 = reg.intercept_[0]
beta_1 = reg.coef_[0][0]


"""
Evaluation du modèle
"""
def predire(x, beta_0, beta_1):
    y = beta_0 + beta_1*x
    return y[0]

y_predictions = []
for x in X_test:
    y = predire(x, beta_0, beta_1)
    y_predictions.append(y)


"""
Creation de la courbe matplotlib
"""
figure(figsize=(8, 6), dpi=80)
plt.scatter(X_test, y_test, color='black')
plt.scatter(X_test, y_predictions, color='red')
plt.grid(axis='x', color='0.95')
plt.xlabel("Nombre produits vendus (Jeu de test)")
plt.ylabel("Chiffres d'affaires (Jeu de test)")
plt.show()