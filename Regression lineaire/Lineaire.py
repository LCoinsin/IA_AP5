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

# Calcul de l'erreur
erreurs = []
for i in range(0, len(X_test)):
    attendu = y_test[i]
    predit = y_predictions[i]
    erreurs.append(attendu-predit)
# print(erreurs)

# Choix de n observations
n = 8
print("X : ", X_test[:n])
print("Attendu : ", y_test[:n])
print("Predit : ", y_predictions[:n])
print("Erreur : ", erreurs[:n])

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#ME
me = (1 / len(erreurs)) * sum(erreurs)
print("ME : ", me)

#MAE
mae = (1/len(erreurs)) * sum(erreurs)
print("MAE sklearn : ", mean_absolute_error(y_test, y_predictions))

#MAPE
print("MAPE sklearn : ", mean_absolute_percentage_error(y_test, y_predictions))

#MSE
print("MSE sklearn : ", mean_squared_error(y_test, y_predictions))

#RMSE
print("RMSE sklearn : ", mean_squared_error(y_test, y_predictions, squared=False))

#R^2
print("R2 sklearn : ", r2_score(y_test, y_predictions))

from statsmodels.stats.stattools import durbin_watson

#Durbin watson
print("Durbin watson : ", durbin_watson(erreurs))

# Résidus ont une moyenne de 0 et normalement distribué
print("Meaen of residuals : ", np.mean(erreurs))
# H0 : Les résidus sont normalement distribué
# H2 : Les résidus ne sont pas normalement distribué

# P-value = Prob de faire une erreur en rejettant H0
from scipy import stats
shapiro_test = stats.shapiro(erreurs)
print("Shapiro p-value ", shapiro_test.pvalue)

if (shapiro_test.pvalue>0.05):
    print("Les residus sont normalement distribués")
else : 
    print("Les résidus en sont pas normalement distribués")

"""
figure(figsize=(8, 6), dpi=80)
plt.plot(erreurs, ".", color="black")
plt.axhline(y=0, color="r", linestyle='-')
plt.grid(axis="x", color="0.95")
plt.title("Recherche de pattern dans les erreurs")
plt.show()
"""

"""
Creation de la courbe matplotlib
"""
"""
figure(figsize=(8, 6), dpi=80)
plt.scatter(X_test, y_test, color='black')
plt.scatter(X_test, y_predictions, color='red')
plt.grid(axis='x', color='0.95')
plt.xlabel("Nombre produits vendus (Jeu de test)")
plt.ylabel("Chiffres d'affaires (Jeu de test)")
plt.show()
"""