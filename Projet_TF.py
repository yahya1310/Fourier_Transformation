# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 14:38:15 2023

@author: Farehan Yahya
"""
from math import pi
import cmath
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def ajuster_taille_tableau(I):
    # Vérifier si la taille du tableau est déjà une puissance de 2
    taille_actuelle = len(I)
    puissance_deux = 1
    while puissance_deux < taille_actuelle:
        puissance_deux *= 2

    # Si la taille actuelle est déjà une puissance de 2, ne rien faire
    if puissance_deux == taille_actuelle:
        return I

    # Sinon, ajouter des zéros jusqu'à ce que la taille soit une puissance de 2
    nouveau_taille = puissance_deux
    I.extend([0] * (nouveau_taille - taille_actuelle))
    return I

# Exemple d'utilisation
I = [1, 2, 3,4,5]
I = ajuster_taille_tableau(I)
print(I)

def ajuster_taille_matrice(matrice):
    # Vérifier la taille des lignes et colonnes
    lignes_actuelles = len(matrice)
    colonnes_actuelles = len(matrice[0]) if lignes_actuelles > 0 else 0

    # Trouver la puissance de 2 supérieure ou égale au nombre de lignes et de colonnes
    puissance_deux_lignes = 1
    while puissance_deux_lignes < lignes_actuelles:
        puissance_deux_lignes *= 2

    puissance_deux_colonnes = 1
    while puissance_deux_colonnes < colonnes_actuelles:
        puissance_deux_colonnes *= 2

    # Si la taille actuelle est déjà une puissance de 2, ne rien faire
    if puissance_deux_lignes == lignes_actuelles and puissance_deux_colonnes == colonnes_actuelles:
        return matrice

    # Ajouter des lignes ou des colonnes de zéros au besoin
    nouvelle_taille_lignes = puissance_deux_lignes
    nouvelle_taille_colonnes = puissance_deux_colonnes

    for i in range(lignes_actuelles, nouvelle_taille_lignes):
        matrice.append([0] * colonnes_actuelles)

    for row in matrice:
        row.extend([0] * (nouvelle_taille_colonnes - colonnes_actuelles))

    return matrice

def ajuster_taille_image(image_path):
    # Ouvrir l'image avec Pillow
    image = Image.open(image_path)

    # Obtenir les dimensions actuelles de l'image
    largeur, hauteur = image.size

    # Trouver la puissance de 2 supérieure ou égale aux dimensions de l'image
    puissance_deux_largeur = 1
    while puissance_deux_largeur < largeur:
        puissance_deux_largeur *= 2

    puissance_deux_hauteur = 1
    while puissance_deux_hauteur < hauteur:
        puissance_deux_hauteur *= 2

    # Si la taille actuelle est déjà une puissance de 2, ne rien faire
    if puissance_deux_largeur == largeur and puissance_deux_hauteur == hauteur:
        return image

    # Ajuster la taille de l'image à la puissance de 2 la plus proche
    nouvelle_taille = (puissance_deux_largeur, puissance_deux_hauteur)
    image_redimensionnee = image.resize(nouvelle_taille)

    return image_redimensionnee

#Lire une image
chemin_image = 'C:/Users/Farehan Yahya/Downloads/calimero.jpg'
chemin_image2 =  'C:/Users/Farehan Yahya/Downloads/test.jpeg'


image = Image.open(chemin_image2).convert('L')  # Conversion en niveaux de gris

# Ajuster la taille de l'image
image_redimensionnee = ajuster_taille_image(chemin_image2).convert('L')

# Convertir l'image redimensionnée en tableau NumPy
image_tab_redimensionnee = np.array(image_redimensionnee)

image = Image.open(chemin_image2).convert('L')#conversion en gris
image_Tab = np.array(image)
#affiche image original
plt.subplot(221), plt.imshow(image_tab_redimensionnee, cmap='gray'), plt.title('Original Image')
#plt.subplot(222), plt.imshow(np.abs(np.fft.fft(np.fft.ifft(image_Tab))), cmap='gray'), plt.title('')

def dft2D(I):
    M = len(I)
    N = len(I[0])
    dft_result = [[0.0j for _ in range(N)] for _ in range(M)]

    for u in range(M):
        for v in range(N):
            for x in range(M):
                for y in range(N):
                    dft_result[u][v] += I[x][y] * cmath.exp(-2j * cmath.pi * ((u * x) / M + (v * y) / N))

    return dft_result
#test
#plt.subplot(222), plt.imshow(np.log(1 + np.abs(dft2D(image_Tab))), cmap='gray'), plt.title('FFT Result')
print("\n\n################ Test pour la transformée de fourier discrète directe #############\n\n")
# Exemple d'utilisation avec une matrice I
I = [[1, 2, 3],[4, 5, 6],[7, 8, 9]]

dft_result = dft2D(ajuster_taille_matrice(I))
print(dft_result)
print("\n")
# Display the DFT result
#plt.subplot(222), plt.imshow(np.log(1 + np.abs(dft2D(image_Tab))), cmap='gray'), plt.title('DFT Result')
# TF Rapide 1D
def FFT(I):
    n=len(I)
    if n==1:
        return I
    y=[0]*n
    I1 = I[0::2]
    I2 = I[1::2]
    ye,yo = FFT(I1),FFT(I2)  
    w = np.exp((-2j*pi)/n)
    for k in range(n//2):
            y[k] = ye[k] + w**k*yo[k]
            y[k+n//2] = ye[k] - w**k *yo[k]
    return y
        
#test
print("################ Test pour la transformée de fourier discrète rapide #############\n\n")
I = [1, 2, 3, 4,5,6,7,8]
print(FFT(I))
print("\n")

print("############## Comparaison avec la fft de la bibliothèque numpy ##############\n\n")
J = np.fft.fft(I)
print(J)
print("\n\n\n\n\n")



print("########## Test pour la transformée de fourier discrète rapide dimension 2 ##########\n\n")

# FFT 2D
def FFT2D(matrix):
   # Appliquer la FFT sur les lignes
    rows = [FFT(row) for row in matrix]

    # Transposer les résultats pour préparer l'application de la FFT sur les colonnes
    transposed = np.transpose(rows)

    # Appliquer la FFT sur les lignes (qui sont réelement les vrais colonnes)
    cols = [FFT(col) for col in transposed]

    # Transposer à nouveau pour obtenir le résultat final
    result = np.transpose(cols)

    return result


plt.subplot(222), plt.imshow(np.log(1000+np.abs(FFT2D(image_Tab))), cmap='gray'), plt.title('Résultat de la FFT2D')



print("\n")
# Test
matrix_2d = [
    [1, 2, 3, 4],
    [5, 4, 3, 2],
    [1, 2, 3, 4],
    [5, 4, 3, 2],
]

result = FFT2D(matrix_2d)
for row in result:
    print(row,"\n")

print("Comparaison avec fft2d de numpy")
# Appliquer la FFT2D avec numpy.fft.fft2
result_fft2d = np.fft.fft2(matrix_2d)

# Afficher le résultat
print(result_fft2d)

# La matrice result contient la FFT2D de l'image d'entrée

def iFFT(I):
    n = len(I)
    if n == 1:
        return I
    w = cmath.exp((2j * cmath.pi) / n)#pas de moins - puisque c'est l'inverse
    I1 = I[0::2]
    I2 = I[1::2]
    ye, yo = iFFT(I1), iFFT(I2)
    y = [0] * n
    for j in range(n // 2):
        y[j]= (ye[j] + (w ** j) * yo[j])/2#on divise par 2
        y[j + n // 2] = (ye[j] - w ** j * yo[j])/2 #On divise par 2
    return y


print("\nTest invers 1 d\n")

# Test

W = FFT(I)
result = iFFT(W)

print(result)


# iFFT 2D
def iFFT2D(matrix):
   # Appliquer la iFFT sur les colonnes
    rows = [iFFT(row) for row in matrix]

    # Transposer les résultats pour préparer l'application de la iFFT sur les lignes
    transposed = np.transpose(rows)

    # Appliquer la iFFT sur les lignes
    cols = [iFFT(col) for col in transposed]

    # Transposer à nouveau pour obtenir le résultat final
    result = np.transpose(cols)

    return result

res = FFT2D(image_tab_redimensionnee)
plt.subplot(224), plt.imshow(np.abs(iFFT2D(res)), cmap='gray'),plt.title('Reconstitution')


print("\n",iFFT2D(FFT2D(matrix_2d)))


for i in range(1,4):
    for j in  range(1,6):
        print(i*j)