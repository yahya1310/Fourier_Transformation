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

#Lire une image
chemin_image = 'C:/Users/Farehan Yahya/Downloads/calimero.jpg'
image = Image.open(chemin_image).convert('L')#conversion en gris
image_Tab = np.array(image)
#affiche image original
plt.subplot(221), plt.imshow(image_Tab, cmap='gray'), plt.title('Original Image')
#plt.subplot(222), plt.imshow(np.log(1 + np.abs(np.fft.fft(np.fft.ifft(image_Tab)))), cmap='gray'), plt.title('')

def dft2D(I):
    M, N = I.shape
    dft_result = np.zeros((M, N), dtype=np.complex128)

    for u in range(M):
        for v in range(N):
            for x in range(M):
                for y in range(N):
                    dft_result[u, v] += I[x, y] * np.exp(-2j * np.pi * ((u * x) / M + (v * y) / N))

    return dft_result
#test

print("################ Test pour la transformée de fourier discrète directe #############\n\n")
# Exemple d'utilisation avec une matrice I
I = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

dft_result = dft2D(I)
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
   # Appliquer la FFT sur les colonnes
    cols = [FFT(row) for row in matrix]

    # Transposer les résultats pour préparer l'application de la FFT sur les lignes
    transposed = np.transpose(cols)

    # Appliquer la FFT sur les lignes
    rows = [FFT(col) for col in transposed]

    # Transposer à nouveau pour obtenir le résultat final
    result = np.transpose(rows)

    return result


#plt.subplot(222), plt.imshow(np.log(1 + np.abs(FFT2D(image_Tab))), cmap='gray'), plt.title('FFT Result')



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
    w = cmath.exp((2j * cmath.pi) / n)#pas de mois - puisque c'est l'inverse
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
    cols = [iFFT(row) for row in matrix]

    # Transposer les résultats pour préparer l'application de la iFFT sur les lignes
    transposed = np.transpose(cols)

    # Appliquer la iFFT sur les lignes
    rows = [iFFT(col) for col in transposed]

    # Transposer à nouveau pour obtenir le résultat final
    result = np.transpose(rows)

    return result
res = iFFT2D(image_Tab)
plt.subplot(222), plt.imshow(np.log(1 + np.abs(FFT2D(res))), cmap='gray')
print("\n",iFFT2D(FFT2D(matrix_2d)))
