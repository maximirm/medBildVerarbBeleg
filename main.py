# Aufgabe 1.1: Laden Sie das Testbild 'CT.png', wandeln es in ein Bild mit einem Kanal um und stellen es mittels
# matplotlib.pyplot mit der Funktion imshow und der Option cmap='gray' dar.

# Aufgabe 1.1
import matplotlib.pyplot as plt
import cv2
import numpy as np


def print_result(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.imshow(image, cmap='gray')
    plt.show()


print_result('CT.png')


# Aufgabe 1.2: Geben Sie die Dimensionen des Bildes (Breite und Höhe) in Pixel sowie den jeweils
# kleinsten und größten Wert im Bild aus.
# Aufgabe 1.2
def print_result(image_path):
    image = cv2.imread(image_path)
    print('Höhe: ', image.shape[0], 'Pixel')
    print('Breite: ', image.shape[1], 'Pixel')
    print('min: ', np.min(image[:, :]))
    print('max: ', np.max(image[:, :]))


print_result('CT.png')


# Aufgabe 1.3: Als nächstes soll im CT Bild der schwarze Rand abgeschnitten werden, da dort keine Informationen
# über die aufgenommenen Strukturen zu finden sind. Zu diesem Zweck sollen Sie eine Bounding Box um den relevanten
# Bereich bestimmen. Dabei handelt es sich idealerweise um das kleinste parallel zu den Achsen verlaufendes Rechteck,
# dass sich über alle Bildbereiche erstreckt, in denen Inhalte zu finden sind (hier Pixel die nich schwarz sind).
# Nutzen Sie nun die gefundene Bounding Box, um das Bild entsprechend zuzuschneiden (oder auch zu "croppen").

# Aufgabe 1.3
def find_bounding_box(img):
    non_zero_values = np.where(img != 0)
    x_max = np.max(non_zero_values[0])
    x_min = np.min(non_zero_values[1])
    y_max = np.max(non_zero_values[1])
    y_min = np.min(non_zero_values[0])
    return x_min, x_max, y_min, y_max


def crop_image(img, bounding_box):
    cropped_img = np.delete(img, slice(bounding_box[3], img.shape[1]), 1)
    cropped_img = np.delete(cropped_img, slice(bounding_box[1], cropped_img.shape[0]), 0)
    cropped_img = np.delete(cropped_img, slice(0, bounding_box[0]), 1)
    cropped_img = np.delete(cropped_img, slice(0, bounding_box[2]), 0)
    return cropped_img


def print_result(image_path):
    image = cv2.imread(image_path)
    cropped_image = crop_image(image, find_bounding_box(image))
    plt.imshow(cropped_image)
    plt.show()


print_result('CT.png')


# Implementieren Sie den Median-Cut Algorithmus so dass die Method sowohl für Grauwertebilder
# (ein Kanal) als auch Farbbilder (3 Kanäle) anwendbar ist.

def median_cut(image_path, depth):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cluster = convert_image_to_cluster(image)
    clustering(image, cluster, depth)
    return image


def convert_image_to_cluster(image):
    list = []
    for rindex, rows in enumerate(image):
        for cindex, color in enumerate(rows):
            list.append([color[0], color[1], color[2], rindex, cindex])
    return np.array(list)


def clustering(image, cluster, depth):
    if depth == 0:
        rebuild_image(image, cluster)
        return

    highest_range = get_highest_range(cluster)
    sorted_cluster = sort_cluster_by_range(cluster, highest_range)
    median_index = find_median_index(sorted_cluster)

    clustering(image, sorted_cluster[0:median_index], depth - 1)
    clustering(image, sorted_cluster[median_index:], depth - 1)


def rebuild_image(image, cluster):
    r_average = np.mean(cluster[:, 0])
    g_average = np.mean(cluster[:, 1])
    b_average = np.mean(cluster[:, 2])

    for data in cluster:
        image[data[3]][data[4]] = [r_average, g_average, b_average]


def get_highest_range(cluster):
    r_max = np.max(cluster[:, 0]) - np.min(cluster[:, 0])
    g_max = np.max(cluster[:, 1]) - np.min(cluster[:, 1])
    b_max = np.max(cluster[:, 2]) - np.min(cluster[:, 2])

    if g_max >= r_max and g_max >= b_max:
        return 1
    elif b_max >= r_max and b_max >= g_max:
        return 2
    else:
        return 0


def sort_cluster_by_range(cluster, highest_range):
    sorted_cluster = cluster[cluster[:, highest_range].argsort()]
    return sorted_cluster


def find_median_index(sorted_cluster):
    return int((len(sorted_cluster) + 1) / 2)


# Reduzieren Sie die Farben / Grauwert in den Bildern Baboon.png und Lena.png auf 2,4,8,16 mittels
# des Median-Cut Algorithmus und stellen Sie das Ergebnis jeweils dar.

# Aufgabe 2.2
def print_result(paths, colors):
    for path in paths:
        for color in colors:
            print('number of colors: ', color)
            plt.imshow(median_cut(path, color / 2))
            plt.show()


colors = [2, 4, 8, 16]
paths = [
    'Baboon.png',
    'Lena.png'
]
print_result(paths, colors)

# Aufgabe 3.1: Implementieren Sie eine Funktion welche einen beliebigen Faltungskern auf ein Grauwerte oder Farbbild
# anwendet. Der Faltungskern und das Bild sollen hierbei Übergabewerte für die Funktion sein.


# Aufgabe 3.1
from scipy.ndimage import correlate


def convolve(image, kernel):
    image_as_array = np.array(image)
    result = []
    for i in range(image.shape[2]):
        result.append(correlate(image_as_array[:, :, i], kernel, int, 'mirror'))
    return np.moveaxis(result, 0, 2)


# Aufgabe 3.2: Implementieren Sie eine Funktion welche den Faltungskern für den Binomialfilter erzeugt.
# Die Dimension des Filterkerns soll hierbei frei wählbar sein und der Funktion übergeben werden.
# Achten Sie auf die Normierung des Kerns

import scipy as sp


def get_binom_kernel(n):
    vec1 = get_binom_vector(n).reshape(-1, 1)
    vec2 = get_binom_vector(n).reshape(1, -1)
    kern = vec1.dot(vec2)
    return kern / np.sum(kern)


def get_binom_vector(n):
    return np.array([sp.special.binom(n - 1, i) for i in range(n)])


# Aufgabe 3.3 Wenden Sie auf das Bild 'Lena.png' Binomialfilter der Größen 3x3, 5x5, 11x11, 21x21 an
# und stellen Sie die Resultate dar.

def print_result(image_path, filter_sizes):
    for size in filter_sizes:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(size, 'x', size)
        plt.imshow(convolve(image, get_binom_kernel(size)))
        plt.show()


image_path = 'Lena.png'
filter_sizes = [3, 5, 11, 21]
print_result(image_path, filter_sizes)
