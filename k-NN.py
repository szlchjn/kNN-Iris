import numpy as np
from random import random
from heapq import nsmallest
import csv

def load_data(file, split, training_data=[], test_data=[]):
    with open(file, 'r') as csv_file:
        data_set = list(csv.reader(csv_file))
        for i in range(len(data_set)):
            for j in range(4):
                data_set[i][j] = float(data_set[i][j])
            if random() < split:
                training_data.append(data_set[i])
            else:
                test_data.append(data_set[i])

def euclid_distance(a, b):
    return np.sqrt(sum([(a[i] - b[i])**2 for i in range(3)]))

def get_k_neighbours(test_instance, training_data, k):
    distances = [(item, euclid_distance(test_instance, item)) for item in training_data]
    return nsmallest(k, distances, key=lambda x: x[1])

def predict(neighbours):
    pool = {}
    for el in neighbours:
        vote = el[0][-1]
        if vote in pool:
            pool[vote] += 1
        else:
            pool[vote] = 1
    return max(pool, key=pool.get)

def is_correct(prediction, instance):
    return prediction == instance[-1]

def accuracy(test_data, predictions):
    return (predictions / len(test_data)) * 100.0

def main():
    training_data = []
    test_data = []
    correct = 0
    k = 4

    load_data('iris.data', 0.67, training_data, test_data)

    for instance in test_data:
        neighbours = get_k_neighbours(instance, training_data, k)
        prediction = predict(neighbours)

        if is_correct(prediction, instance):
            print('Prediction:', prediction, '-> Label:', instance[-1])
            correct += 1
        else:
            print('Prediction:', prediction, '-> Label:', instance[-1], 'FALSE')

    acc = accuracy(test_data, correct)
    print('Accuracy: {} %'.format(round(acc, 5)))


if __name__ == '__main__':
    main()