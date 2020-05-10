import sys
from pyspark import SparkConf, SparkContext
import numpy as np
import json
import re

## Constants
APP_NAME = "CSE 545"

def cosin_dis(X, Y):
    mean_X = np.sum(X)/np.count_nonzero(X)
    mean_Y = np.sum(Y)/np.count_nonzero(Y)
    X[X != 0] -= mean_X
    Y[Y != 0] -= mean_Y
    return 1 - np.dot(X, Y)/(np.linalg.norm(X)*np.linalg.norm(Y))

def mapper(x, usrs):
    item = x[0]
    v = x[1]
    dict = {}
    for pair in v:
        dict[pair[0]] = pair[1]
    users = usrs.value
    rate = np.zeros((len(users), ))
    for i in range(len(users)):
        try:
            rate[i] = dict[users[i]]
        except KeyError:
            continue
    return (item, rate)

def intersection_count(X, Y):
    count = 0
    for i in range(len(X)):
        if X[i] != 0 and Y[i] != 0:
            count += 1
    return count


def main(sc, inputpath, products):
    rdd = sc.textFile(inputpath)
    rdd = rdd.map(lambda line: json.loads(line)) \
             .filter(lambda record: set(['overall', 'reviewerID', 'asin']).issubset(set(record.keys()))) \
             .map(lambda record: ((record['reviewerID'], record['asin']), record['overall'])) \
             .reduceByKey(lambda a, b: a) \
             .map(lambda x: (x[0][1], (x[0][0], x[1])))
    usr_count = rdd.countByKey()
    rdd = rdd.filter(lambda x: usr_count[x[0]] >= 25) \
             .map(lambda x: (x[1][0], (x[0], x[1][1])))
    item_count = rdd.countByKey()
    rdd = rdd.filter(lambda x: item_count[x[0]] >= 5)
    usrs = sc.broadcast(sorted(rdd.groupByKey().keys().collect()))

    rate_matrix = rdd.map(lambda x: (x[1][0], (x[0], x[1][1]))) \
                     .groupByKey() \
                     .map(lambda x: mapper(x, usrs)) \
                     .sortByKey()
    rate_matrix.persist()

    result = {}
    users = usrs.value
    for item in products:
        prediction = []
        main_row = rate_matrix.lookup(item)[0]
        similarity = rate_matrix.filter(lambda x: x[0] != item) \
                   .filter(lambda x: intersection_count(x[1], main_row) >= 2) \
                   .map(lambda x: (x[0], cosin_dis(main_row, x[1]))) \
                   .filter(lambda x: x[1] > 0) \
                   .sortBy(lambda x: x[1]) \
                   .take(50)
        dict = {}
        for pair in similarity:
            dict[pair[0]] = pair[1]
        sim_sum = 0
        count = 0
        pred = np.zeros((len(main_row, )))
        neighbors = {}
        for neighbor, sim in dict.items():
            neighbors[neighbor] = rate_matrix.lookup(neighbor)[0]
            pred = pred + sim * neighbors[neighbor]
        print(len(neighbors))
        for i in range(len(main_row)):
            if main_row[i] != 0:
                prediction.append(main_row[i])
            else:
                for neighbor, sim in dict.items():
                    if neighbors[neighbor][i] != 0:
                        sim_sum += sim
                        count += 1
                if count >= 2:
                    prediction.append(pred[i]/sim_sum)
                else:
                    prediction.append('Null')
        result[item] = [(users[i], prediction[i]) for i in range(len(prediction))]

    with open('output_2.txt', 'w') as file:
        for item, pred in result.items():
            file.write('Prediction for item ' + item + ':\n')
            i = 0
            for pair in pred:
                i += 1
                #file.write('(' + pair[0] + ' ' + str(pair[1]) + ')' + ' ')
                try:
                    file.write('(' + '{:.2f}'.format(float(pair[1])) + ')' + ' ')
                except ValueError:
                    file.write('(' + pair[1] + ')' + ' ')
                if i%20 == 0:
                    file.write('\n')
            file.write('\n\n')

if __name__ == "__main__":
   # Configure Spark
   conf = SparkConf().setAppName(APP_NAME).setMaster("local")
   sc = SparkContext(conf=conf)
   inputpath = sys.argv[1]
   products = re.findall(r'((?:[\.!?;"])|(?:(?:\#|\@)?[A-Za-z0-9_\-]+(?:\'[a-z]{1,3})?))', sys.argv[2])
   # Execute Main functionality
   main(sc, inputpath, products)