import sys
from pyspark import SparkConf, SparkContext
import numpy as np
from scipy import stats
import json
import re

## Constants
APP_NAME = "CSE 545"

def mapper(record, word_list):
    #word_list = set(word_list.value)
    rate = float(record["overall"])
    verified = 1 if record["verified"] == "true" else 0
    review = re.findall(r'((?:[\.,!?;"])|(?:(?:\#|\@)?[A-Za-z0-9_\-]+(?:\'[a-z]{1,3})?))', record['reviewText'].lower(), re.M)
    freq = {}
    sum = len(review)
    for word in word_list.value:
        freq[word] = 0
    for word in review:
        try:
            freq[word] += (1 / sum)
        except KeyError:
            continue
    return [(word, [f, rate, verified]) for word, f in freq.items()]

def linearRegression(word, l, controlled=False):
    if controlled:
        X = np.column_stack(([1]*len(l[0]), l[0], l[1]))
        Y = np.array(l[1])
        beta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)
        Y_yield = np.dot(X, beta)
        RSS = np.linalg.norm(Y-Y_yield)**2
        t = beta[1] / np.sqrt(RSS / ((len(Y) - 2) * np.linalg.norm(X[:, 1] - X[:, 1].mean())**2))
        pval = stats.t.sf(np.abs(t), len(Y) - 3)*2*1000
    else:
        X = np.column_stack(([1]*len(l[0]), l[0]))
        Y = np.array(l[1])
        beta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)
        Y_yield = np.dot(X, beta)
        RSS = np.linalg.norm(Y-Y_yield)**2
        t = beta[1] / np.sqrt(RSS / ((len(Y) - 2) * np.linalg.norm(X[:, 1] - X[:, 1].mean())**2))
        pval = stats.t.sf(np.abs(t), len(Y) - 2)*2*1000
    return (word, beta[1], pval)
    

def main(sc, inputpath):
    rdd = sc.textFile(inputpath)
    jsonRDD = rdd.map(lambda line: json.loads(line)) \
               .filter(lambda record: set(['reviewText', 'overall']).issubset(set(record.keys())))

    jsonRDD.persist()

    wordsRDD = jsonRDD.flatMap(lambda record: re.findall(r'((?:[\.,!?;"])|(?:(?:\#|\@)?[A-Za-z0-9_\-]+(?:\'[a-z]{1,3})?))', record['reviewText'].lower(), re.M)) \
               .map(lambda word: (word, 1)) \
               .reduceByKey(lambda a, b: a + b)
    word_list = wordsRDD.sortBy(lambda x: x[1], ascending=False).keys().take(1000)
    word_list = sc.broadcast(word_list)

    frequencyRDD = jsonRDD.flatMap(lambda record: mapper(record, word_list)) \
                          .reduceByKey(lambda l1, l2: np.column_stack((l1, l2)))
    #frequencyRDD.saveAsTextFile('/user/jyao/output')

    frequencyRDD.persist()

    correlationRDD = frequencyRDD.map(lambda x: linearRegression(x[0], x[1], False))
    pc_word = correlationRDD.filter(lambda x: x[1] > 0) \
                            .sortBy(lambda x: x[2], ascending=True) \
                            .take(20)
    nc_word = correlationRDD.filter(lambda x: x[1] < 0) \
                            .sortBy(lambda x: x[2], ascending=True) \
                            .take(20)

    controlled_corRDD = frequencyRDD.map(lambda x: linearRegression(x[0], x[1], True))
    controlled_pc_word = controlled_corRDD.filter(lambda x: x[1] > 0) \
                                          .sortBy(lambda x: x[2], ascending=True) \
                                          .take(20)
    controlled_nc_word = controlled_corRDD.filter(lambda x: x[1] < 0) \
                                          .sortBy(lambda x: x[2], ascending=True) \
                                          .take(20)
    
    with open('output.txt', 'w') as file:
        file.write('The top 20 word positively correlated with rating: \n')
        for l in pc_word:
            file.write(str(l) + '\n')
        file.write('The top 20 word negatively correlated  with rating: \n')
        for l in nc_word:
            file.write(str(l) + '\n')
        file.write('The top 20 words positively related to rating, controlling for verified: \n')
        for l in controlled_pc_word:
            file.write(str(l) + '\n')
        file.write('The top 20 words negatively related to rating, controlling for verified: \n')
        for l in controlled_nc_word:
            file.write(str(l) + '\n')
    

if __name__ == "__main__":
   # Configure Spark
   conf = SparkConf().setAppName(APP_NAME).setMaster("local")
   sc = SparkContext(conf=conf)
   inputpath = sys.argv[1]
   # Execute Main functionality
   main(sc, inputpath)