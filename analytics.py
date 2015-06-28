from copy import deepcopy
from budgetparser import all_ingredients, ingredient_dict
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import OrderedDict, defaultdict

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

zero_vector = np.zeros(len(all_ingredients))

vector_dict = OrderedDict()

for ingredient in ingredient_dict.keys():
    ingredient_vector = deepcopy(zero_vector)
    for i, recipe in enumerate(all_ingredients):
        if ingredient in recipe:
            ingredient_vector[i] = 1 / ingredient_dict[ingredient]

    vector_dict[ingredient] = ingredient_vector


clusterer = KMeans(n_clusters=20, n_init=100)
clusterer.fit([val for val in vector_dict.values()])

clusters = defaultdict(list)

for k, v in vector_dict.items():
    group = clusterer.predict(v)[0]
    clusters[group].append(k)


def main():
    for group, ingredients in clusters.items():
        print("Group no. %s" % group)
        print(*list(ingredients), sep=", ")


if __name__ == '__main__':
    main()
