import csv
import numpy as np
import numpy.random as rndm
import time
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier

def load_recipes(path):
    ''' Return a list of lists of the csv file'''
    with open(path) as c:
        reader = csv.reader(c)
        return [', '.join(row) for row in reader]

def randomize_recipes(path):
    '''Randomizes a csv file of recipes, helps prevent patterns in the data order'''
    recipes = load_recipes(path)
    recipes = [recipe.split(', ') for recipe in recipes]
    rndm.shuffle(recipes)
    random_file = path[:-4] + 'random.csv'
    with open(random_file, 'w') as cr:
        writer = csv.writer(cr, delimiter=',')
        writer.writerows(recipes)

def nationality_ingredients(recipes):
    # nationalities = [recipe[0] for recipe in recipes]
    # ingredients = [recipe[1:] for recipe in recipes]
    nationalities = []
    ingredients = []
    for recipe in recipes:
        splits = recipe.split(', ', 1)
        nationalities.append(splits[0])
        ingredients.append(splits[1])
    return np.array(nationalities), np.array(ingredients)

def train_test(recipes):
    size = len(recipes)
    split = 2 * size // 3
    train = recipes[:split]
    test = recipes[split:]
    return train, test

# randomize_recipes('data/srep00196-s3.csv')

## Split the data and transform it into frequency-based counts
train_data, test_data = train_test(load_recipes('data/srep00196-s3random.csv'))

y, train_X = nationality_ingredients(train_data)

categories = list(set(y))


text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier())])

# text_clf = text_clf.fit(train_X, y)

# test data
test_y, test_X = nationality_ingredients(test_data)
# predicted = text_clf.predict(test_X)

# results = np.mean(predicted == np.array(test_y))

# For Bayesian Classifier
parameters = {'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3)}

parameters = {'tfidf__use_idf': (True, False),
              'clf__loss': ('hinge', 'log', 'modified_huber', 'squared_hinge'),
              'clf__penalty': ('l2', 'l1', 'elasticnet'),
              'clf__alpha': (1e-3, 1e-2, 1e-1, 1)}

gs_clf = GridSearchCV(text_clf, parameters, n_jobs=3)

def train_classifier():
    start_time = time.time()
    gs_clf.fit(train_X, y)
    end_time = time.time()
    print("Time taken to train data: %r min" % ((end_time - start_time) / 60))
    best_params, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, best_params[param_name]))

    predicted = gs_clf.predict(test_X)
    print(metrics.classification_report(test_y, predicted,
                                        target_names=categories))


def get_average():
    predicted = gs_clf.predict(test_X)
    return np.mean(predicted == test_y)

if __name__ == '__main__':
    train_classifier()
