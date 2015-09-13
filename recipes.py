from sklearn import cross_validation
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import csv
import numpy as np
import numpy.random as rndm
import pickle
import pandas as pd
import time


try:
    with open('./classifiers.p', 'rb') as f:
        classifiers = pickle.load(f)
except IOError:
    classifiers = {}


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
train_data, test_data = cross_validation.train_test_split(
    load_recipes('data/srep00196-s3random.csv'),
    test_size=0.3,
    random_state=42)

y, train_X = nationality_ingredients(train_data)

categories = sorted(list(set(y)))


def create_text_pipeline(classifier):
    return Pipeline(
        [('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
         ('clf', classifier)])

# test data
test_y, test_X = nationality_ingredients(test_data)


# predicted = text_clf.predict(test_X)
def training_data_vectorized():
    vect = CountVectorizer()
    tfidf = TfidfTransformer()
    vect_X = vect.fit_transform(train_X)
    return tfidf.fit_transform(vect_X), y


def test_data_vectorized():
    vect = CountVectorizer()
    tfidf = TfidfTransformer()
    vect_X = vect.fit_transform(test_X)
    return tfidf.fit_transform(vect_X), test_y


def create_grid_clf(text_clf, clf_params={}, num_jobs=2):
    parameters = {}
    # parameters = {'tfidf__use_idf': (True, False)}
    parameters.update(clf_params)
    gs_clf = GridSearchCV(text_clf, parameters,
                          n_jobs=num_jobs,
                          error_score=-1)
    return gs_clf


def train_classifier(gs_clf, name='', verbose=True, x=train_X, y=y):
    if verbose:
        print('Results for: %s\n' % name)
        start_time = time.time()
    gs_clf.fit(x, y)
    end_time = time.time()
    if verbose:
        print("Time taken to train data: %r min" %
              ((end_time - start_time) / 60))
    best_params, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
    if verbose:
        for param_name, value in sorted(best_params.items()):
            print("%s: %r" % (param_name, best_params[param_name]))

    stats, percentage, matrix = get_stats(gs_clf)
    if verbose:
        print(stats)
        print('Percent Correct: {}'.format(percentage))
        print('Confusion matrix:\n {}'.format(matrix))
        print()
    return gs_clf.best_estimator_


def get_stats(clf, test_X=test_X, test_y=test_y):
    predicted = clf.predict(test_X)
    stats = metrics.classification_report(test_y, predicted,
                                          target_names=categories)
    percentage = metrics.accuracy_score(test_y, predicted)
    y_true = pd.Series(test_y)
    y_pred = pd.Series(predicted)
    confusion = pd.crosstab(y_true, y_pred,
                            rownames=['True'],
                            colnames=['Predicted'],
                            margins=True)
    return stats, percentage, confusion


def use_clf(clf, name, params={}, verbose=True, x=train_X, y=y):
    try:
        best_clf = classifiers[name]
    except KeyError:
        text_clf = create_text_pipeline(clf)
        gs_clf = create_grid_clf(text_clf, params, num_jobs=-1)
        best_clf = train_classifier(gs_clf, name, verbose, x, y)
        classifiers[name] = best_clf
    return best_clf

# classifiers = {}  # Resetting for now

if __name__ == '__main__':
    use_clf(LogisticRegression(fit_intercept=True,
                               class_weight='auto'),
            name='Logistic1',
            params={
                'clf__C': (10 ** np.arange(0, 3)),
                'clf__solver': ('newton-cg', 'lbfgs', 'liblinear'),
                'clf__penalty': ('l1', 'l2')
            })
    use_clf(MultinomialNB(fit_prior=True),
            name='Bayes',
            params={
                'clf__alpha': np.arange(0, 1.0, 0.1)
            })
    use_clf(LogisticRegression(fit_intercept=True, ),
            name='Logistic2',
            params={
                'clf__C': (10 ** np.arange(0, 3)),
                'clf__solver': ('newton-cg', 'lbfgs', 'liblinear'),
                'clf__penalty': ('l1', 'l2')
            })
    use_clf(SVC(class_weight='auto',
                probability=True),
            name='SVM',
            params={
                'clf__C': 10 ** np.arange(-3, 3),
                'clf__gamma': np.arange(0.0, 0.1, 0.05),
            })

    with open('./classifiers.p', 'wb') as f:
        pickle.dump(classifiers, f)
