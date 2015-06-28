import collections
import os
import os.path
import re
from os.path import curdir, isfile

import nltk
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import json
import subprocess


# Note, remove "meal" and "how to" urls


main_url = "http://www.budgetbytes.com/category/recipes/vegetarian/"

veggie_soup = BeautifulSoup(requests.get(main_url).text)

example_page = requests.get("http://www.budgetbytes.com/2015/05/chili-lime-cantaloupe/").text

example_soup = BeautifulSoup(example_page)

# Also using stop words from nltk
word_filter_list = set(['tsp', 'tbsp', 'about', 'cup', 'cups', 'half', 'cooked', 'frozen', 'to', 'taste', 'or', 'small', 'medium', 'large', 'a', 'an', 'pinch', 'dash', 'few', 'handful', 'any', 'as', 'needed', 'bag', 'bags', 'bunch', 'each', 'box', 'boxes', 'leaves', 'handfull', 'hand', 'full', 'inch',
'med', 'lg', 'generous', 'crumpled', 'freshly', 'size', 'shredded', 'package'])

stop_words = set(stopwords.words("english"))
wnl = WordNetLemmatizer()

def get_all_links():
    links = []
    for i in range(1, 5):
        page = main_url + 'page/%s/' % i
        soup = BeautifulSoup(requests.get(page).text)
        links += [a.get('href') for a in soup.findAll('a', attrs={'rel': 'bookmark'})]

    return links



def get_good_links():
    return [link for link in get_all_links() if "how-to" not in link and "meal" not in link
            and 'meze-lunchbox' not in link][:-20]

def get_ingredients_list(url):
    # time.sleep(1)
    soup = BeautifulSoup(requests.get(url).text)
    try:
        lis = soup.find('div', attrs={'class': 'ERSIngredients'}).findAll('li')
        split_ingredients = [li.contents[0].split() for li in lis]
        ingredients = [' '.join([wnl.lemmatize(i) for i in map(lambda x: x.lower(),
                                                               ingredient)
                                 if i not in word_filter_list
                                 and i not in stop_words
                                 and re.match(r'^[a-z]+$', i)])
                       for ingredient in split_ingredients]
        return ingredients
    except Exception as e:
        print(e, url)


if isfile(curdir + '/budget_ingredients'):
    with open(os.path.curdir + '/budget_ingredients') as f:
        all_ingredients = json.load(f)

else:
    all_ingredients = [get_ingredients_list(url) for url in get_good_links()]
    with open(curdir + '/budget_ingredients', 'w') as f:
        json.dump(all_ingredients, f, indent=4, sort_keys=True)


if isfile(curdir + '/budget_dict'):
    with open(curdir + '/budget_dict') as f:
        ingredient_dict = json.load(f)

else:
    ingredient_dict = collections.defaultdict(int)
    for recipe in all_ingredients:
        for ingredient in recipe:
            ingredient_dict[ingredient] += 1
    del(ingredient_dict[''])

    with open(curdir + '/budget_dict', 'w') as f:
        json.dump(ingredient_dict, f, indent=4, sort_keys=True)


def remove_files():
    subprocess.call(["rm", "budget_ingredients", "budget_dict"])


def most_common_ingredients():
    ingredients = list(ingredient_dict.keys())
    ingredients.sort(key=lambda x: ingredient_dict[x], reverse=True)
    return ingredients

def main():
    mci = most_common_ingredients()
    for key in range(100):
        print(mci[key], ingredient_dict[mci[key]])

if __name__ == '__main__':
    main()
