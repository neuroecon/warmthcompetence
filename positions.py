import gensim
import numpy as np
import pickle
import os
import csv
import json
import time
from statistics import stdev
from pprint import pprint


with open('input/countries.txt', 'rt') as f:
    country_names = list(map(
        lambda s: s.replace(' ', '_'), f.read().splitlines())
    )

with open('input/labels.json') as f:
    axis_labels = json.load(f)

def load(name):
    def wrapper(func):
        def inner(*args, **kwargs):
            start_time = time.time()
            print('Loading {}...'.format(name))
            result = func(*args, **kwargs)
            print('Finished loading {} in {}s.'.format(
                name,
                round(time.time() - start_time, 3)
            ))
            return result
        return inner
    return wrapper

@load("Google News")
def load_google():
    return gensim.models.KeyedVectors.load_word2vec_format("vectors/googlenews.bin", binary=True)

@load("Fox News")
def load_fox():
    return gensim.models.KeyedVectors.load('vectors/foxnews/model.keyedvectors')
    
@load("Fox News aligned on Google News")
def load_fox_google():
    return gensim.models.KeyedVectors.load('vectors/fox_on_google/fox_on_google.w2v')

def get_target_vecs(model, targets=country_names):
    """
    Takes a model and a list of vectors. Prints out
    any target that isn't in the model, and then returns
    a dictionary of the desired targets
    """
    targets = set(targets)
    included = {t for t in targets if t in model}
    excluded = targets - included
    print("Couldn't find the following targets: " + ", ".join(excluded))
    return {t: model[t] for t in included}

def normalize(vec):
    return vec / np.linalg.norm(vec)

def get_average(model, labels):
    labels = list({l for l in labels if l in model})
    assert len(labels), "None of the provided labels are in the model"
    total = model[labels[0]]
    for label in labels[1:]:
        total = total + model[label]
    return normalize(total)

def get_axis(model, positive_labels, negative_labels):
    axis = get_average(model, positive_labels) - get_average(model, negative_labels)
    return normalize(axis)

def get_wc_axes(model):
    warm_axis = get_axis(model, axis_labels['warm'], axis_labels['cold'])
    comp_axis = get_axis(model, axis_labels['competent'], axis_labels['incompetent'])
    return warm_axis, comp_axis

def project(model, targets, axis):
    scalars = {}
    axis = normalize(axis)
    for t in targets:
        if t not in model:
            print('{} not found in model.'.format(t))
            continue
        tvec = model[t]
        scalars[t] = normalize(tvec).dot(axis)
    return scalars

def mean(lst):
    return sum(lst) / (len(lst) or 1)

def normalize_dict(scores):
    m = mean(list(map(float, scores.values())))
    d = stdev(list(map(float, scores.values())))
    return {target: (score - 0)/1 for target, score in scores.items()}

def create_cache():
    fox = load_fox()
    with open('cache/fox_axes.pickle', 'wb') as f:
        pickle.dump(list(get_wc_axes(fox)), f)
    with open('cache/fox_targets.pickle', 'wb') as f:
        pickle.dump(get_target_vecs(fox), f)
    del fox
    google = load_google()
    with open('cache/google_axes.pickle', 'wb') as f:
        pickle.dump(list(get_wc_axes(google)), f)
    with open('cache/google_targets.pickle', 'wb') as f:
        pickle.dump(get_target_vecs(google), f)
    del google
    fox_on_google = load_fox_google()
    with open('cache/fox_on_google_targets.pickle', 'wb') as f:
        pickle.dump(get_target_vecs(fox_on_google), f)
    del fox_on_google

# create_cache()
def load_cache():
    with open('cache/fox_axes.pickle', 'rb') as f:
        fox_axes = pickle.load(f)
    with open('cache/fox_targets.pickle', 'rb') as f:
        fox_targets = pickle.load(f)
    with open('cache/google_axes.pickle', 'rb') as f:
        google_axes = pickle.load(f)
    with open('cache/google_targets.pickle', 'rb') as f:
        google_targets = pickle.load(f)
    with open('cache/fox_on_google_targets.pickle', 'rb') as f:
        fox_on_google_targets = pickle.load(f)
    return fox_axes, fox_targets, google_axes, google_targets, fox_on_google_targets

fox_axes, fox_targets, google_axes, google_targets, fox_on_google_targets = load_cache()

def output_fox():
    warm, comp = fox_axes
    warm_scores = normalize_dict(project(fox_targets, country_names, warm))
    comp_scores = normalize_dict(project(fox_targets, country_names, comp))
    rows = []
    for n in warm_scores:
        row = {'name': n, 'warm': warm_scores[n], 'comp': comp_scores[n]}
        rows.append(row)
    rows.sort(key=lambda r: r['name'])

    with open('output/fox_raw.csv', 'wt') as f:
        writer = csv.DictWriter(f, ['name', 'warm', 'comp'])
        writer.writeheader()
        writer.writerows(rows)

def output_google():
    warm, comp = google_axes
    warm_scores = normalize_dict(project(google_targets, country_names, warm))
    comp_scores = normalize_dict(project(google_targets, country_names, comp))
    rows = []
    for n in warm_scores:
        row = {'name': n, 'warm': warm_scores[n], 'comp': comp_scores[n]}
        rows.append(row)
    rows.sort(key=lambda r: r['name'])

    with open('output/google_raw.csv', 'wt') as f:
        writer = csv.DictWriter(f, ['name', 'warm', 'comp'])
        writer.writeheader()
        writer.writerows(rows)

def output_fox_on_google():
    warm, comp = google_axes
    warm_scores = normalize_dict(project(fox_on_google_targets, country_names, warm))
    comp_scores = normalize_dict(project(fox_on_google_targets, country_names, comp))
    rows = []
    for n in warm_scores:
        row = {'name': n, 'warm': warm_scores[n], 'comp': comp_scores[n]}
        rows.append(row)
    rows.sort(key=lambda r: r['name'])

    with open('output/fox_on_google_raw.csv', 'wt') as f:
        writer = csv.DictWriter(f, ['name', 'warm', 'comp'])
        writer.writeheader()
        writer.writerows(rows)

def output_fox_google_diff():
    warm, comp = google_axes
    warm_fox = project(fox_on_google_targets, country_names, warm)
    comp_fox = project(fox_on_google_targets, country_names, comp)
    warm_google = project(google_targets, country_names, warm)
    comp_google = project(google_targets, country_names, comp)
    overlap = list(set(comp_fox) & set(comp_google))
    warm_scores = normalize_dict({
        t: warm_fox[t] - warm_google[t] for t in overlap
    })
    comp_scores = normalize_dict({
        t: comp_fox[t] - comp_google[t] for t in overlap
    })

    rows = []
    for n in warm_scores:
        row = {'name': n, 'warm': warm_scores[n], 'comp': comp_scores[n]}
        rows.append(row)
    rows.sort(key=lambda r: r['name'])

    with open('output/fox_google_axes_diff_raw.csv', 'wt') as f:
        writer = csv.DictWriter(f, ['name', 'warm', 'comp'])
        writer.writeheader()
        writer.writerows(rows)


output_fox_on_google()
output_google()
output_fox()
output_fox_google_diff()
