import time
import numpy as np
from pyod.models.ecod import ECOD
from deepod.models.tabular import RCA
from pyod.models.lunar import LUNAR
from deepod.models.tabular import ICL
from deepod.models.tabular import DeepIsolationForest
from deepod.models.tabular import NeuTraL
from deepod.models.tabular import SLAD
from Compare.DTPM import DTECategorical
from Compare.SDADT.main import launch_SDAD
from Tools import utils

def compare(algo, dataname, train_x, train_y, test_x, test_y, seed, neighbor):

    maxauc = 0.0
    maxpr = 0.0
    T = 500
    lamda = 300
    random_state = seed
    maxsample = 256
    if train_x.shape[0] < maxsample:
        maxsample = train_x.shape[0]

    # k-neibors
    sum = test_x.shape[0]
    lis = [neighbor]

    feature = train_x.shape[1]

    if algo == 'ECOD':
        classifiers = {
            'ECOD': ECOD(),
        }
    elif algo == 'NeuTraL':
        classifiers = {
            'NeuTraL': NeuTraL(random_state=random_state),
        }
    elif algo == 'RCA':
        classifiers = {
            'RCA__': RCA(random_state=random_state),
        }
    elif algo == 'LUNAR':
        classifiers = {
            'LUNAR': LUNAR(),
        }
    elif algo == 'ICL':
        classifiers = {
            'ICL__': ICL(random_state=random_state),
        }
    elif algo == 'DIF':
        classifiers = {
            'DIF__': DeepIsolationForest(max_samples=maxsample),
        }
    elif algo == 'SLAD':
        classifiers = {
            'SLAD_': SLAD(random_state=random_state),
        }
    elif algo == 'DTPM':
        classifiers = {
            'DTPM': DTECategorical(seed=random_state),
        }
    elif algo == 'SDAD':
        classifiers = {
            'SDAD': ECOD(),
        }

    name = []
    maxauc_list =  []
    maxpr_list =  []
    timetaken = []

    starttime = time.time()
    for clf_name, clf in classifiers.items():

        if clf_name == 'LUNAR':
            for model_type in ["WEIGHT", "SCORE"]:
                starttime = time.time()
                clf = LUNAR(n_neighbours=neighbor, model_type= model_type)
                clf.fit(train_x)
                test_scores = clf.decision_function(test_x)
                auc, pr = utils.Metrics(test_y, test_scores)
                tim = time.time() - starttime
                if auc > maxauc:
                    maxauc = auc
                    maxpr = pr
                    tims = tim

        if clf_name == 'SDAD':
            maxauc, maxpr = launch_SDAD(dataname, train_x, train_y, test_x, test_y, T, lamda)
            tims = time.time() - starttime

        else:
            clf.fit(train_x)
            test_scores = clf.decision_function(test_x)
            auc, pr = utils.Metrics(test_y, test_scores)
            maxauc = auc
            maxpr = pr
            tims = time.time() - starttime

        # results
        name = clf_name
        maxauc_list = maxauc
        maxpr_list = maxpr
        timetaken = tims


    return name, maxauc_list, maxpr_list, timetaken
