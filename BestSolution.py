import pandas as pd
import numpy as np
from  datetime import datetime
import time
from itertools import combinations
from sklearn.metrics import roc_auc_score
import xgboost as xgb

def statistics_real_feature(statistic_set, train, features, feature, list_func):
    train = train.copy()
    for func in list_func:
        print (func)
        for f in features:
            f_ = list(f)
            c = '&'.join(f_) + '-' + feature + '_' + func.__name__
            try:
                ix = [tuple(i for i in j) for j in train[f_].values.squeeze()]
            except:
                ix = train[f_].values.squeeze()
            train[c] = statistic_set.groupby(f_)[feature].agg([func]).loc[ix].values
    return train

def smooth_mean_target(statistic_set, train, features, target, alpha=10):
    train = train.copy()
    global_mean_y = statistic_set[target].mean()
    
    for f in features:
        f_ = list(f)
        c = '&'.join(f_) + '-' + target
        grp = statistic_set.groupby(f_)
        try:
            ix = [tuple(i for i in j) for j in train[f_].values.squeeze()]
        except:
            ix = train[f_].values.squeeze()
            
        print (f_)
        train[c + '_size'] = grp.size().loc[ix].values
        train[c + '_mean'] = grp[target].mean().loc[ix].values
        K = train[c + '_size'].values
        mean_y = train[c + '_mean'].values
        train[c + 'smooth_mean'] = (mean_y * K + global_mean_y * alpha) / (K + alpha)    
    return train


def mean_target(statistic_set, train, features, target):
    train = train.copy()
    
    for f in features:
        f_ = list(f)
        print (f_)
        c = '&'.join(f_) + '-' + target
        grp = statistic_set.groupby(f_)
        try:
            ix = [tuple(i for i in j) for j in train[f_].values.squeeze()]
        except:
            ix = train[f_].values.squeeze()

        train[c + '_size'] = grp.size().loc[ix].values
        train[c + '_mean'] = grp[target].mean().loc[ix].values
    return train

def predict_test(validation, test, predictors, target, num_tracks_skip):
    validation = validation[validation.track_num < 5]
    dtest_predprobs = []    
    ncols = [0.05, 0.1, 0.25, 0.4, 0.8]
    clf = xgb.XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=5)
    for ncol in ncols:
        print (ncol)
        clf.set_params(colsample_bytree=ncol)
        clf.fit(validation[predictors],
                validation[target])
        dtest_predprobs.append(clf.predict_proba(test[predictors])[:, 1])
        
    test[target] = (np.array(dtest_predprobs) * np.array([0.05, 0.1, 0.25, 0.4, 0.8]).reshape(5, 1)).sum(axis=0) 

    ans = test[['sample_id', target]].copy()
    ans.sort_values(by='sample_id', inplace=True)
    ans['sample_id'] = ans.sample_id.astype(int)
    ans.to_csv('ans_smooth_mean_target_real_features{}.csv'.format(num_tracks_skip), index=False) 

def cross_val_score(X_train, X_test, predictors, target):
    dtrain_predprobs = [] 
    dtest_predprobs = []     
    ncols = [0.05, 0.1, 0.25, 0.4, 0.8]
    clf = xgb.XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=5)
    for ncol in ncols:
        print (ncol)
        clf.set_params(colsample_bytree=ncol)
        clf.fit(X_train[predictors],
                X_train[target])
        
        dtrain_predprobs.append(clf.predict_proba(X_train[predictors])[:, 1])
        dtest_predprobs.append(clf.predict_proba(X_test[predictors])[:, 1])
        
    tmp1 = (np.array(dtrain_predprobs) * np.array([0.05, 0.1, 0.25, 0.4, 0.8]).reshape(5, 1)).sum(axis=0)
    tmp2 = (np.array(dtest_predprobs) * np.array([0.05, 0.1, 0.25, 0.4, 0.8]).reshape(5, 1)).sum(axis=0) 

    print ("AUC Score (Train): %f" % roc_auc_score(X_train[target], tmp1))
    print ("AUC Score (Test): %f" % roc_auc_score(X_test[target], tmp2)) 

def build_model(train, test, target, X_test=None, num_tracks_skip=0):
    validation = []
    num_tracks = 6 
    for i in range(num_tracks):
        print ('num_track:{}'.format(i))

        v = train[train.listen_type==1].groupby(['user_id']).last().reset_index() 
        tmp = v[['user_id', 'ts_listen']].copy()
        tmp.columns = ['user_id', 'max_time']
        train = pd.merge(train, tmp, on='user_id', how='left')
        train = train[train.ts_listen < train.max_time]
        train.drop('max_time',axis=1, inplace=True)
            
        v_new = v.copy()
        print('0')
        for f in real_features:
            v_new = statistics_real_feature(train, v_new, combinations(cat_features , 1), f, funcs)
        print('1')
        v_new = smooth_mean_target(train, v_new, combinations(cat_features, 1), target)
        print('2')
        v_new = smooth_mean_target(train, v_new, combinations(cat_features, 2), target)

        v_new['track_num'] = i
        if i == 0:
            validation = v_new.copy()
        else:
            validation = pd.concat([validation, v_new])
        print(i, validation.shape)
        
    validation.fillna(0, inplace=True)
    #validation.to_csv('train_smooth_mean_target_real_features{}.csv'.format(num_tracks_ago))
    print('train_shape:{}'.format(validation.shape))
    print('train done')

    X_train = validation[(validation.track_num > 0)].copy()
    if num_tracks_skip == 0:
        X_test  = validation[validation.track_num == 0].copy()
    predictors = X_train.columns.difference(['name', 'track_num', 'is_listened', 'title'])

    cross_val_score(X_train, X_test, predictors, target)
    predict_test(validation, test, predictors, target, num_tracks_skip)
    return train, X_test

if __name__ == '__main__':
    train = pd.read_csv('koly_train_cat.csv')
    test = pd.read_csv('kolya_test_cat.csv')

    cat_features = [col_name for col_name in train.columns if col_name[-3:] == 'cat'] + \
                    ['user_id', 'genre_id', 'media_id', 'album_id', 'context_type', 'platform_name',
                     'platform_family', 'listen_type', 'user_gender', 'artist_id', 'radio', 'album_genre_id', 
                     'disk_number', 'explicit_lyrics', 'track_position', 'time_of_day']

    real_features = ['ts_listen', 'media_duration', 'user_age', 'rank', 'time_diff', 'fans', 'bpm', 'gain', 'nb_fan', 'duration']
    funcs = [np.mean, np.median, np.max, np.min]
    target = 'is_listened'


    print('0')
    for f in real_features:
        test = statistics_real_feature(train, test, combinations(cat_features , 1), f, funcs)
    print ('1')
    test = smooth_mean_target(train, test, combinations(cat_features , 1), target)
    print ('2')
    test = smooth_mean_target(train, test, combinations(cat_features , 2), target)

    test.to_csv('test_smooth_mean_target_real_features.csv')
    print('test_shape:{}'.format(test.shape))
    print('test done')

    train, X_test = build_model(train, test, target)
    train, X_test = build_model(train, test, target, X_test, 1)
    train, X_test = build_model(train, test, target, X_test, 2)

    ans = pd.read_csv('ans_smooth_mean_target_real_features0.csv', index_col=0, squeeze=True)
    ans1 = pd.read_csv('ans_smooth_mean_target_real_features1.csv', index_col=0, squeeze=True)
    ans2 = pd.read_csv('ans_smooth_mean_target_real_features2.csv', index_col=0, squeeze=True)

    ans /= ans.max()
    ans1 /= ans1.max()
    ans2 /= ans2.max()

    mean_ans = (ans + ans1 + ans2) / 3
    mean_ans = mean_ans.reset_index()
    mean_ans.sort_values(by='sample_id', inplace=True)
    mean_ans['sample_id'] = mean_ans.sample_id.astype(int)
    mean_ans.to_csv('mean_ans_smooth_mean_target_real_features.csv', index=False)