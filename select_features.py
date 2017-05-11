import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score

def rf_best_features(X_train, X_test, features, target, KList):
    rf_clf = RandomForestClassifier(n_estimators=300, n_jobs=-1)
    rf_clf.fit(X_train[features], X_train[target])
    rf_top = np.argsort(rf_clf.feature_importances_)[::-1]
    top_features = features[rf_top]
    auc = []
    for K in KList:
        predictors = top_features[:K]
        xgb_clf = xgb.XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=5)
        xgb_clf.fit(X_train[predictors], X_train[target])
        proba_test = xgb_clf.predict_proba(X_test[predictors])[:, 1]
        auc.append(roc_auc_score(X_test[target], proba_test))
        print('AUC(Test):{}'.format(auc[-1]))
        print ('X_train:{}'.format(X_train[predictors].shape))
    return features[rf_top[:KList[np.argmax(auc)]]], np.max(auc)

def greedy_best_features(X_train, features, cat_features, target, KList):
    throw_features = []
    best_auc = 0.0
    cat_features = cat_features.copy()
    while True:
        predictors = np.setdiff1d(features, throw_features)
        auc = []
        for f in cat_features:
            new_predictors = np.setdiff1d(predictors, [col_name for col_name in predictors if f in col_name])
            new_predictors = np.hstack((new_predictors, [f]))
            xgb_clf = xgb.XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=5)
            xgb_clf.fit(X_train[new_predictors], X_train[target])
            proba_train = xgb_clf.predict_proba(X_train[new_predictors])[:, 1]
            auc.append(roc_auc_score(X_train[target], proba_train))
        throw_feature = cat_features[np.argmax(auc)]
        throw_features.extend([col_name for col_name in predictors if throw_feature in col_name])
        best_auc = np.max(auc)
        print ('Throw Features (new5):{}'.format(throw_features[::-1][:5]))
        print ('AUC(Train):{}'.format(best_auc))
        cat_features = np.setdiff1d(cat_features, throw_feature)
        if len(cat_features) == 0:
            break
    return np.setdiff1d(features, throw_features), best_auc

def target_corr_best_features(X_tain, X_test, features, target, KList):
    auc = []
    skb = SelectKBest(f_classif, k='all')
    skb.fit(X_train[features], X_train[target])
    target_corr_top = np.argsort(skb.scores_)[::-1]
    for K in KList:
        predictors = features[target_corr_top][:K]
        xgb_clf = xgb.XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=5)
        xgb_clf.fit(X_train[predictors], X_train[target])
        proba_test = xgb_clf.predict_proba(X_test[predictors])[:, 1]
        auc.append(roc_auc_score(X_test[target], proba_test))
        print('AUC(Test):{}'.format(auc[-1]))
        print ('X_train:{}'.format(X_train[predictors].shape))
    return features[target_corr_top[:KList[np.argmax(auc)]]], np.max(auc)

def select_features(X_train, X_test, features, cat_features, target, KList):
    methods = [rf_best_features, greedy_best_features, target_corr_best_features]
    features_all = []
    auc_all = []
    for method in methods:
        print('{} start processing'.format(method.__name__ ))
        if method.__name__ == 'greedy_best_features':
            features, auc = method(X_train, features, cat_features, target, KList)
        else:
            features, auc = method(X_train, X_test, features, target, KList)
        print ('--------------------------------------------')
        np.save('{}.npy'.format(method.__name__), features)
        features_all.append(features)
        auc_all.append(auc)
    return features_all[np.argmax(auc_all)]

validation = pd.read_csv('train_mean_target.csv', index_col=0)
test = pd.read_csv('test_mean_target.csv', index_col=0)
X_train = validation[(validation.track_num > 0)].copy()
X_test  = validation[validation.track_num == 0].copy()
target = 'is_listened'
predictors = validation.columns.difference([target, 'name', 'track_num', 'title', 'listen_type', 'rating', 'sample_id'])
cat_features = [col_name for col_name in validation.columns if col_name[-3:] == 'cat'] + \
                ['user_id', 'genre_id', 'media_id', 'album_id', 'context_type', 'platform_name',
                 'platform_family', 'user_gender', 'artist_id', 'radio', 'album_genre_id', 
                 'disk_number', 'explicit_lyrics', 'track_position', 'time_of_day']    
KList = (len(predictors) * np.array([0.25, 0.4, 0.8, 1.0])).astype(int)
best_features = select_features(X_train, X_test, predictors, cat_features, target, KList)
np.save('best_features.npy', best_features)