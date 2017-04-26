import pandas as pd
import numpy as np
import time
from scipy import sparse
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix

def generate_similarity_matrix(out_file, mode=0):
    t = time.time()
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    
    # преобразуем media_id, так чтобы оно было индексом в матрице, а не абстрактным айдишником
    from sklearn.preprocessing import LabelEncoder
    train_test = pd.concat([train, test])
    shape = (train_test.user_id.unique().shape[0], train_test.media_id.unique().shape[0])
    le = LabelEncoder().fit(train_test.media_id.unique())
    train.media_id = le.transform(train.media_id)
    test.media_id = le.transform(test.media_id)
    
    # так как трек прослушыватся мог несколько раз, то делаем голосование
    # -1 -- трек непонравился (он его чаще скипал), 1 -- понравился (чаще слушал, чем скипал)
    table = train.groupby(['user_id', 'media_id']).is_listened.mean().reset_index()
    table.is_listened = table.is_listened.round().astype(int)
    table.user_id = table.user_id.astype(int)
    table.media_id = table.media_id.astype(int)
    table.loc[table.is_listened==0, 'is_listened'] = -1
    
    del train
    del test
    del train_test
    
    
    
    X = sparse.csr_matrix(((table.is_listened), ((table.user_id), (table.media_id))), shape=shape, dtype=np.int16)
    X_abs = sparse.csr_matrix(((np.abs(table.is_listened)), ((table.user_id), (table.media_id))), shape=shape, dtype=np.int16)
    del table
    
    
    if mode == 0: # for users
        a = X.dot(X.T)         # разность между одинаковыми и разными не нулевыми оценками
        b = X_abs.dot(X_abs.T) # количество совпадений по песням
    else: # for tracks
        a = X.T.dot(X)
        b = X_abs.T.dot(X_abs)
    
    a = a+b
    b.data = np.float32(0.5)/b.data
    similarity = a.multiply(b)
    del a
    del b

    print(time.time()-t)
    similarity.eliminate_zeros()
    similarity = similarity.astype(np.float32)
    
    print(time.time()-t)
    
    sparse.save_npz(out_file, similarity)


generate_similarity_matrix('../similarity_data/users_similarity.npz', mode=0)
generate_similarity_matrix('../similarity_data/tracks_similarity.npz', mode=1)
