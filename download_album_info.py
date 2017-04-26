import deezer
import time
import pandas as pd
import numpy as np
client = deezer.Client()
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
train_test = pd.concat([train, test])
albums = np.unique(train_test.album_id)
albums_info = []
while len(albums_info) < len(albums):
    for album in albums[len(albums_info):]:
        try:
            d = client.get_album(album).asdict()
            albums_info.append((album, d['genre_id'], d['title'], d['nb_tracks'], d['rating'], d['duration'], d['fans']))
            if len(albums_info)%10==0:
                print(len(albums_info))
        except KeyError as e:
            if d['error']['code'] == 800:
                albums_info.append((album, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
            else:
                print(d)
                break
        except:
            break
        pass
    time.sleep(1)

ai = pd.DataFrame(data=np.array(albums_info), columns=['album_id', 'album_genre_id', 'title', 'nb_tracks', 'rating', 'duration','fans'])
ai.to_csv('../api_data/albums_info.csv',index=False)
