import deezer
import pandas as pd
import numpy as np
client = deezer.Client()
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
train_test = pd.concat([train, test])
artists = np.unique(train_test.artist_id)
art_info = []
import time
while len(art_info) < len(artists):
    for artist in artists[len(art_info):]:
        try:
            d = client.get_artist(artist).asdict()
            art_info.append((artist, d['name'], d['radio'], d['nb_fan'], d['nb_album']))
            if len(art_info)%10==0:
                print(len(art_info))
        except KeyError as e:
            if d['error']['code'] == 800:
                art_info.append((artist, np.nan, np.nan, np.nan, np.nan))
            else:
                print(d)
                break
        except:
            break
    time.sleep(1)
ar = pd.DataFrame(data=np.array(art_info), columns=['artist_id', 'name', 'radio', 'nb_fan', 'nb_album'])
ar.to_csv('../api_data/artist_info.csv', index=False)
