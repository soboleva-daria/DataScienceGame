#!/usr/bin/env python3

# import pandas as pd
# import numpy as np
import aiohttp
import asyncio
import csv
import pandas as pd
import os.path
from six.moves import cPickle as pickle


NUM_COROUTINES = 50
OUTPUT_FILE = '../input/songs_info.csv'


async def fetch_tracks(track_ids, writer):
    for track_id in track_ids:
        track_url = 'https://api.deezer.com/track/{}'.format(track_id)
        media_info = {'media_id': track_id}
        try:
            async with aiohttp.request('GET', track_url) as track_resp:
                resp = await track_resp.json()
                if 'error' in  resp:
                    print(track_id, resp['error']['code'], 'FAIL')
                    if resp['error']['code'] == 800:
                        writer.writerow(media_info)
                else:
                    title = media_info['title'] = resp['title']
                    media_info['disk_number'] = resp['disk_number']
                    media_info['bpm'] = resp['bpm']
                    media_info['explicit_lyrics'] = resp['explicit_lyrics']
                    media_info['gain'] = resp['gain']
                    media_info['track_position'] = resp['track_position']
                    media_info['rank'] = resp['rank']
                    media_info['artist_id'] = resp['artist']['id']
                    media_info['album_id'] = resp['album']['id']
                    media_info['release_date'] = resp['release_date']
                    media_info['duration'] = resp['duration']
                    writer.writerow(media_info)
                    print(track_id, track_resp.status, title)
        except:
            pass
        await asyncio.sleep(5 + 1e-3)


if __name__ == '__main__':
    with open('../input/songs_id.pkl', 'rb') as fl:
        songs_id = pickle.load(fl)
    fieldnames = [
        'media_id',
        'title',
        'disk_number',
        'bpm',
        'explicit_lyrics',
        'gain',
        'track_position',
        'rank',
        'artist_id',
        'album_id',
        'release_date',
        'duration',
    ]
    found_songs = set()
    if os.path.exists(OUTPUT_FILE):
        df = pd.read_csv(OUTPUT_FILE, usecols=['media_id'], squeeze=True)
        found_songs = set(df)
    print('Already downloaded {} songs'.format(len(found_songs)))
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as fl:
        writer = csv.DictWriter(fl, fieldnames,
                                quoting=csv.QUOTE_NONNUMERIC, delimiter=',')
        if not found_songs:
            writer.writeheader()
        songs_id = [x for x in songs_id if x not in found_songs]
        loop = asyncio.get_event_loop()
        tasks = [
            fetch_tracks(songs_id[i::NUM_COROUTINES], writer)
            for i in range(NUM_COROUTINES)
        ]
        tasks = asyncio.gather(*tasks)
        loop.run_until_complete(tasks)
