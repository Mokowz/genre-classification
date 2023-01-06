from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
import bz2
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials



app = Flask(__name__)

cid = "967ea998dca34b6a897edf6ad66bdb13"
csec = "bdd75e6de00842b188eb901ab409d128"

auth_man = SpotifyClientCredentials(client_id=cid, client_secret=csec)
sp = spotipy.Spotify(auth_manager=auth_man)

songs = pd.read_csv("songs.csv")

ifile = bz2.BZ2File("Classifier",'rb')
model = pickle.load(ifile)
ifile.close()


@app.route("/") 
def hello():
    return render_template("index.html")



@app.route('/predict', methods = ["POST"])
# def submit():
#     if request.method == "POST":
#         artist = request.form["artist"]

#     return render_template("sub.html")
def predict():
    ex = getFeatures()
    y_pred = model.predict(ex)
    y_pred = int(y_pred)
    y_pred

    for index, row in songs.iterrows():
        if (row["track_genre_code"] == y_pred):
            genre = (row["track_genre"])
            break
    
    return render_template("index.html", prediction_text="The song's genre is: {}".format(genre))



def getTrackId():
    artist = request.form["artist"]
    track = request.form["track"]
    
    track_id = sp.search(q='artist:' + artist + ' track:' + track, type='track')
    if (track_id['tracks']['items'][0]['id']):
        trackId = track_id['tracks']['items'][0]['id']
    else:
        print("This song has no ID. Try another one.")
        return
    
    return trackId


def getFeatures():
    features = sp.audio_features(getTrackId())
    
    song = []

    # song.append(features[0]["duration_ms"])
    song.append(features[0]["danceability"])
    song.append(features[0]["energy"])
    song.append(features[0]["key"])
    song.append(features[0]["loudness"])
    song.append(features[0]["mode"])
    song.append(features[0]["speechiness"])
    song.append(features[0]["acousticness"])
    song.append(features[0]["instrumentalness"])
    song.append(features[0]["liveness"])
    song.append(features[0]["valence"])
    song.append(features[0]["tempo"])
    # song.append(features[0]["time_signature"])
    

    song = np.array(song).reshape(1, -1)
    
    return song

# model = pickle.load(open("Music Genre Classification.pkl", "rb"))



           


if __name__ == "__main__":
    app.run(debug=True)