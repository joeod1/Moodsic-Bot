import discord
from discord.ext import commands # for the Discord bot
import spotipy # reading oauth & Spotify playlists
import os # reading env
import json
import dotenv
import requests # for huggingface
import math # for sqrt, square
import pickle
import time
import numpy
dotenv.load_dotenv()

# Load the audio feature values for each emotion
#   Potentially define each value's weight at some point?
#   Different emotions shouldn't always imply specific audio features
emf = {}
with open("emoFeatures.json", "rb") as emoFF:
    emf = json.load(emoFF)

# Load a hashmap of songs used. Goes by track IDs
# usedSongs.pkl will be created later if it doesn't already exist
usedSongs = {}
if os.path.isfile('usedSongs.pkl'):
    with open('usedSongs.pkl', 'rb') as inp:
        usedSongs = pickle.load(inp)

# Emotions in use right now (max 10 with the current model/api)
# Any emotion can be used as long as its features are given in emoFeatures.json
emotions = ["fear", "anger", "anticipation", "trust", "surprise", "positive", "negative", "sadness", "disgust", "joy"]

# Multiplier for the weight of emotions. 0.5 is half
emotionWeight = {
    #"positivity": 0.5,
    #"negativity": 0.5
}

# Browser prompt user for login if access not already granted
# Response URL must be copied into the cli
def spotifyUserPrompt(username):
    spotipy.util.prompt_for_user_token(username, 
                                       scope=('playlist-read-private', 'playlist-read-collaborative', 'user-read-recently-played', 'user-library-read'),
                                       client_id=os.getenv('SPOTIPY_CLIENT_ID'),
                                       client_secret=os.getenv('SPOTIPY_CLIENT_SECRET'),
                                       redirect_uri=os.getenv('SPOTIPY_CLIENT_URI'))

# Returns a spotipy.Spotify instance given SPOTIPY_USER, SPOTIPY_CLIENT_ID, 
# SPOTIPY_CLIENT_SECRET, and SPOTIPY_CLIENT_URI defined in the environment
def startSpotify():
    for user in json.loads(os.getenv('SPOTIPY_USER')): # remove? sounds like there can only be one active user at a time
        spotifyUserPrompt(user)
    return spotipy.Spotify(client_credentials_manager=spotipy.oauth2.SpotifyClientCredentials())

# Returns a full list of track data from the specified playlist (should be in id format)
def getPLTracks(playlist, spotify):
    results = spotify.playlist_tracks(playlist, fields='items,uri,name,id,total,next', market='US', additional_types=('track', 'episode'))
    
    tracks = results['items']
    print(len(tracks))
    while results['next']:
        time.sleep(1.5)
        results = spotify.next(results)
        tracks.extend(results['items'])
        print(len(tracks))
    return tracks

# Returns a full list of track features based on a list of track data
def getPLFeatures(playlist, spotify):
    results = []
    for x in range(0, len(playlist) // 100 + 1):
        time.sleep(1.5)
        ids = []
        for track in playlist[x * 100 : (x + 1) * 100]:
            if (track != None and track['track'] != None and track['track']['id'] != None):
                ids.append(track['track']['id'])
        results.extend(spotify.audio_features(tracks = ids))
        print(len(results))
    return results

# Returns track features, but more importantly caches them so that we don't need to ask Spotify again
def cachePLFeatures(playlist, spotify):
    uid = playlist.split(":")[2]

    # Load features if they exist
    if os.path.isfile(uid + "fea.pkl"):
        print("Loading playlist features")
        with open(uid + "fea.pkl", "rb") as file:
            return pickle.load(file)

    # Grab features using cached playlist if it exists
    tracks = {}
    if os.path.isfile(uid + "tra.pkl"):
        print("Loading playlist tracks")
        with open(uid + "tra.pkl", "rb") as file:
            tracks = pickle.load(file)
        with open(uid + "fea.pkl", "wb") as file:
            features = getPLFeatures(tracks, spotify)
            pickle.dump(features, file, pickle.HIGHEST_PROTOCOL)
            return features
    
    # Neither file exists; create both
    tracks = {}
    with open(uid + "tra.pkl", "wb") as file:
        tracks = getPLTracks(playlist, spotify)
        pickle.dump(tracks, file, pickle.HIGHEST_PROTOCOL)
    with open(uid + "fea.pkl", "wb") as file:
        features = getPLFeatures(tracks, spotify)
        pickle.dump(features, file, pickle.HIGHEST_PROTOCOL)
        return features

# Returns a Discord client that can read messages and take commands
def startDiscordBot():
    intents = discord.Intents.default()
    intents.message_content = True
    client = commands.bot.Bot(command_prefix='$', intents=intents)
    return client

# Processes text and returns a dictionary of emotions (based on earlier emotions list)
def sentimentAnalyze(text):
    response = requests.post("https://api-inference.huggingface.co/models/facebook/bart-large-mnli", 
                                headers={"Authorization": os.getenv('HF_TOKEN')}, 
                                json={"inputs": text,
                                    "parameters": {"candidate_labels": emotions}})
    output = response.json()
    print(output)
    return dict(zip(output['labels'], output['scores']))

# Converts sentiment into audio features using values from emoFeatures.json
def sentimentToAudioFeatures(sentiment):
    r2 = {
            'energy': 0,     'acousticness': 0,
            'valence': 0,       'mode': 0,
            'loudness': 0,   'instrumentalness': 0,
            'liveness': 0,    'danceability': 0
        }
    numWeights = 0
    for emo, vals in sentiment.items():
        if emo in emotionWeight:
            vals /= emotionWeight[emo]

        for fea, val in emf[emo].items():
            r2[fea] += val * vals
        numWeights += vals

    for fea, val in r2.items():
        r2[fea] /= numWeights
        r2[fea] = math.atan(5 * r2[fea] - 2.5) / 3 + 0.5 # aggressively nudge values away from 0.5

    return r2

def trackAccuracy(features, tFeatures):
    r3 = {'energy': 0, 'danceability': 0, 'acousticness': 0, 'valence': 0, 'mode': 0, 'loudness': 0, 'instrumentalness': 0, 'liveness': 0, 'instrumentalness': 0, 'liveness': 0}

    accuracy = 0
    for fea, val in features.items():
        if (fea == 'loudness'):
            #r3[fea] = (tFeatures[fea]/60 + 1 - val) ** 2
            None
        else:
            r3[fea] = (tFeatures[fea] - val) ** 2
        accuracy += r3[fea]

    return math.sqrt(accuracy)

def closestTrack(features, plFeatures):
    closest = {}
    dist = 1000
    
    # at some point the tracks need to be clustered for efficiency
    for index, track in enumerate(plFeatures):
        if track['id'] in usedSongs:
            continue
        acc = trackAccuracy(features, track)
        if (acc < dist):
            closest = track
            dist = acc
    usedSongs[closest['id']] = True
    with open("usedSongs.pkl", "wb") as output:
        pickle.dump(usedSongs, output, pickle.HIGHEST_PROTOCOL)
    return {"closest": closest, "dist": dist}

async def postTrack(text, ctx):
    sentiment = sentimentAnalyze(text)
    print(str(sentiment))
    audioFeatures = sentimentToAudioFeatures(sentiment)
    track = closestTrack(audioFeatures, playlist)
    dist = track['dist']
    await ctx.send(f"https://open.spotify.com/track/{track['closest']['id']}")
    return sentiment

spotify = startSpotify()
#tracks = getPLTracks(os.getenv('SPOTIPY_PLAYLIST'), spotify)
#print(f"Raw length is {len(tracks)}")
#playlist = getPLFeatures(tracks, spotify)
playlist = cachePLFeatures(os.getenv('SPOTIPY_PLAYLIST'), spotify)
print(f"Playlist length is {len(playlist)}")
client = startDiscordBot()

emojis = {'fear': '😨', 'anger': '😡', 'anticipation': '🫣', 'trust': '🤝', 'surprise': '😲', 
          'positive': '😄', 'negative': '🙁', 'sadness': '😢', 'disgust': '🤢', 'joy': '😀'}

@client.command()
async def prev(ctx):
    messages = [message async for message in ctx.channel.history(limit = 2)]
    sentiment = await postTrack(messages[1].content, ctx)
    for emo, val in sentiment.items():
        if val > 0.1:
            await messages[1].add_reaction(emojis[emo])
            
    


client.run(os.getenv('DISCORD_TOKEN'))