import os, sys
import youtube_dl

ydl_opts = {
    'format': 'bestaudio/best',
    'outtmpl': './Youtube-DL/Samples/%(channel)s/%(id)s.%(ext)s',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
    }],
    'cachedir': False, # Important to avoid cache for downloaded files. It speeds up the process but tends to throw 403: Forbidden error.
    'verbose': True
    }

ydl = youtube_dl.YoutubeDL(ydl_opts)

link_file = os.path.normpath('./Youtube-DL/Japanese_Links.txt')

file1 = open(link_file, 'r')
#lines = file1.readlines()
lines = file1.read().splitlines()

with ydl:
    for line in lines:
        ydl.download([line])