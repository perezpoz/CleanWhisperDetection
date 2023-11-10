import os, sys, glob
from pyannote.audio import Pipeline
import pandas as pd


data_path = os.path.normpath('./Dataset/')

audio_files = glob.glob(os.path.join(data_path,'*Test.wav'))

pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection")

fs = 100

for file in audio_files:
    output = pipeline(file)

    start_list = []
    end_list = []

    timestamps = output.get_timeline()
    for stamp in timestamps:
        start_list.append(stamp.start)
        end_list.append(stamp.end)
    
    time_dict = {'start': start_list, 'end': end_list}

    time_df = pd.DataFrame.from_dict(time_dict)

    output_name = file[:-4] + '_Pyannote.csv'

    time_df.to_csv(output_name)