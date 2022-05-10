##!/usr/bin/env python
## coding: utf-8

# In[1]:


from joblib import load
import librosa.display
import pandas as pd
import numpy as np
import keyboard
import sklearn
import librosa
import pyaudio
import joblib
import wave
import pyttsx3


# In[ ]:


#define sampling rate
sr4=44100
#Function for data normalisaiton
def normalize(x1, axis=0):
    return sklearn.preprocessing.minmax_scale(x1, axis=axis)

#Load Scaler and Model
sc=load(open('./scaler.sav', 'rb'))
loaded_model = joblib.load('./logreg_model.sav')


# Record in chunks of 1024 samples
CHUNK = 1024
# 16 bits per sample
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 1.5
WAVE_OUTPUT_FILENAME = "temp.wav"

def recordAudio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("Recording...")
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("Finished recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

# authenticate voice engine
def authenticate_voice():
    featuredf=pd.DataFrame(np.zeros((1,21)))
    ind=0
    data4,sample_rate4=librosa.load(WAVE_OUTPUT_FILENAME)
    featuredf.iloc[ind,1]=np.mean(data4)
    featuredf.iloc[ind,2]=np.std(data4)  
    zero_crossings4 = librosa.zero_crossings(data4, pad=False)
    spectral_centroids4 = librosa.feature.spectral_centroid(data4, sr=sr4)[0]
    spectral_rolloff4 = librosa.feature.spectral_rolloff(data4+0.01, sr=sr4)[0]
    mfccs4 = librosa.feature.mfcc(data4, sr=sr4,n_mfcc=14)
    #Spectral Centroid
    featuredf.iloc[ind,4]=np.mean(normalize(spectral_centroids4))
    #Zero Crossings
    featuredf.iloc[ind,3]=sum(zero_crossings4)
    #Spectral Rolloff
    featuredf.iloc[ind,5]=np.mean(normalize(spectral_rolloff4))
    #mfcc
    featuredf.iloc[ind,6:20]=np.mean(mfccs4,1)

    #data
    featuredf.columns=['User', 'Average', 'std', 'zcr', 'centroid', 'rollof', 'mfcc1', 'mfcc2',
           'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10',
           'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'label']
    featuredf=featuredf.drop(['User','label'],axis=1)
    test_data=np.array(featuredf)
    test_data=pd.DataFrame(test_data)
    test_data.columns=[ 'Average', 'std', 'zcr', 'centroid', 'rollof', 'mfcc1', 'mfcc2',
           'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10',
           'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14']
    transformed_data=sc.transform(test_data)

    prediction=loaded_model.predict(np.array(transformed_data).reshape(1,-1))
    
  
    return int(prediction[0])


while True:
    keyboard.wait('space')
    recordAudio()
    result = authenticate_voice()
    if(result==1.0):
    
     engine = pyttsx3.init()
     engine.say("Hello AJ, how are you doing today?")
     engine.runAndWait()
    

    else:
    
     engine = pyttsx3.init()
     engine.say("User is not authenticated?")
     engine.runAndWait()


# In[ ]:




