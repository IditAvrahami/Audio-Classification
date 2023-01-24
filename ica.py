"""
Names: Valeri Materman 321133324
       Idit Avrahami 207565748
"""



import wave
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plot
from scipy.io import wavfile 
from sklearn.decomposition import FastICA
from sklearn import preprocessing
import librosa
from scipy import signal
from sklearn import datasets, svm, metrics,preprocessing,linear_model,model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, metrics
from sklearn import linear_model
from sklearn. model_selection import cross_val_predict
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle

from scipy.fft import fftshift

#Part A question 4:

file_list = ["source1.wav","source2.wav","source3.wav","source4.wav","source5.wav","source6.wav"]
file_list_final = ["final_source1.wav","final_source2.wav","final_source3.wav","final_source4.wav","final_source5.wav","final_source6.wav"]
tags =[   "noise",  "person","person" ,  "person",   "noise",   "person"]

def plot_sound(data,text):
    """
    Plots histogram and returns appended data

    Parameters
    ----------
    data : the wave.open objects
    Returns
    -------
    X : array that contains the wav data as arrays of float
        DESCRIPTION.

    """
    i=0
    X = []
    for data_of_wave in data:
       plot.subplot(211)
       plot.title('Spectrogram of the {} sound {}'.format(i,text))
       X.append(data_of_wave[1])
       plot.plot(data_of_wave[1])
       plot.xlabel('Sample') 
       plot.ylabel('Amplitude')
       plot.subplot(212)
       plot.specgram(data_of_wave[1],Fs=data_of_wave[0])
       plot.xlabel('Time')
       plot.ylabel('Frequency')
       plot.show()
       i = i +1
    return X  
        


def run_ICA(wave_data):
    """
    Runs ICA on the wave files
    creates the modified files to run ICA on
    then runs ICA and imports resutls to final_FILE_NAME

    Parameters
    ----------
    wave_data : data imported

    Returns
    -------
    None.

    """
    random_matrix = np.random.uniform(0.5,2.5,(6,6))
    X = plot_sound(wave_data,"")
    X = np.array(X)
    X = np.c_[X[0],X[1],X[2],X[3],X[4],X[5]]
    X = X/ X.std(axis=0)
    S = np.dot(X,random_matrix.T)
    i=0
    
    for file in file_list:
        wavfile.write("modified_"+file,8000,S[:,i].astype(np.uint8))
        i= i +1 
    
    
    ica = FastICA(n_components=6,whiten="arbitrary-variance")
    S_ = ica.fit_transform(S)
    i=0
    for file in file_list:
        vector = S_[:,i]
    
        
        vector = vector-vector.mean()
        vector = vector / np.linalg.norm(vector)
          
        wavfile.write("final_"+file,8000,vector.astype(np.float32))
        i= i +1
        
    
    
    
#get wave data:

wave_data =[]
for file in file_list:    
    wave_data.append(wavfile.read(file))

run_ICA(wave_data)

#Sound after ICA
#wave_data =[]
#for file in file_list_final:
    #wave_data.append(wavfile.read(file))    
#plot_sound(wave_data,"after ICA")
    










