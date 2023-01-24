# -*- coding: utf-8 -*-
"""
Names: Valeri Materman 321133324
       Idit Avrahami 207565748
"""


import numpy as np
from scipy.io import wavfile 
from sklearn import preprocessing
import librosa
from scipy import signal
from sklearn import  metrics,linear_model
from sklearn. model_selection import cross_val_predict
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

tags =[   "noise",  "person","person" ,  "person",   "noise",   "person"]
file_list = ["source1.wav","source2.wav","source3.wav","source4.wav","source5.wav","source6.wav"]
file_list_final = ["final_source1.wav","final_source2.wav","final_source3.wav","final_source4.wav","final_source5.wav","final_source6.wav"]
##################################################################################### Classification:

def add_noise(sample,noise_factor):
    """
    Adds noise to audio sample

    Parameters
    ----------
    sample : array of wav data
    noise_factor : the factor

    Returns
    -------
    augmented_data : TYPE
        DESCRIPTION.

    """
    noise = np.random.randn(len(sample))
    augmented_data = sample + noise_factor * noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(sample[0]))
    return augmented_data


def manipulate(data,pitch_factor):
    """
    Changes pitch of wave file

    Parameters
    ----------
    data : the sample data
        
    pitch_factor : int or float of the factor

    Returns
    -------
    data modifie.

    """
    return librosa.effects.pitch_shift(data.astype(np.float32), 8000, pitch_factor)


def varMatrix(sample):
    """
    calculates the variance of the matrix

    Parameters
    ----------
    sample : [129*223] array
        the music sample Sxx matrix

    Returns
    -------
    double
        the varienc of the imag.

    """
    return np.var(sample) 



def sum_matrix(sample):
    """
    Calculates the sum of vlaues in the matrix

    Parameters
    ----------
    sample : [129*223] array
        the music sample Sxx matrix

    Returns
    -------
    double
        the sum of values in the whole matrix

    """
    return np.sum(sample)     

def mean_matrix(sample):
    """
    Returns the mean of the sample

    Parameters
    ----------
    sample : matrix of Sxx

    Returns
    -------
    mean as float

    """
    results = []
    for i in range(len(sample[0])):
        results.append(max(sample[:,i])-min(sample[:,i]))
    return np.mean(results)

def number_of_cols_with_close_to_zero_var(sample):
    """
    Count number of cols in matrix that have var close to 0

    Parameters
    ----------
    sample : the Sxx matrix.

    Returns
    -------
    n : number

    """
    n = 0
    for i in range(len(sample[0])):
        array = sample[:,i].astype(np.float32)
        array = preprocessing.minmax_scale(array,feature_range=(-1, +1))
        if np.var(array) == 0.0:
            
            n = n +1
    
    return n


def number_of_max_points(sample):
    """
    Count number of maximum points 

    Parameters
    ----------
    sample : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    sample_scaled = preprocessing.minmax_scale(sample,feature_range=(-1, +1))
    max_point = sample_scaled.max()
    delta = 0.4
 
    return ((sample_scaled > max_point - delta) & (sample_scaled  < max_point + delta)).sum()

def number_of_min_points(sample):
        """
        Count number of min points 

        Parameters
        ----------
        sample : the data

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        sample_scaled = preprocessing.minmax_scale(sample,feature_range=(-1, +1))
        min_point = sample_scaled.min()
        delta = 0.0001
     
        return ((sample_scaled > min_point - delta) & (sample_scaled  < min_point + delta)).sum()


def mean_on_col(sample):
    return np.var(sample,axis = 1)

def indecies_max_val(sample):
    """
    Return the indecies of the max values

    Parameters
    ----------
    sample : the data

    Returns
    -------

    """
    return sample.argmax(axis=1)
                
def number_of_rows_with_close_to_zero_var(sample):
    """
    Count number of rows in matrix that have var close to 0

    Parameters
    ----------
    sample : the Sxx matrix.

    Returns
    -------
    n : number

    """
    n = 0
    for row in sample:
        array = row.astype(np.float32)
        array = preprocessing.minmax_scale(array,feature_range=(-1, +1))
        if np.var(array) == 0.0:
          
            n = n +1
    
    return n
              

def LR(y,feature_names,*args):
    """
    Run logistic regression based on y and feature argumets
    
    Parameters
    ----------
    y : array of text []
        the true classifcation.
    *args : arrays of feauters
        all the feature arrays
    
    Returns
    -------
    None.
    
    """
 
    feature_str = ""
    for name in feature_names:
        feature_str = feature_str + name
        feature_str = feature_str + ", "
        
    X = np.column_stack(args)
    X, y = shuffle(X,y,random_state=1)
  
    Y = y
    # Training Logistic regression   
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)
    
    logistic_classifier = linear_model.LogisticRegression(solver='lbfgs')
    logistic_classifier.fit(X_train, y_train)
    # show how good is the classifier on the training data
    
    expected = y_test
    predicted = logistic_classifier.predict(X_test)
    
    
    print("Logistic regression using %s features On 15 precent of the data: \
    \n%s\n" % ( \
    feature_str,  
    metrics.classification_report( \
    expected, \
    predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))     

    return logistic_classifier
  

  
def create_data_with_noise(X):
    """
    Adds noise to data

    Parameters
    ----------
    X : data

    Returns
    -------
    None.

    """
    X_train = None
    Y_train = None
    first = True
    
    index = 0
    for sample in X:
        if first:
            X_train = np.array([sample])
            Y_train = np.array([tags[index]])
        else:
            X_train = np.append(X_train,[sample],axis=0)
            Y_train = np.append(Y_train,tags[index])
        first = False

        for i in range(1,5):
     
            X_train = np.append(X_train,[add_noise(sample,i)],axis=0)
            Y_train = np.append(Y_train,tags[index])
            
        index = index +1
    return(X_train,Y_train)
    
def create_data_with_pitch(X,Y):
    """
    Adds pitch to data

    Parameters
    ----------
    X : data

    Returns
    -------
    None.

    """
    index = 0
    for sample in X:
        for i in range(2,10):
            X = np.append(X,[manipulate(sample,i)],axis=0)
            Y = np.append(Y,Y[index])
            
        index = index +1
    
    return(X,Y)


def data_to_Sxx_matrix(X_train):
    """
    Takes the wav data and makees it into an Sxx matrix we can work with later as an image

    Parameters
    ----------
    X_train : data of all samples

    Returns
    -------
    X_temp : after conversion

    """
    X_temp = None
    for i in range(len(X_train)):
      
        _, _, Sxx = signal.spectrogram(X_train[i],fs=8000)

        if i ==0 :
            X_temp = np.array([Sxx])
        else:
            X_temp = np.append(X_temp, [Sxx],axis= 0)
    return X_temp


    

wave_data = []
for file in file_list:
    wave_data.append(wavfile.read(file))

X = []
for data_of_wave in wave_data:    
   X.append(data_of_wave[1])
   

X_train, Y_train = create_data_with_noise(X)
X_train, Y_train = create_data_with_pitch(X_train,Y_train)
temp = X_train
for i in range(len(X_train)):

    vector = X_train[i]
    
    
    vector = vector-vector.mean()
    vector = vector / np.linalg.norm(vector)

    X_train[i] = vector
    
X_train = data_to_Sxx_matrix(X_train)
"""
Dict of all feature functions
"""
feature_func_dict = {
       
        varMatrix : "variance of matrix",
        sum_matrix : "the sum of the matrix",
        number_of_cols_with_close_to_zero_var: "cols with 0 variance",
        number_of_rows_with_close_to_zero_var: "rows with 0 variance",
        mean_matrix : "mean of matrix",
        number_of_max_points: "number of max points",
        number_of_min_points : "number of min points",
        mean_on_col: "mean on colunm",
        indecies_max_val : "max value"
       
    }

     
feature_array_1 = np.array(list(map(sum_matrix, X_train)))
feature_array_2 = np.array(list(map(varMatrix, X_train)))
feature_array_3 = np.array(list(map(number_of_cols_with_close_to_zero_var, X_train)))
feature_array_4 = np.array(list(map(number_of_rows_with_close_to_zero_var, X_train)))
feature_array_5 = np.array(list(map(mean_matrix, X_train)))
feature_array_6 = np.array(list(map(number_of_max_points, X_train)))
feature_array_7 = np.array(list(map(number_of_min_points, X_train)))
feature_array_8 = np.array(list(map(mean_on_col, X_train)))
feature_array_9 = np.array(list(map(indecies_max_val, X_train)))

feature_names = []
feature_names.append(feature_func_dict[sum_matrix])
feature_names.append(feature_func_dict[varMatrix])
feature_names.append(feature_func_dict[number_of_cols_with_close_to_zero_var])
feature_names.append(feature_func_dict[mean_on_col])
feature_names.append(feature_func_dict[indecies_max_val])
l_r_model = LR(Y_train,feature_names,\

                          feature_array_1,\
                          feature_array_2,\
                          feature_array_3,\
                          feature_array_4,\
                          feature_array_8,\
                          feature_array_9
                         )


wave_data = []

for file in file_list_final:
    wave_data.append(wavfile.read(file)[1])
    
X_temp = None

#Getting the Sxx:
for i in range(len(wave_data)):
  
    _, _, Sxx = signal.spectrogram(wave_data[i],fs=8000)
    if i ==0 :
        X_temp = np.array([Sxx])
    else:
        X_temp = np.append(X_temp, [Sxx],axis= 0)


f_a_1 = np.array(list(map(sum_matrix, X_temp)))
f_a_2 = np.array(list(map(varMatrix, X_temp)))
f_a_3 = np.array(list(map(number_of_cols_with_close_to_zero_var, X_temp)))
f_a_4 = np.array(list(map(number_of_rows_with_close_to_zero_var, X_temp)))
f_a_8 = np.array(list(map(mean_on_col, X_temp)))
f_a_9 = np.array(list(map(indecies_max_val, X_temp)))

#Runing with our mode on the test data
X = np.column_stack((f_a_1,f_a_2,f_a_3,f_a_4,f_a_8,f_a_9))
X_scaled = preprocessing.scale(X)
expected = ["person","person","person","noise","person","noise"]
predicted = l_r_model.predict(X)
print("Logistic regression on created data:\n%s\n" % (
metrics.classification_report(
expected,
predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))


