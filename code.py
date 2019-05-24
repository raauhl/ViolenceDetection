import os
import random
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import time

maxFrames = 40
dims = 128
folds = 5
seed = random.uniform(0, 1)

"""
#code snippet to extract the dataset compressed in zip format , insert in file_name
from zipfile import ZipFile
file_name = ""
with ZipFile(file_name,'r') as zip:
    zip.extractall()
    print('extracted!')
"""

#given a openCV object refrencing the video, returns
def pullFrame(vidobj):
    framedVideo = []
    length = int(vidobj.get(cv2.CAP_PROP_FRAME_COUNT))
    success = 1
    count = 0
    flag = False
    if length < maxFrames:
        while True:
            success, image = vidobj.read()
            if success == 0 or count >= maxFrames:
                break
            resize = cv2.resize(image, (dims, dims))
            framedVideo.append(resize)
            count += 1
        while count < maxFrames:
            arr = np.zeros((dims,dims,3))
            framedVideo.append(arr)
            count += 1
    else:
        if length >= maxFrames and length <= 55:
            while True:
                success, image = vidobj.read()
                if success == 0 or count >= maxFrames:
                    break
                resize = cv2.resize(image, (dims, dims))
                framedVideo.append(resize)
                count += 1
        else:
            step = length // maxFrames
            while True:
                success, image = vidobj.read()
                if success == 0 or count >= length or len(framedVideo) >= maxFrames:
                    break
                if count % step == 0:
                    resize = cv2.resize(image, (dims, dims))
                    framedVideo.append(resize)
                count += 1
                
            while len(framedVideo) < maxFrames:
                flag = True
                arr = np.zeros((dims, dims, 3))
                framedVideo.append(arr)
                count += 1
            
    return (np.array(framedVideo))


def setEpochSteps(path, batchSize):
    listFights = os.listdir(path + 'fights' + '/')
    listnoFights = os.listdir(path + 'noFights' + '/')
    sum = len(listFights) + len(listnoFights)
    print('total data points : ' + str(sum))
    trainSteps = (sum * (folds-1)) // (folds * batchSize)
    testSteps = math.ceil(sum / (folds * batchSize))
    return (trainSteps,testSteps)

    
def generate_data(path, batch_size, currentFold, dataType):
    Dict = {}
    fileList = []
    folders = ['noFights', 'fights']

    for x in range(2):
        listData = os.listdir(path + folders[x] + '/')
        #sorted(listData)
        random.seed(seed)
        random.shuffle(listData)
        
        tmp = []
        for item in listData:
            Dict[path + folders[x] + '/' +  item] = x
            tmp.append(path + folders[x] + '/' +  item)
        listData = tmp
        
        chunkSize = len(listData) // folds
        listPartition = [listData[j * chunkSize:(j + 1) * chunkSize] for j in range((len(listData) + chunkSize - 1) // chunkSize)]

        if dataType == 'train':
            for k in range(folds):
                if k != currentFold:
                    fileList.extend(listPartition[k])
        if dataType == 'test':
                fileList.extend(listPartition[currentFold - 1])
    
    
    i = 0
    random.shuffle(fileList)
    while True:
        output_x = []
        output_y = []
        for b in range(batch_size):
            if i == len(fileList):
                i = 0
                random.shuffle(fileList)
                
            vid = fileList[i]
            vidObj = cv2.VideoCapture(vid)
            framedVideo = pullFrame(vidObj)
            output_x.append(framedVideo)
            yLabel = Dict[vid]
            output_y.append(yLabel)
            i += 1

        output_x = np.array(output_x)
        output_x = output_x / 255.0  # min-max normalization
        
        """
        #print(output_x.shape)
        #xMean = output_x.mean(axis=(4,0,3)).reshape((1, 1, dims, dims, 1))
        xMean = output_x.mean(axis=(4,1,0), keepdims=True)
        #xStd = output_x.std(axis=(4,0)).reshape((1, 1, dims, dims, 1))
        xStd = output_x.std(axis=(4,1,0), keepdims=True)
        newOutput_x = np.subtract(output_x, xMean) / xStd
        #print(newOutput_x)
        """
        
        output_y = np.array(output_y).reshape(-1, 1)
        yield (output_x, output_y)

from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, TimeDistributed
from keras.layers import AveragePooling2D, Dense, GRU, Input, LSTM
from keras.models import Model
from keras import optimizers

path = ''                          #path where dataset resides as raw videos
batchSize = 16
no_of_epochs = 5
start = time.time()
cvscores = []


(trainSteps,testSteps) = setEpochSteps(path, batchSize)
for num in range(folds):
    cnn = Sequential()
    cnn.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(dims, dims,3), padding='same'))
    cnn.add(MaxPooling2D(2))
    cnn.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'))
    cnn.add(MaxPooling2D(2))
    cnn.add(Conv2D(16, kernel_size=3, activation='relu', padding='same'))
    cnn.add(Conv2D(16, kernel_size=3, activation='relu', padding='same'))
    cnn.add(MaxPooling2D(2))
    cnn.add(Flatten())
    #cnn.summary()

    rnn = Sequential()
    rnn.add(GRU(64, return_sequences=True))
    rnn.add(GRU(64))

    dense = Sequential()
    dense.add(Dense(64,activation='relu'))
    dense.add(Dense(64,activation='relu'))
    dense.add(Dense(1,activation='sigmoid'))

    main_input = Input(shape = (maxFrames, dims, dims, 3))    #input a sequence of 40 images
    model = TimeDistributed(cnn)(main_input)                  #this makes cnn run 40 times
    model = rnn(model)
    model = dense(model)

    adm = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    final_model = Model(inputs = main_input, outputs = model)
    final_model.compile(loss='binary_crossentropy', optimizer=adm, metrics=['accuracy'])
    #final_model.summary()
    
    
    print("\n\nFOLD : " + str(num+1))
    history = final_model.fit_generator(generate_data(path, batchSize, num, 'train'), 
                                        steps_per_epoch = trainSteps,
                                        validation_data = generate_data(path, batchSize, num, 'test'),
                                        validation_steps= testSteps,
                                        epochs=no_of_epochs, 
                                        verbose=1)
    #scores = final_model.evaluate_generator(generate_data(path, batchSize, num, 'test'), steps= testSteps, verbose=1)
    cvscores.append(history.history.get('val_acc')[-1] * 100)
    #print(history.history.keys())
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    print('\n'+str(num+1)+ ". accuracy : " + str(history.history.get('val_acc')[-1]*100) + ' %')
    final_model.save('hockeyFight_model_' + str(num) + '.h5')  # creates a HDF5 file 
    
    
print("\n%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
end = time.time()
print('\nAvg. Execution time per fold: ' + str(((end - start)/60)/5) + ' mins')
print('\nTotal Execution time: ' + str((end - start)/60) + ' mins')
