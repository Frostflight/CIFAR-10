def train():
    #---------------------------------------------------------------------------------------------------
    #TRAINING DATA LOCATIONS
    #---------------------------------------------------------------------------------------------------
    PATH = os.getcwd()
    trainPath = PATH+'/data/train/'
    trainData = os.listdir(trainPath)


    #---------------------------------------------------------------------------------------------------
    #LOAD TRAINING DATA
    #---------------------------------------------------------------------------------------------------
    count = 0
    imgArray = np.empty((len(trainData),32,32,3), dtype=float)
    Y_train = np.empty((len(trainData)), dtype=float)
    for sample in trainData:
        if sample != "Thumbs.db":
            print("Loading ",sample)
            img = Image.open(trainPath+sample)
            img = img.resize((32, 32), Image.ANTIALIAS)
            img = img.convert("RGB")
            imgArray[count] = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)
            if "cat" in sample:
                Y_train[count] = 1
            else:
                Y_train[count] = 0
            count += 1

    X_train = imgArray
    X_train = X_train.astype('float32')
    X_train /= 255
    Y_train = np_utils.to_categorical(Y_train, 2)

    print("Initialization complete")

    #---------------------------------------------------------------------------------------------------
    #KERAS MODEL DEFINITION
    #---------------------------------------------------------------------------------------------------
    model = Sequential()

    model.add(keras.layers.Conv2D(512,(3,3),activation='relu',input_shape=(32,32,3)))
    model.add(keras.layers.Conv2D(512,(3,3),activation='relu'))
    model.add(keras.layers.Conv2D(512,(3,3),activation='relu'))
    model.add(keras.layers.Conv2D(512,(3,3),activation='relu'))
    model.add(keras.layers.Conv2D(512,(3,3),activation='relu'))
    model.add(keras.layers.Conv2D(512,(3,3),activation='relu'))
    model.add(keras.layers.Conv2D(512,(3,3),activation='relu'))
    model.add(keras.layers.Conv2D(512,(3,3),activation='relu'))
    model.add(keras.layers.Conv2D(512,(3,3),activation='relu'))
    model.add(keras.layers.Conv2D(512,(3,3),activation='relu'))
    model.add(keras.layers.Conv2D(512,(3,3),activation='relu'))
    model.add(keras.layers.Conv2D(512,(3,3),activation='relu'))
    model.add(keras.layers.Conv2D(512,(3,3),activation='relu'))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))

    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    plotConv(model,0)
    
    for i in range(0,101):
        if i > 0:
            model = keras.models.load_model("CatModelV8-"+str(i-1)+".h5")
            print("Loaded CatModelV8-"+str(i-1)+".h5")
            plotConv(model,i)
        model.fit(X_train, Y_train,batch_size=100,epochs=1,verbose=1)
        model.save("CatModelV8-"+str(i)+".h5")
        print("Model",i,"saved")

    print("Complete!")


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#VISUALIZATION CODE IS NOTE MINE, I JUST MODIFIED IT:
#https://gist.github.com/oeway/f0ed87d3df671b351b533108bf4d9d5d
import pylab as pl
import matplotlib.cm as cm
import numpy.ma as ma
import matplotlib.pyplot as plt

def make_mosaic(imgs, nrows, ncols, border=1):
    nimgs = imgs.shape[0]
    imshape = imgs.shape[1:]
    
    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border),
                            dtype=np.float32)
    
    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in xrange(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols
        
        mosaic[row * paddedh:row * paddedh + imshape[0],
               col * paddedw:col * paddedw + imshape[1]] = imgs[i]
    return mosaic

from mpl_toolkits.axes_grid1 import make_axes_locatable

def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    divider = make_axes_locatable(ax)
    im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    pl.colorbar(im)

#--------------------------------------------------------------------------------------------------------------

def plotConv(model,epoch):
    os.mkdir('D:\\Jason Dunn\\FrostNet Evolved\\visualizations\\version-'+str(visCount)+'\\epoch-'+str(epoch))
    for k in [0,1,2,3,4,5,6,7,8,9,10,11,12]:
        W = model.get_layer(index=k).get_weights()[0]
        numOfNodes = model.get_layer(index=k).get_weights()[0].shape[3]
        if len(W.shape) == 4:
            W = np.squeeze(W)
            W = W.reshape((W.shape[0], W.shape[1], W.shape[2]*W.shape[3])) 
            fig, axs = plt.subplots(32,16, figsize=(8,8))
            fig.subplots_adjust(hspace = 0, wspace=0)
            axs = axs.ravel()
            for i in range(numOfNodes):
                axs[i].imshow(W[:,:,i])
                axs[i].tick_params( #Clear axis labels. They don't contribute anything meaningful
                    axis='both',
                    which='both',
                    bottom=False,
                    top=False,
                    left=False,
                    right=False,
                    labelbottom=False,
                    labeltop=False,
                    labelleft=False,
                    labelright=False
                    )
            pl.savefig('D:\\Jason Dunn\\FrostNet Evolved\\visualizations\\version-'+str(visCount)+'\\epoch-'+str(epoch)+'\\layer'+str(k)+'.png')
            plt.close()
                
#---------------------------------------------------------------------------------------------------
#IMPORT STATEMENTS
#---------------------------------------------------------------------------------------------------
import os
print("Imported OS")

visCount = len(os.listdir('D:\\Jason Dunn\\FrostNet Evolved\\visualizations\\'))
os.mkdir('D:\\Jason Dunn\\FrostNet Evolved\\visualizations\\version-'+str(visCount))

from PIL import Image, ImageFilter
print("Imported PIL")
import numpy as np
print("Imported NumPy")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import Sequential
print("Imported Keras Model")
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
print("Imported Keras Layers")
from keras.utils import np_utils
print("Imported Keras Utilities")
import keras
print("Imported Keras")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
train()

