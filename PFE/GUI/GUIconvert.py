#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function
from tkinter import *
from PIL import Image, ImageTk
#from load import Model
import webbrowser
import numpy as np
import pickle
from tkinter import filedialog as fd
from os import path
from keras.models import load_model
from keras.optimizers import Adam
from keras.backend import clear_session
import tensorflow as tf
from keras.utils import plot_model
from keras import backend as K
from keras.layers import Layer
from keras import activations
from keras import utils
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint , TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from skimage import  color
import colorsys

def convertHSL(image):    
    for x in range(0, image.shape[0]):
        for x1 in range(0, image.shape[1]):
            image[x][x1][0] ,image[x][x1][1] ,image[x][x1][2]= colorsys.rgb_to_hls(image[x][x1][0] ,image[x][x1][1] ,image[x][x1][2])            
    return image


def convert(colorspace ,image):

    # Normalize data.    
    image = image.astype('float32') / 255
    #conver to HSL
    if colorspace=="HSL" :
            image =convertHSL(image)
    #conver to HSV
    if colorspace=="HSV" :
            image = color.rgb2hsv(image)
            
    #conver to XYZ
    if colorspace=="XYZ" :
            image = color.rgb2xyz(image)          
          
    #conver to LUV
    if colorspace=="LUV" :
            image = color.rgb2luv(image) 

    #conver to LAB
    if colorspace=="LAB" :
            image = color.rgb2lab(image) 
            
    #conver to YUV
    if colorspace=="YUV" :
            image = color.rgb2yuv(image) 
 
    print("finish")
    from scipy.misc import imsave
    imsave('test.png', image)
    return image

# the squashing function.
# we use 0.5 in stead of 1 in hinton's paper.
# if 1, the norm of vector will be zoomed out.
# if 0.5, the norm will be zoomed in while original norm is less than 0.5
# and be zoomed out while original norm is greater than 0.5.
def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
    return scale * x


# define our own softmax function instead of K.softmax
# because K.softmax can not specify axis.
def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)


# define the margin loss like hinge loss
def margin_loss(y_true, y_pred):
    lamb, margin = 0.5, 0.1
    return K.sum(y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (
        1 - y_true) * K.square(K.relu(y_pred - margin)), axis=-1)


class Capsule(Layer):
    """A Capsule Implement with Pure Keras
    There are two vesions of Capsule.
    One is like dense layer (for the fixed-shape input),
    and the other is like timedistributed dense (for various length input).

    The input shape of Capsule must be (batch_size,
                                        input_num_capsule,
                                        input_dim_capsule
                                       )
    and the output shape is (batch_size,
                             num_capsule,
                             dim_capsule
                            )

    Capsule Implement is from https://github.com/bojone/Capsule/
    Capsule Paper: https://arxiv.org/abs/1710.09829
    """

    def __init__(self,
                 num_capsule,
                 dim_capsule,
                 routings=3,
                 share_weights=True,
                 activation='squash',
                 **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)

    def build(self, input_shape):
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(1, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(input_num_capsule, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)

    def call(self, inputs):
        """Following the routing algorithm from Hinton's paper,
        but replace b = b + <u,v> with b = <u,v>.

        This change can improve the feature representation of Capsule.

        However, you can replace
            b = K.batch_dot(outputs, hat_inputs, [2, 3])
        with
            b += K.batch_dot(outputs, hat_inputs, [2, 3])
        to realize a standard routing.
        """

        if self.share_weights:
            hat_inputs = K.conv1d(inputs, self.kernel)
        else:
            hat_inputs = K.local_conv1d(inputs, self.kernel, [1], [1])

        batch_size = K.shape(inputs)[0]
        input_num_capsule = K.shape(inputs)[1]
        hat_inputs = K.reshape(hat_inputs,
                               (batch_size, input_num_capsule,
                                self.num_capsule, self.dim_capsule))
        hat_inputs = K.permute_dimensions(hat_inputs, (0, 2, 1, 3))

        b = K.zeros_like(hat_inputs[:, :, :, 0])
        for i in range(self.routings):
            c = softmax(b, 1)
            o = self.activation(K.batch_dot(c, hat_inputs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(o, hat_inputs, [2, 3])
                if K.backend() == 'theano':
                    o = K.sum(o, axis=1)

        return o

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)



class Demo2:
    def __init__(self, master):
        self.master = master
        self.frame = Frame(self.master)
        image = Image.open("./test.png")
        image = image.resize((200, 200), Image.ANTIALIAS)
        self.img1 = ImageTk.PhotoImage(image)
        self.w1 = Label(self.frame, image=self.img1)
        self.w1.pack()
        self.frame.pack()

    def close_windows(self):
        self.master.destroy()
        
class PredictCifar10(object):
    model = None
    graph = None
    def loadmodel(self ,modelfile):
        clear_session()
        self.modelfile = modelfile
        #load the model
        if "Capsule" not in self.modelfile :
            self.model  = load_model(self.modelfile)
        else :
            # A common Conv2D model
            input_image = Input(shape=(None, None, 3))
            x = Conv2D(64, (3, 3), activation='relu')(input_image)
            x = Conv2D(64, (3, 3), activation='relu')(x)
            x = AveragePooling2D((2, 2))(x)
            x = Conv2D(128, (3, 3), activation='relu')(x)
            x = Conv2D(128, (3, 3), activation='relu')(x)
            
            
            """now we reshape it as (batch_size, input_num_capsule, input_dim_capsule)
            then connect a Capsule layer.
            
            the output of final model is the lengths of 10 Capsule, whose dim=16.
            
            the length of Capsule is the proba,
            so the problem becomes a 10 two-classification problem.
            """
            
            x = Reshape((-1, 128))(x)
            capsule = Capsule(10, 16, 3, True)(x)
            output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule)
            self.model = Model(inputs=input_image, outputs=output)
            
            # we use a margin loss
            self.model.compile(loss=margin_loss, optimizer='adam', metrics=['accuracy'])
            #load weights
            self.model.load_weights(self.modelfile)
            self.model.summary()            
        self.graph = tf.get_default_graph()
        #self.model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=1e-3),metrics=['accuracy'])
        print("model loaded")
        self.modelloded=True
        self.val_acc_by_class.insert(END, "model loaded"+ '\n')
        if self.datasetloded:
            self.loadimagefromcifar(self.index)
            self.PredictR(0) 
            # Score trained model.
            self.val_acc_by_class.insert(END, "evaluating model in progress ...."+ '\n')
            with self.graph.as_default():
                scores = self.model.evaluate(self.x_test,self.y_test, verbose=1)
            self.val_acc_by_class.insert(END, " test loss : " +str(scores[0]) + '\n')
            self.val_acc_by_class.insert(END, "test accuracy : " + str(scores[1])+ '\n')
        # visualization graph to disk
        #plot_model(self.model, to_file=self.modelfile+".png", show_shapes=True)
            #self.model.summary()
    def loaddataset(self,datasetpath):
                #load the dataset
        if datasetpath:
            self.datasetpath = datasetpath
            with open(self.datasetpath, 'rb') as f:
                (self.x_train, self.y_train), (self.x_test, self.y_test) = pickle.load(f)         
            print("dataset loaded")
                    
        self.val_acc_by_class.insert(END, "dataset loaded"+ '\n')
        self.datasetloded = True
        if self.modelloded:
            self.loadimagefromcifar(self.index)
            print('image loaded')
            self.PredictR(0) 
            # Score trained model.
            self.val_acc_by_class.insert(END, "evaluating model in progress ...."+ '\n')
            with self.graph.as_default():
                scores = self.model.evaluate(self.x_test,self.y_test, verbose=1)
            self.val_acc_by_class.insert(END, " test loss : " +str(scores[0]) + '\n')
            self.val_acc_by_class.insert(END, "\n test accuracy : " + str(scores[1])+ '\n')


    def Predict(self, index):
        #if self.datasetpath: 
            truelabel="unknow"
            predict_correct =False 
            if index ==-1:
                img=self.img.reshape(-1,32,32,3)
            else:
                img=self.x_test[index].reshape(-1,32,32,3)

            
            #get the scores for each class
            with self.graph.as_default():
                scores = self.model.predict(img)
            number = 0
            bestScore = -1
            prediction = -1
            labelprediction=""
            # get the highest prediction
            for score in scores[0]:
                if score > bestScore:
                    bestScore = score
                    prediction = number
                number += 1
            labelprediction=self.label_names[prediction]
            # test if the prediction is right
            if index ==-1 :
                predict_correct =True
                truelabel="unknow"
            else:    
                if self.y_test[index][prediction] ==1:
                    predict_correct =True
                    truelabel=labelprediction
                else:   
                    for i in range(10):
                        if self.y_test[index][i] ==1:
                            truelabel=self.label_names[i]  
            print(labelprediction)    
            return labelprediction,scores[0] , predict_correct,truelabel

        
    def getimage(self, index):
        return self.x_test[index]
    
    def evaluate_by_class(self):
        val_acc_by_class=np.zeros(10)
        for y in range(10):
            total=0
            tmpval_acc=0   
            for x in range(self.x_test.shape[0]):
                if self.y_test[x][y]==1 :
                    total=total+1
                    #print(x)
                    with self.graph.as_default():
                        scores = self.model.predict(self.x_test[x].reshape(-1,32,32,3))
                    number = 0
                    bestScore = -1
                    prediction = -1
                    for score in scores[0]:
                        if score > bestScore:
                            bestScore = score
                            prediction = number
                        number += 1
                    if self.y_test[x][prediction] ==1 :
                        tmpval_acc=tmpval_acc+1
            val_acc_by_class[y] = tmpval_acc /total
            self.val_acc_by_class.insert(END, " in progress :" +str(y/10) +'\n')
            print("in progress" + str(y/10))
        return val_acc_by_class
    def __init__(self):
        #get the label name for each class
        f = open('batches.meta', 'rb')
        dict = pickle.load(f)
        self.label_names =dict['label_names'] 
        self.root = Tk()
        self.index =0
        self.modelloded = False
        self.datasetloded = False
        self.root.title("CIFAR10 Predictor")
        self.root.configure(background='white')
        self.root.resizable(0,0)
        self.modelfile= "/home/hatem/Desktop/PFE/GUI/RGB_sbm_NDA_resnet20-cnn-best.hdf5"
        self.cifar10path =""
        #self.model = load_model(self.modelfile)
        self.predictionLabel = Text(self.root, fg='white', height=1, width=30,
                                            borderwidth=0, highlightthickness=0,
                                            relief='ridge')
        self.predictionLabel.grid(row=0, column=6)
        self.trueLabel = Text(self.root, fg='black', height=1, width=30,
                                            borderwidth=0, highlightthickness=0,
                                            relief='ridge')
        self.trueLabel.grid(row=0, column=1)
        self.predictionScores = Text(self.root, height=10, width=30, padx=10,
                                        borderwidth=0, highlightthickness=0,
                                        relief='ridge')
        labelfont = ('times', 16, 'bold')
        self.predictionScores.configure(font=labelfont)
        self.predictionScores.grid(row=1, column=6)

        self.val_acc_by_class = Text(self.root, height=10, width=60, padx=10,
                                        borderwidth=0, highlightthickness=0,
                                        relief='ridge')
        self.val_acc_by_class.configure(font=labelfont)
        self.val_acc_by_class.grid(row=2, column=6)

        self.image = Canvas(self.root, width=500, height=500,
                                highlightthickness=0, relief='ridge',bg='white')
        self.image.create_image(10, 10, anchor=NW, tags="IMG")
        self.image.grid(row=1,  rowspan=5, columnspan=5)

        self.pre_button = Button(self.root, text='Precident', command=self.pre)
        self.pre_button.grid(row=6, column=2)

        self.next_button = Button(self.root, text='next', command=self.nex)
        self.next_button.grid(row=6, column=3)


        self.github = Label(self.root, text="https://github.com/hatemamine", cursor="hand2")
        self.github.bind("<Button-1>", self.openGitHub)
        self.github.grid(row=6, column=6)
        # Add Widgets to Manage Files model
        self.lb = Button(self.root, text="Browse to modelfile",
                        command=self.getFileName)
        self.lb.grid(column=1, row=7)
        # Add Widgets to Manage Files model
        self.l0 = Button(self.root, text="evaluate by class",
                        command=self.evaluate_by_classR)
        self.l0.grid(column=3, row=7)        
        # Add Widgets to Manage Files dataset
        self.l1 = Button(self.root, text="Browse to dataset",
                        command=self.getFileNamedataset)
        self.l1.grid(column=1, row=8)  
        self.loadimage = Button(self.root, text="Browse to image file",
                        command=self.getimages)
        self.loadimage.grid(column=1, row=9)  
        self.img = Image.open("into.jpg")
        self.resizeAndSetImage(self.img)
        self.root.mainloop()

    def getFileName(self):   
        fDir = path.dirname(__file__)
        self.modelfile = fd.askopenfilename(parent=self.root, initialdir=fDir ,filetypes=[("model", "*.hdf5")])
        self.val_acc_by_class.insert(END, "load model from"+ '\n')
        self.val_acc_by_class.insert(END, self.modelfile + '\n')
        self.val_acc_by_class.insert(END, "please waiting model are been loaded it my take some minute"+ '\n')
        import threading
        x = threading.Thread(target=self.loadmodel, args=(self.modelfile,))
        x.start()
    def getimages(self):   
        fDir = path.dirname(__file__)
        self.imgfile = fd.askopenfilename(parent=self.root, initialdir=fDir ,filetypes=[("model", "*.jpg")])
        self.img = Image.open(self.imgfile)
        self.resizeAndSetImage(self.img)
        import numpy as np
        self.img= np.array(self.img)
        from pathlib import Path 
        self.img = convert(Path(self.modelfile).name[:3] ,self.img)
        print(self.img)
        #image = Image.open("./test.png")
        #image = image.resize((32, 32), Image.ANTIALIAS)
        #self.img = ImageTk.PhotoImage(image)
        self.newWindow = Toplevel(self.root)
        self.app = Demo2(self.newWindow)
        from skimage.transform import resize
        self.img = resize(self.img,(32,32),anti_aliasing=True)
        #imageresized = self.img.resize((32,32), Image.ANTIALIAS)
        #from scipy.misc import imsave
        #imsave('test.png', imageresized)
        #print(self.img)
        #import numpy as np
        #self.img= np.array(self.img)
        self.PredictR(-1)


        

   

    def getFileNamedataset(self):
        fDir = path.dirname(__file__)
        self.cifar10path = fd.askopenfilename(parent=self.root, initialdir=fDir , filetypes=[("dataset", "*subtract_pixel_meancifar10_normalized.pkl")])
        from pathlib import Path    
        print(Path(self.cifar10path).name)
        if Path(self.cifar10path).name[:3]==Path(self.modelfile).name[:3]:
            self.val_acc_by_class.configure(fg='green')
            self.val_acc_by_class.insert(END, "load dataset from"+ '\n')
            self.val_acc_by_class.insert(END, self.cifar10path + '\n') 
            self.val_acc_by_class.insert(END, "please waiting dataset are been loaded it my take some minute"+ '\n')
            import threading
            x = threading.Thread(target=self.loaddataset, args=(self.cifar10path ,))
            x.start()
        else:
            self.val_acc_by_class.insert(END, "model and dataset mismatch"+ '\n' )
            self.val_acc_by_class.configure(fg='red')
   
    def openGitHub(self, event):
        webbrowser.open_new(r"https://github.com/hatemamine/impact-Of-Image-Colourspace-On-deep-learning-Convolution-Neural-Networks-Performance")
        

    def nex(self):
        if self.index < 10000:
            self.index=self.index+1 
            self.loadimagefromcifar(self.index)
            self.PredictR(0)
    
    def pre(self):
        if self.index > 0 :
            self.index=self.index-1  
            self.loadimagefromcifar(self.index)
            self.PredictR(0)
    
    def resizeAndSetImage(self, image):
        size = (400, 400)
        resized = image.resize(size, Image.ANTIALIAS)
        self.nnImage = ImageTk.PhotoImage(resized)
        self.image.delete("IMG")
        self.image.create_image(10, 10, image=self.nnImage, anchor=NW, tags="IMG")

        
    def loadimagefromcifar(self ,index): 
        from scipy.misc import imsave
        imsave('current.png', self.getimage(index))     # image no #
       

    def load_classes(self , index):
        file = 'batches.meta'
        f = open(file,'rb')
        dict = pickle.load(f)
        return dict['label_names'][index]
    
    def PredictR(self , ind):
        self.predictionLabel.delete(1.0, END)
        self.predictionScores.delete(1.0, END)
        self.trueLabel.delete(1.0, END)
        if ind ==-1:
            prediction, scores ,predict_correct , truelabel= self.Predict(-1)
        else :
            img = Image.open("current.png")
            self.resizeAndSetImage(img)
            prediction, scores ,predict_correct, truelabel = self.Predict(self.index)
            
        n = 0
        labelfont = ('times', 20, 'bold')
        self.predictionLabel.configure(font=labelfont)
        self.trueLabel.configure(font=labelfont)
        if predict_correct :
            self.predictionLabel.configure(bg='green')
        else:
            self.predictionLabel.configure(bg='red') 
        print(truelabel)
        self.predictionLabel.insert(END, "This is a {}".format(prediction))
        self.trueLabel.insert(END, "True label is "+truelabel)
        for score in scores:
            self.predictionScores.insert(END, "{}: {}\n".format(self.load_classes(n), float("{:.8f}".format(float(score)))))
            n += 1
            
    def evaluate_by_classR(self):  
        self.l0.configure(state=DISABLED)
        import threading
        x = threading.Thread(target=self.evaluate_by_classth, args=())
        x.start()

    def evaluate_by_classth(self):  
        val_acc_by_class = self.evaluate_by_class()
        n=0
        for score in val_acc_by_class:
            self.val_acc_by_class.insert(END, "{}: {}\n".format(self.load_classes(n), float("{:.8f}".format(float(score)))))
            n += 1   
        self.l0.configure(state=NORMAL)     
if __name__ == '__main__':
    PredictCifar10 = PredictCifar10()



