from tensorflow.keras.models import load_model , Sequential
from tensorflow.keras import Model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
import pandas as pd

(_ ,_) ,(test  , _) = cifar10.load_data()

model= load_model("final_model.h5") 


kernel_list= model.get_weights()[0]

#image kernel visualisation 
for image_kernel_iter in range(1 ,33):
    plt.subplot(4 ,8 ,image_kernel_iter)
    plt.imshow(kernel_list[: , :  , 0 , image_kernel_iter-1] ,cmap ="binary")
    
plt.show()
    
   

plt.show()

outs= [0 , 3]
outputs= [model.layers[iter].output for iter in outs]

new_model =Model(inputs = model.inputs ,outputs = outputs) 

to_predict = test[1000].reshape(1 ,32 ,32 ,3)

score = new_model.predict(to_predict)


feature_map1 = score[0].reshape(29 ,29 ,32)

for iter in range(1 , 33):
    plt.subplot(4 , 8 ,iter)
    plt.imshow(feature_map1[: ,: ,iter-1])
    
plt.show()

feature_map2 =score[1].reshape(15 ,15,16)

for iter in range(1 ,17):
    plt.subplot(4 ,4 ,iter)
    plt.imshow(feature_map2[:,: ,iter-1])

plt.show()

