from tensorflow.keras.models import load_model , Sequential
from tensorflow.keras import Model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
import pandas as pd

(_ ,_) ,(test  , _) = cifar10.load_data()

img_to_process = test[1000]


model= load_model("final_model.h5") 




kernel_list= model.get_weights()[0]

#image kernels visualisation 
for image_kernel_iter in range(1 ,33):
    plt.subplot(4 ,8 ,image_kernel_iter)
    plt.imshow(kernel_list[: , :  , 0 , image_kernel_iter-1] ,cmap ="binary")
    
plt.show()
    

conv_layers_outputs= [0 , 3] # without activation funtion


#conv_layers_outputs= [1 , 4] # with ativation funtion (activation maps)

#model to plot feature maps

outputs= [model.layers[iter].output for iter in conv_layers_outputs]
new_model =Model(inputs = model.inputs ,outputs = outputs) 



to_predict = img_to_process.reshape(1 ,32 ,32 ,3)
score = new_model.predict(to_predict)


#plot feature maps after first convolution process
feature_map1 = score[0].reshape(29 ,29 ,32)

for conv_layer1_index in range(1 , 33):
    plt.subplot(4 , 8 ,conv_layer1_index)
    plt.imshow(feature_map1[: ,: ,conv_layer1_index-1] ,cmap = "binary")
    
plt.show()


#plot feature maps after first convolution process
feature_map2 =score[1].reshape(15 ,15,16)


for conv_layer2_index in range(1 ,17):
    plt.subplot(4 ,4 ,conv_layer2_index)
    plt.imshow(feature_map2[:,: ,conv_layer2_index-1] ,cmap ="binary")

plt.show()


#cifar10 datasets categories
prediction ={0: "airplane" ,
             1: "automobile" ,
             2 : "bird" ,
             3 :"cat" ,
             4:"deer" ,
             5:"dog" ,
             6:"frog" ,
             7:"horse" ,
             8:"ship" ,
             9:"truck" }

final_prediction = model.predict_classes(img_to_process.reshape(1 ,32 ,32 ,3))
if final_prediction[0] in prediction:
    print("Algorithm recongnized a %s"  % prediction[final_prediction[0]])
