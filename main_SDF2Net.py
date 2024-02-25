import scipy.io as sio
import numpy as np
from SAR_utils import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from model_SDF2Net import SDF2Net
from tensorflow import keras
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from Load_Data import load_data


# Get the data
dataset = 'FL_T' #
windowSize = 5 #int(input("Enter the window size\n"))
train_per = 1 #int(input("Enter the percentage of training data:\n"))
data, gt = load_data(dataset)

# Standardize the data
data = Standardize_data(data)



X_coh, y = createComplexImageCubes(data, gt, windowSize)
X_coh = np.expand_dims(X_coh, axis=4)


X_train, X_test, y_train, y_test = splitTrainTestSet(X_coh, y, 1-train_per/100, randomState=345)
del data


print("Training Percentage = ", len(X_train)/len(X_coh)*100, "%")


y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

model = SDF2Net(X_train, num_classes(dataset))
model.summary()

# Setup early stopper using callbacks
from tensorflow.keras.callbacks import EarlyStopping
early_stopper = EarlyStopping(monitor='accuracy', 
                              patience=10,
                              restore_best_weights=True
                              )


    
# Perform Training    
history = model.fit(X_train, y_train,
                            batch_size = 64, 
                            verbose = 1, 
                            epochs = 10, #100 for SF and FL 
                            shuffle = True,
                            #class_weight = class_weights,
                            callbacks = [early_stopper] )
    
    

Y_pred_test = model.predict([X_test])
y_pred_test = np.argmax(Y_pred_test, axis=1)
       
     
kappa = cohen_kappa_score(np.argmax(y_test, axis=1),  y_pred_test)
oa = accuracy_score(np.argmax(y_test, axis=1), y_pred_test)
confusion = confusion_matrix(np.argmax(y_test, axis=1), y_pred_test)
each_acc, aa = AA_andEachClassAccuracy(confusion)
    

    

print('Average Accuracy = ', format(aa*100, ".4f"))
print('Overall Accuracy = ', format(oa*100, ".4f"))
print('Kappa x 100 = ', format(kappa*100, ".4f"))


# Create the class map (Very Fast way)
data, gt = load_data(dataset)
data = Standardize_data(data)

X, y = createComplexImageCubes(data, gt, windowSize, removeZeroLabels = False)
X = np.expand_dims(X, axis=4)


Y_pred_test = model.predict(X)
y_pred_test = (np.argmax(Y_pred_test, axis=1)).astype(np.uint8)

Y_pred = np.reshape(y_pred_test, gt.shape) + 1

gt_binary = gt.copy()

gt_binary[gt_binary>0]=1


predicted_map = Y_pred*gt_binary

# Plot Refernce Map
plt.imshow(gt, vmin=0, vmax= num_classes(dataset), cmap='jet')


#Plot Predicted Map
plt.imshow(predicted_map, vmin=0, vmax= num_classes(dataset), cmap='jet')


sio.savemat('./SDF2Net.mat', {'SDF2Net': new_map})









