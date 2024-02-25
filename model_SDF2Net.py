''' 
Shallow to deep network
'''

import tensorflow as tf
import cvnn.layers as complex_layers
from tensorflow import keras
from SAR_utils import *
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, Input

###############################################################################
from SAR_utils import cmplx_SE_Block




def SDF2Net(X_cmplx, num_classes):
    
    cmplx_inputs = complex_layers.complex_input(shape=(X_cmplx.shape[1:]))
    
    # Shallow Path
    c0 = complex_layers.ComplexConv3D(16, activation='cart_relu', kernel_size=(3,3,3), padding="same")(cmplx_inputs)
    
    # Mid Path
    c1 = complex_layers.ComplexConv3D(16, activation='cart_relu', kernel_size=(3,3,3), padding="same")(cmplx_inputs)
    c1 = complex_layers.ComplexConv3D(16, activation='cart_relu', kernel_size=(3,3,3), padding="same")(c1)
    

    # Deep Path
    c2 = complex_layers.ComplexConv3D(16, activation='cart_relu', kernel_size=(3,3,3), padding="same")(cmplx_inputs)
    c2 = complex_layers.ComplexConv3D(16, activation='cart_relu', kernel_size=(3,3,3), padding="same")(c2)
    c2 = complex_layers.ComplexConv3D(16, activation='cart_relu', kernel_size=(3,3,3), padding="same")(c2)
   
    # Attenstion Block
    features_concat = tf.concat([c0, c1, c2], axis = 4)
    se = cmplx_SE_Block_3D(features_concat, se_ratio = 8)
    se = cmplx_SE_Block_3D(se, se_ratio = 8)
    se = cmplx_SE_Block_3D(se, se_ratio = 8)

    # Flatenning Features
    features_concat_flat = complex_layers.ComplexFlatten()(se)

    
    # Dense and Dropout
    c3 = complex_layers.ComplexDense(128, activation='cart_relu')(features_concat_flat)
    c3 = complex_layers.ComplexDropout(0.25)(c3)
    c4 = complex_layers.ComplexDense(64, activation='cart_relu')(c3)
    c4 = complex_layers.ComplexDropout(0.25)(c4)
    
    # Prediction
    predict = complex_layers.ComplexDense(num_classes,activation="softmax_real_with_abs")(c4)


    model = tf.keras.Model(inputs=[cmplx_inputs], outputs=predict)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    
    return model

#######################################################################################################
def cmplx_SE_Block_3D(xin, se_ratio = 8):
    # Squeeze Path
    xin = tf.transpose(xin, perm=[0, 1, 2, 4, 3])
    xin_gap =  GlobalCmplxAveragePooling3D(xin)
    sqz = complex_layers.ComplexDense(xin.shape[-1]//se_ratio, activation='cart_relu')(xin_gap)
    
    # Excitation Path
    excite1 = complex_layers.ComplexDense(xin.shape[-1], activation='cart_sigmoid')(sqz)
    
    out = tf.keras.layers.multiply([xin, excite1])
    out = tf.transpose(out, perm=[0, 1, 2, 4, 3])

    return out
    

def GlobalCmplxAveragePooling3D(inputs):
    inputs_r = tf.math.real(inputs)
    inputs_i = tf.math.imag(inputs)
    
    output_r = tf.keras.layers.GlobalAveragePooling3D()(inputs_r)
    output_i = tf.keras.layers.GlobalAveragePooling3D()(inputs_i)
    
    if inputs.dtype == 'complex' or inputs.dtype == 'complex64' or inputs.dtype == 'complex128':
           output = tf.complex(output_r, output_i)
    else:
           output = output_r
    
    return output

