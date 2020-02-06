#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
path="C:/Users/Dell/Downloads/mnist.npz"
data=np.load(path)
x_train,y_train=data['x_train'],data['y_train']
x_test,y_test=data['x_test'],data['y_test']
data.close()
import matplotlib.pyplot as plt

plt.imshow(x_test[1])


# In[2]:


model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10,activation=tf.nn.softmax))


# In[3]:


model.compile(optimizer="adam",loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.compile(optimizer="adam",loss='sparse_categorical_crossentropy',metrics=['accuracy'])
test_loss,test_accuracy=model.evaluate(x_test,y_test)
previous_TA,previous_testloss=test_accuracy,test_loss
current_epochs=0
DoNext=True

while(DoNext):
    next_epochs=current_epochs+1
    print(previous_TA*100," ",next_epochs)
    if((previous_TA*100)<90):
        model.fit(x_train,y_train,epochs=next_epochs)
        test_loss,test_accuracy=model.evaluate(x_test,y_test)
        previous_TA,previous_testloss=test_accuracy,test_loss
    else:
        DoNext=False


# In[4]:



print("Ëstimated Accuracy",test_accuracy)
print("Estimated Loss",test_loss)
print("Ëpochs",current_epochs)


# In[5]:


model.compile(optimizer="adam",loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.compile(optimizer="adam",loss='sparse_categorical_crossentropy',metrics=['accuracy'])
test_loss,test_accuracy=model.evaluate(x_test,y_test)
previous_TA,previous_testloss=test_accuracy,test_loss
current_epochs=0
DoNext=True

while(DoNext):
    next_epochs=current_epochs+1
    
    if((previous_TA*100)<95):
        model.fit(x_train,y_train,epochs=next_epochs)
        test_loss,test_accuracy=model.evaluate(x_test,y_test)
        previous_TA,previous_testloss=test_accuracy,test_loss
    else:
        DoNext=False

