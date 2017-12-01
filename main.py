#########################################################################
######## SOURCE CODE FOR SCIKIT-LEARN IMAGE RECOGNITION TUTORIAL ########
#########################################################################

import scipy.io 
import numpy as np
import matplotlib.pyplot as plt 
# %matplotlib inline 				# for Jupyter notebooks only

from sklearn.utils import shuffle 
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split 


#########################################################################


# load our data file 
train_data = scipy.io.loadmat('extra_32x32.mat') 

# extract the images (X) and labels (y) from the dict
X = train_data['X'] 
y = train_data['y'] 

# view an image (e.g. 25) and print its corresponding label
img_index = 25
plt.imshow(X[:,:,:,img_index])
plt.show()
print(y[img_index])


#########################################################################


# reshape our matrices into 1D vectors and shuffle (still maintaining the index pairings)
X = X.reshape(X.shape[0]*X.shape[1]*X.shape[2],X.shape[3]).T 
y = y.reshape(y.shape[0],) 
X, y = shuffle(X, y, random_state=42)  # use a seed of 42 to replicate the results of tutorial

# optional: reduce dataset to a selected size (rather than all 500k examples)
size = X.shape[0] 	# change to real number to reduce size
X = X[:size,:] 		# X.shape should be (num_examples,img_dimensions*colour_channels)
y = y[:size] 		# y.shape should be (num_examples,)


#########################################################################


# split data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# define our classifier and print out specs
clf = RandomForestClassifier() 
print(clf) 

# fit the model on training data and predict on unseen test data
clf.fit(X_train, y_train) 
preds = clf.predict(X_test) 
# pred = x[:-1,:].reshape(1, -1) 	# if predicting a single example it needs reshaping

# check the accuracy of the predictive model
print("Accuracy:", accuracy_score(y_test,preds)) 



