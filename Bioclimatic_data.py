#Load observations file for comparison 
import pandas as pd
df_obs = pd.read_csv("observations.csv", index_col="observation_id") 
print(df_obs)

#Amount of Genus in dataset
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x = 'genus', data = df_obs)
plt.title('Countplot for Genus')
plt.ylabel('Number of Observations')
#plt.xticks(rotation='vertical')
plt.xlabel('Type of Genus')
plt.legend(['Genus'])
plt.show()

#Load pre-extracted environmental vectors
#Each line of this file correspond to an observation 
#and each column to one of the environmental variable.
df_env = pd.read_csv("environmental_vectors.csv", index_col="observation_id")
print(df_env)

#Drop observation_id numbers in dataset less than 20000000 
#because Observations dataset does not contain any
df_env = df_env.tail(df_env.shape[0] -688540)
print(df_env)

#Note that it typically contains NaN values due to absence of data over the seas and oceans 
#for both types of data as well as rivers and others for the pedologic data
print("Variables which can contain NaN values:")
df_env.isna().any()

#Create an array of the observation_id from the observation file for both train values and test values
obs_id_train = df_obs.index[df_obs["subset"] == "train"].values
obs_id_val = df_obs.index[df_obs["subset"] == "val"].values
env_id_arr = df_env.index.values

#print(obs_id_train)
#print(obs_id_val)
#print(env_id_arr)

#Reduce the observation_id set by checking if the observation is present in the environmental_id set
#and deleting the element from the array of not present
#If index from observation is not present in environmental set, then there is an error
id_train = []
for i in range(len(obs_id_train)):
    found = 0;
    for j in range(len(env_id_arr)):
        if(obs_id_train[i] == env_id_arr[j]):
            found = 1;
            break;
    if(found == 1):
        #print(i)
        id_train.append(obs_id_train[i])
#print(id_train)

id_val = []
for i in range(len(obs_id_val)):
    found = 0;
    for j in range(len(env_id_arr)):
        if(obs_id_val[i] == env_id_arr[j]):
            found = 1;
            break;
    if(found == 1):
       # print(i)
        id_val.append(obs_id_val[i])
        
print(df_env.loc[id_train].values)
print(df_env.loc[id_val].values)
 
#Split the dataset for training and testing based on the observation_id from observation file
X_train = df_env.loc[id_train].values
X_val = df_env.loc[id_val].values

#We need to handle properly the missing values
import numpy as np
from sklearn.impute import SimpleImputer
imp = SimpleImputer(
    missing_values=np.nan,
    strategy="constant",
    fill_value=np.finfo(np.float32).min,
)
imp.fit(X_train)

X_train = imp.transform(X_train)
X_val = imp.transform(X_val)

#Set the target values for training
y_train = df_obs.loc[id_train]["genus"].values
y_val = df_obs.loc[id_val]["genus"].values

#Now start training our Random Forest 
from sklearn.ensemble import RandomForestClassifier
est = RandomForestClassifier(n_estimators=16, max_depth=10, n_jobs=-1)
est.fit(X_train, y_train)

y_pred = est.predict(X_val) #predict output
print('Accuracy of Random Forest Classifier: ') 
est.score(X_val, y_val) #calculates how accurate random forest is

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
cm = confusion_matrix(y_val, y_pred) #creates a confusion matrix
cm #prints the matrix

print(classification_report(y_val, y_pred))

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)

y_pred_KNN = knn.predict(X_val) #predict output
print('Accuracy of KNN: ') 
knn.score(X_val, y_val) #calculates how accurate KNN is

cm = confusion_matrix(y_val, y_pred_KNN) #creates a confusion matrix
cm #prints thpe matrix

print(classification_report(y_val, y_pred_KNN))

from sklearn.svm import SVC  
clf = SVC(kernel='linear') 
clf.fit(X_train, y_train) # fitting x samples and y classes 

y_pred_SVM = clf.predict(X_val) #predict output
print('Accuracy of SVM: ') 
clf.score(X_val, y_val) #calculates how accurate SVM is

cm = confusion_matrix(y_val, y_pred_SVM) #creates a confusion matrix
cm #prints the matrix

print(classification_report(y_val, y_pred_SVM))
