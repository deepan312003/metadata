import streamlit as st
from sklearn import datasets
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
st.set_page_config(
    page_icon="1.py",
    page_title= "Metadata tool"
)
st.title("MetaData tool")

st.write("""
         ### Explore different machine learning models
        """)

datasetname = st.sidebar.selectbox("Select dataset",("Iris","Breast Cancer","Wine"))

classifier_name = st.sidebar.selectbox("Select classifier",("KNN","SVM","Random Forest"))

def get_dataset(dataset_name):
    if dataset_name=="Iris":
        data=datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    x = data.data
    y = data.target
    names = data.feature_names
    return x,y,names

X,y,names = get_dataset(datasetname)
st.write("Shape of dataset",X.shape)
st.write("number of classes",len(np.unique(y)))
df = pd.DataFrame(X,y,columns=names)
st.dataframe(df)

def add_parameter_vi(clf_name):
    params = dict()
    if clf_name == "KNN":
        k=st.sidebar.slider("K",1,15)
        params["K"] = k
    elif clf_name == "SVM":
        C =st.sidebar.slider("C",0.01,10.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("max_depth",2,15)
        n_estimators = st.sidebar.slider("n_estimators",1,100)
        params["max_depth"] = max_depth
        params["n_estimators"]=n_estimators
    return params

params = add_parameter_vi(classifier_name)

def get_classifier(clf_name,params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
    else:
        clf = RandomForestClassifier(n_estimators=params["n_estimators"],max_depth=params["max_depth"],random_state=42)
    return clf

clf = get_classifier(classifier_name, params)

plot1 = plt.figure()
st.write("Independent variables are:")
option= st.selectbox('Select variables for their distribution plots',names)
for i in range(len(names)):
    if option == names[i]:
        index=i
plt.boxplot(x=X[:,index])
st.pyplot(plot1)

#Classification
def classification(X,y,clf):
    x_train,x_test,y_train,y_test = tts(X,y,test_size=0.2,random_state=42)
    clf.fit(x_train,y_train)
    y_predict = clf.predict(x_test)
    acc = accuracy_score(y_test, y_predict)
    return acc

acc = classification(X, y, clf)
st.write(f'Classifier Name ={classifier_name}')
st.write(f'Accuracy = {acc}')

#plot

pca =PCA(2)

X_projected = pca.fit_transform(X)
x1=X_projected[:,0]
x2=X_projected[:,1]

fig = plt.figure()
plt.scatter(x1,x2,c=y,alpha=0.8,cmap="rainbow")
plt.xlabel("Principal component 1")
plt.xlabel("Principal component 2")
plt.colorbar()
st.pyplot(fig)