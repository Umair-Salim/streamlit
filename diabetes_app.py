#importing libraaries
import streamlit as st 
import matplotlib.pyplot as plt  
import pandas as pd
import numpy  as np
import seaborn as sns 
from PIL import Image 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score
from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock

df = pd.read_csv('diabetes.csv')


#Headings
st.title('Diabetes prediction App')
st.sidebar.header('Patient Data')
st.subheader('Description stats of data')
st.write(df.describe())

#Dataa split
X = df.drop(['Outcome'],axis=1)
y = df['Outcome']


X_train, X_test, y_train,y_test = train_test_split(X, y, train_size=0.8, random_state=0)

#Function
def user_report():
    pregnancies = st.sidebar.slider('Pregnancies',0,20,4)
    glucose = st.sidebar.slider('Glucose',0,300,120)
    bp = st.sidebar.slider('BloodPressure',0,200,70)
    sk = st.sidebar.slider('SkinThickness',0,200,20)
    insulin = st.sidebar.slider('Insulin',0,1200,80)
    bmi = st.sidebar.slider('BMI',0,150,32)
    dpf = st.sidebar.slider('DiabetesPedigreeFunction',0.07,20.0,1.0)
    age = st.sidebar.slider('Age',0,200,33)

    user_report_data = {
        "pregnancies" : pregnancies,
        "glucose" : glucose,
        "bp" : bp,
        "sk" : sk,
        "insulin" : insulin,
        "bmi" : bmi,
        "dpf" : dpf,
        "age" : age}
    report_data = pd.DataFrame(user_report_data,index=[0])
    return report_data

#patient data
user_data = user_report()
st.subheader('Patient Data')
st.write('user_data')

#model
rc = RandomForestClassifier()
rc.fit(X_train,y_train)
user_result = rc.predict(user_data)

#visualization
st.title('Visualizing Patient Data')

#color function
if user_result[0]==0:
    color = 'blue'
else:
    color = 'red'

#Age vs Pregnancies
st.header('Pregnancy count graph(Other vs yours)')
fig_preg = plt.figure()
ax1= sns.scatterplot(x = 'Age', y = 'Pregnancies', data=df, hue = 'Outcome', palette = 'Greens')
ax2= sns.scatterplot(x = user_data['age'], y = user_data['pregnancies'], s=150, color = color)
plt.xticks(np.arange(10,200,20))
plt.yticks(np.arange(0,20,5))
plt.title('0 - Healthy & 1 - Diabetic')
st.pyplot(fig_preg)    

#Age vs BMI
st.header('BMI count graph(Other vs yours)')
fig_bmi = plt.figure()
ax1= sns.scatterplot(x = 'Age', y = 'BMI', data=df, hue = 'Outcome', palette = 'Greens')
ax2= sns.scatterplot(x = user_data['age'], y = user_data['bmi'], s=150, color = color)
plt.xticks(np.arange(10,200,20))
plt.yticks(np.arange(0,80,10))
plt.title('0 - Healthy & 1 - Diabetic')
st.pyplot(fig_bmi) 

#BMI vs Pregnancies
st.header('Pregnancies count graph(Other vs yours)')
fig_preg_bmi = plt.figure()
ax1= sns.scatterplot(x = 'BMI', y = 'Pregnancies', data=df, hue = 'Outcome', palette = 'Greens')
ax2= sns.scatterplot(x = user_data['bmi'], y = user_data['pregnancies'], s=150, color = color)
plt.xticks(np.arange(10,80,10))
plt.yticks(np.arange(0,20,5))
plt.title('0 - Healthy & 1 - Diabetic')
st.pyplot(fig_preg_bmi)



# output
st.header('Your Report')
output_H = 'You are Healthy'
output_D = 'You are diabetic'
if user_result[0]==0:
    output_H
    st.balloons()
else:
    output_D
    st.warning('Please take care of your health')

#accuracy
st.title('Output')
st.subheader('Accuracy')
st.write(str(accuracy_score(y_test, rc.predict(X_test))*100))


#precission score
st.subheader('precission score')
st.write(str(precision_score(y_test,rc.predict(X_test))*100))

#recall score
st.subheader('recall score')
st.write(str(recall_score(y_test,rc.predict(X_test))*100))

#confusion matrix
st.subheader('confusion matrix')
st.write(str(confusion_matrix(y_test,rc.predict(X_test))*100))

#showing confusion matrix

# Get and reshape confusion matrix data

matrix = confusion_matrix(y_test, rc.predict(X_test))
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

# Build the plot
fig_mat=plt.figure(figsize=(16,7))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=plt.cm.Greens, linewidths=0.2)

# Add labels to the plot
class_names = ['Pregnancies','Age']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Model')
st.pyplot(fig_mat)






