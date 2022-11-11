import streamlit as st
import plotly.express as px
import pandas as pd
import seaborn as sns

header = st.container()
datasets = st.container()
features = st.container()


with header:
    #headers
    st.title('Practice app to publish on github and send to BABA G')
    st.markdown('**In this app we will work on titanic dataset and go through visualizations of scatter plot via the town from which passengers embarked**')


with datasets:

    st.header('Importing titanic dataset and checking it.')
    #importing data set
    df = sns.load_dataset('titanic')
    df = df.dropna()
    st.write(pd.set_option('display.max_columns', None))
    st.write(df.head())
    st.write(df.columns)

    # summary stat
    st.write(df.describe())


with features:

    st.header('Adding features and graphs')
    
    #data management

    embark_option = df['embark_town'].unique().tolist()
    embark_town = st.selectbox('Which town should we plot?', embark_option,0)

    df = df[df['embark_town'] == embark_town]

    #plotly

    fig = px.scatter(df, x='age', y='pclass', size='fare', color='sex', hover_name='survived',
                log_x=True, size_max = 55, range_x=[1,200], range_y=[0,4])
    fig.update_layout(width=800,height=400)

    st.write(fig)