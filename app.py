import os
import streamlit as st

#machine learning packages
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

def main():
    st.title("Hello")
    html_template = """
    <div style="background-color:pink;">
    <p>The Dataset Exploror Web Application</p>
    </div>
    """
    st.markdown(html_template,unsafe_allow_html=True)

    def file_selector(folder_path = "./datasets"):
        filenames = os.listdir(folder_path)
        selected_filename = st.selectbox("Select the Dataset",filenames)
        array = [os.path.join(folder_path,selected_filename),selected_filename]
        return array

    filename = file_selector()
    st.write("Dataset: {}".format(filename[1])) 
    st.write("Local Path: {}".format(filename[0]))
    
    #read csv
    df = pd.read_csv(filename[0])
    
    #show dataset
    if st.checkbox("Show Dataset:"):
        number = int(st.number_input("Number of rows to view:"))
        st.dataframe(df.head(number))

    #show columns
    if st.button("Column Names"):
        st.write(df.columns)

    #show Shape
    if st.checkbox("Shape of Dataset"):
        temp = df.shape
        st.text("Number of Rows: {}".format(temp[0]))
        st.text("Number of columns: {}".format(temp[1]))

    #show specfied column
    if st.checkbox("Show Specified Columns"):
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect("Select",all_columns)
        new_df = df[selected_columns]
        st.dataframe(new_df)

if __name__ == '__main__':
    main()