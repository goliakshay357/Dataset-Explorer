import os
import streamlit as st

#machine learning packages
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

#model Packages
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def main():
	
	
	
	html_template = """
	<div style="background-color:pink; padding: 10px; margin-bottom:20px">
	<p style="margin:auto;font-size:30px"><strong>The Dataset Explorer Web Application</strong></p>
	</div> 
	"""
	st.markdown(html_template,unsafe_allow_html=True)

	activies = ["EDA","Plot","Model Building"]
	choice = st.sidebar.selectbox("Select Activity", activies)
	st.sidebar.text("All")

	def file_selector(folder_path = "./datasets"):
		filenames = os.listdir(folder_path)
		selected_filename = st.selectbox("Select Famous Datasets",filenames)
		array = [os.path.join(folder_path,selected_filename),selected_filename]
		return array

	filename = file_selector()
	
	#read csv
	df = pd.read_csv(filename[0])
	
	#or html
	html_template1 = """
	<div style="text-align:center;">
	<p style="margin:auto;"><strong>or</strong></p>
	</div> 
	"""
	st.markdown(html_template1,unsafe_allow_html=True)

	#upload dataSet
	data = st.file_uploader("Upload Dataset",type = ["csv","txt"])
	if data is not None:
		df = pd.read_csv(data)
	
	if data is not None: 
			st.success("Uploaded")
	else:
			st.write("DataSet: {}".format(filename[1]))    
 
	#show dataset
	if st.checkbox("Show Dataset:"):
		number = int(st.number_input("Number of rows to view:"))
		st.dataframe(df.head(number))

	#if EDA:
	if choice == 'EDA':     
	################
		#show columns
		if st.button("Column Names"):
			st.write(df.columns)

		#show Datastypes:
		if st.button("Show the Datastypes"):
			st.write(df.dtypes)

		#show Shape
		if st.button("Shape of Dataset"):
			temp = df.shape
			st.text("Number of Rows: {}".format(temp[0]))
			st.text("Number of columns: {}".format(temp[1]))

		#show specfied column
		if st.checkbox("Show Specified Columns"):
			all_columns = df.columns.tolist()
			selected_columns = st.multiselect("Select",all_columns)
			new_df = df[selected_columns]
			st.dataframe(new_df)

		#Summert=y:
		if st.button("Show the summary of the dataset"):
			st.write(df.describe().T)
	#########################
	if choice == 'Plot':
		##plots:
		st.header("Data Visualization:")
		st.subheader("Commanly used plots")

		#pie chat:
		if st.checkbox("Pie Plot"):
			all_column_names = df.columns.tolist()
			if st.button("Generate Plot of Target"):
				st.success("Generating a Pie Plot...")
				st.write(df.iloc[:,-1].value_counts().plot.pie(autopct = "%1.1f%%"))
				st.pyplot()
		#count Plot:
		if st.checkbox("Plot of value counts"):
			st.text("value counts by Target")
			all_column_names = df.columns.tolist()
			primary_col = st.selectbox("Primary column to groupby",all_column_names)
			selected_columns_names = st.multiselect("Select the columns",all_column_names)
			if st.button("Generate plot"):
				st.success("Generating")
				if selected_columns_names:
					vc_plot = df.groupby(primary_col)[selected_columns_names].count()
				else:
					vc_plot = df.iloc[:,-1].value_counts()
				st.write(vc_plot.plot(kind="bar"))
				st.pyplot()
		# corr plot
		if st.checkbox("Correlation Plot"):
			if st.button("Generate plot"):
						st.success("Generating...")
						st.write(sns.heatmap(df.corr(),annot=True))
						st.pyplot()


		#customisable plot
		st.subheader("Customizable Plot")
		all_column_names = df.columns.tolist()
		type_of_plot = st.selectbox("Select the type of box",["area","bar","line","hist","box","kde"])
		selected_columns_names = st.multiselect("Select Columns to Plot",all_column_names)

		if st.button("Generate Plot"):
			st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))

			#default plots by streamlit
			if type_of_plot == 'area':
				cust_data = df[selected_columns_names]
				st.area_chart(cust_data)
			elif type_of_plot == 'bar':
				cust_data = df[selected_columns_names]
				st.bar_chart(cust_data)
			elif type_of_plot == 'line':
				cust_data = df[selected_columns_names]
				st.line_chart(cust_data)
			elif type_of_plot:
				cust_data = df[selected_columns_names].plot(kind = type_of_plot)
				st.write(cust_data)
				st.pyplot()

	if choice == "Model Building":
		st.header("Building ML Models")
		st.warning("Make sure the last column is Target Variable")

		# Model Building
		X = df.iloc[:,0:-1] 
		Y = df.iloc[:,-1]
		seed = 7

		# prepare models
		print("LLLLLLL")
		models = []
		models.append(('LR', LogisticRegression()))
		models.append(('LDA', LinearDiscriminantAnalysis()))
		models.append(('KNN', KNeighborsClassifier()))
		models.append(('CART', DecisionTreeClassifier()))
		models.append(('NB', GaussianNB()))
		models.append(('SVM', SVC()))

		# evaluate each model in turn
			
		model_names = []
		model_mean = []
		model_std = []
		all_models = []

		for name,model in models:
			kfold = model_selection.KFold(n_splits=10,random_state = seed)
			cv_results = model_selection.cross_val_score(model,X,Y,cv=kfold,scoring = 'accuracy')
			model_mean.append(cv_results.mean())
			model_std.append(cv_results.std())
			model_names.append(name)				
			accuracy_results = {
								"model name":name,
								"model_accuracy":cv_results.mean(),
								"standard deviation":cv_results.std()
							   }
			all_models.append(accuracy_results)

		for_dataFrame = {
			'Model name' : model_names,
			'model_acuuracy' : model_mean,
			'model_std' : model_std
		}

		if st.checkbox("Metrics As Table111"):
			st.dataframe(pd.DataFrame(for_dataFrame))
			
		if st.checkbox("Metrics As JSON"):
				st.json(all_models)


		


if __name__ == '__main__':
	main()