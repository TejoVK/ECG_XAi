import streamlit as st
import pandas as pd
from io import StringIO
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
import imblearn
import streamlit_authenticator as stauth
import pickle
from pathlib import Path
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import streamlit as st
from streamlit_login_auth_ui.widgets import __login__
import math

#use predict_proba
#divide the page into different sets: preprocessing, visualisation & prediction
#display the best values in 
#create a input for accpeting a test dataset and give output values for all the orws



__login__obj = __login__(auth_token = "courier_auth_token", 
                    company_name = "Shims",
                    width = 200, height = 250, 
                    logout_button_name = 'Logout', hide_menu_bool = False, 
                    hide_footer_bool = False, 
                    lottie_url = 'https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json')

LOGGED_IN = __login__obj.build_login_ui()

if LOGGED_IN == True:
    tab1, tab2,tab3 = st.tabs(["Preprocess","Apply Regression","Apply  Classification"])
    class DataPreprocessing:
        def __init__(self,data):
            self.data = data
            from warnings import filterwarnings
            filterwarnings("ignore")
            self.objects=DataPreprocessing.initialize()
            self.input = None
            self.output = None
            self.features = list(data.columns)
            self.output_name = None
            self.train_features,self.train_target,self.test_target,self.test_features,self.val_features,self.val_target = None,None,None,None,None,None
        def drop_columns(self,columns):
            self.data.drop(columns,axis=1,inplace=True)
            if type(columns) == list:
                self.features = [i for i in self.features if i not in columns]
            else:
                self.features.remove(columns)
        def handle_null(self,type='drop'):
            if type=='drop':
                self.data.dropna(axis=0,inplace=True)
            if type=="mean":
                self.data=self.data.apply(lambda x:x.fillna(x.mean()))
        def initialize():
            from sklearn import preprocessing,model_selection,decomposition
            return {
                    'Standard scaler':preprocessing.StandardScaler,
                    'Min Max Scalar':preprocessing.MinMaxScaler,
                    'PCA':decomposition.PCA,
                    'train test split':model_selection.train_test_split,
                }
        def out_in(self,output_name):
            self.input = self.data.drop(output_name,axis=1)
            self.output = self.data[output_name]
            self.features.remove(output_name)
            self.output_name = output_name
        def apply_count_vectorize(self,col,count_vect_obj=None):
            if count_vect_obj ==None:
                from sklearn.feature_extraction.text import CountVectorizer
                self.objects['Countvec_'+col] = CountVectorizer()#instantiating the count vectorizer
                self.data[col] = self.objects['Countvec_'+col].fit_transform(self.data[col])
            else:
                self.objects['Countvec_'+col] = count_vect_obj
                self.data[col] = self.objects['Countvec_'+col].fit_transform(self.data[col])
        def split(self,test_percent,validation_percent=0.1,rs = 42):
            self.train_features,self.test_features,self.train_target,self.test_target = self.objects['train test split'](self.input,self.output,test_size=test_percent,random_state=rs)
            self.test_features,self.val_features,self.test_target,self.val_target = self.objects['train test split'](self.test_features,self.test_target,test_size = validation_percent,random_state = rs)
        def get_object_column(self):
            import numpy as np
            edit_col = [i for i in self.features if self.data[i].dtype == np.object_]
            return edit_col
        def encode_categorical_columns(self):
            import numpy as np
            from sklearn.preprocessing import LabelEncoder
            label_encoder_objects ={}
            edit_columns = self.get_object_column()
            # st.write(edit_columns)
            name_mapping = {}
            for col in edit_columns:
                label_object = LabelEncoder()
                # conversion = label_object.get_params(deep=True)
                self.data[col]=label_object.fit_transform(self.data[col])
                label_encoder_objects[col+"_encoder_object"] = label_object
            self.objects['Label_Encoder'] = label_encoder_objects
            # val = label_object.classes_
            # for i in val:
            #     mapped_val = int(label_object.transform(list(i)))
            #     name_mapping[i] = mapped_val
            # return name_mapping
        def change_columns(self,columns):
            self.data = self.data[columns]
        def apply_smote_data(self):
            from imblearn.over_sampling import SMOTE
            smote_object = SMOTE()
            self.train_features,self.train_target = smote_object.fit_resample(self.train_features,self.train_target)
            self.objects['Smote object'] = smote_object
        def standardize_or_normalize(self,scale_type=None):
            if scale_type == "Standard":
                from pandas import DataFrame as df
                scale_object  = self.objects['Standard scaler']()
                st.write(scale_object)
                self.train_features=df(data = scale_object.fit_transform(self.train_features),columns = self.features)
                self.test_features = df(data = scale_object.transform(self.test_features),columns = self.features)
                self.val_features = df(data = scale_object.transform(self.val_features),columns = self.features)
            elif scale_type == "Normalize":
                from pandas import DataFrame as df
                scale_object  = self.objects['Min Max Scalar']()
                self.train_features=df(data = scale_object.fit_transform(self.train_features),columns = self.features)
                self.test_features = df(data = scale_object.transform(self.test_features),columns = self.features)
                self.val_features = df(data = scale_object.transform(self.val_features),columns = self.features)




    class MacineLearningClassification:
        def __init__(self,data_pr,prediction_array=None,k_fold_num=None,models=None):
            self.best_accuracy = 0
            self.prediction_array=prediction_array
            self.best_model = None
            self.best_model_object = None
            self.data = data_pr.data
            self.train_features = data_pr.train_features
            self.train_target = data_pr.train_target
            self.test_features = data_pr.test_features
            self.test_target = data_pr.test_target
            self.trained_models = []
            if models==None:
                from sklearn.linear_model import LogisticRegression
                from sklearn.tree import DecisionTreeClassifier
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.naive_bayes import GaussianNB
                from sklearn.neighbors import KNeighborsClassifier
                from sklearn.svm import SVC
                models = [LogisticRegression(),DecisionTreeClassifier(),RandomForestClassifier(),GaussianNB(),KNeighborsClassifier(),SVC()]
            self.model_evaluvation_dict = {str(i).replace("()",""):{'model_object':i} for i in models}
            self.model_prediction = {str(i).replace("()",""):None for i in models}
        def fit(self):
            for model,dic in self.model_evaluvation_dict.items():
                self.model_evaluvation_dict[model]['model_object'].fit(self.train_features,self.train_target)
                self.trained_models.append(self.model_evaluvation_dict[model]['model_object'])
                self.model_prediction[model] = self.model_evaluvation_dict[model]['model_object'].predict(self.test_features)
        def Score_test_data(self):
            for model,dic in self.model_evaluvation_dict.items():
                self.model_evaluvation_dict[model]['score on test data'] = self.model_evaluvation_dict[model]['model_object'].score(self.test_features,self.test_target)*100
                if (self.model_evaluvation_dict[model]['score on test data']>self.best_accuracy):
                    self.best_model = {'Model_obj':self.model_evaluvation_dict[model]['model_object'],
                                    'Name':model,
                                    'Accuracy':self.model_evaluvation_dict[model]['score on test data']}
                    self.best_accuracy = self.model_evaluvation_dict[model]['score on test data']
        def create_confusion_matrix(self):
            from sklearn.metrics import confusion_matrix
            for model,dic in self.model_evaluvation_dict.items():
                self.model_evaluvation_dict[model]['confusion matrix for test data'] = confusion_matrix(self.test_target,self.model_prediction[model]).tolist()
        def create_f1_precision_recall(self):
            from sklearn.metrics import f1_score,recall_score,precision_score
            for model,dic in self.model_evaluvation_dict.items():
                self.model_evaluvation_dict[model]['f1 score for test data'] = f1_score(self.test_target,self.model_prediction[model],average='macro')*100
                self.model_evaluvation_dict[model]['precision for test data'] = precision_score(self.test_target,self.model_prediction[model],average='macro')*100
                self.model_evaluvation_dict[model]['recall for test data'] = recall_score(self.test_target,self.model_prediction[model],average='macro')*100
        def evaluvate(self):
            import numpy as np
            self.fit()
            self.Score_test_data()
            self.create_confusion_matrix()
            self.create_f1_precision_recall()
            if type(self.prediction_array)==np.ndarray:
                self.model_evaluvation_dict['prediction']=self.best_model['Model_obj'].predict(np.array([self.prediction_array]))[0]
            for model in self.model_evaluvation_dict:
                if model!='prediction':
                    del self.model_evaluvation_dict[model]['model_object']
            self.best_model_object = self.best_model['Model_obj']
            del self.best_model['Model_obj']
            self.model_evaluvation_dict['best model'] = self.best_model
            return self.model_evaluvation_dict





    class MachineLearningRegression:
        def __init__(self,data_pr,prediction_array=None,k_fold_num=None,models=None):
            self.best_r2_score = 0
            self.best_model = None
            self.best_model_object = None
            self.prediction_array=prediction_array
            self.data = data_pr.data
            self.train_features = data_pr.train_features
            self.train_target = data_pr.train_target
            self.test_features = data_pr.test_features
            self.trained_models = []
            self.test_target = data_pr.test_target
            if models == None:
                from sklearn.linear_model import LinearRegression,Ridge,Lasso
                from sklearn.tree import DecisionTreeRegressor
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.neighbors import KNeighborsRegressor
                models = [LinearRegression(),Ridge(),Lasso(),DecisionTreeRegressor(),RandomForestRegressor(),KNeighborsRegressor()]
            self.model_evaluvation_dict = {str(i).replace("()",""):{'model_object':i} for i in models}
            self.model_prediction = {str(i).replace("()",""):None for i in models}
        def fit(self):
            for model,dic in self.model_evaluvation_dict.items():
                self.model_evaluvation_dict[model]['model_object'].fit(self.train_features,self.train_target)
                self.trained_models.append(self.model_evaluvation_dict[model]['model_object'])
                self.model_prediction[model] = self.model_evaluvation_dict[model]['model_object'].predict(self.test_features)
        def Score_test_dataset(self):
            from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,mean_absolute_percentage_error
            metrics = {'r2 score':r2_score,'MAE':mean_absolute_error,'MSE':mean_squared_error,'MAPE':mean_absolute_percentage_error}
            for model,dic in self.model_evaluvation_dict.items():
                for metric,obj in metrics.items():
                    self.model_evaluvation_dict[model][metric] = obj(self.model_prediction[model],self.test_target)
                    if self.model_evaluvation_dict[model]['r2 score']>self.best_r2_score:
                        self.best_model = {'Name':model,
                                        'r2 score':self.model_evaluvation_dict[model]['r2 score'],
                                            'model_obj':self.model_evaluvation_dict[model]['model_object']}
                        self.best_r2_score = self.model_evaluvation_dict[model]['r2 score']
        def evaluvate(self):
            import numpy as np
            self.fit()
            self.Score_test_dataset()
            if type(self.prediction_array)==np.ndarray:
                self.model_evaluvation_dict['prediction']=self.best_model['model_obj'].predict(np.array([self.prediction_array]))[0]
            for model in self.model_evaluvation_dict:
                if model!='prediction':
                    del self.model_evaluvation_dict[model]['model_object']
            self.best_model_object = self.best_model['model_obj']
            del self.best_model['model_obj']
            self.model_evaluvation_dict['best model'] = self.best_model
            return self.model_evaluvation_dict




    with tab1:
        uploaded_file = st.file_uploader("Enter the dataset you want to work on: ")
        if uploaded_file is not None:
            # Can be used wherever a "file-like" object is accepted:
            dataframe = pd.read_csv(uploaded_file)
            st.write(dataframe.head(5))
            data_p_object = DataPreprocessing(dataframe)
            from sklearn.preprocessing import LabelEncoder
            label_encoder_objects ={}
            # st.write(data_p_object.features)
            # st.write(data_p_object.data['diagnosis'].dtype)
            edit_col = [i for i in dataframe.columns if data_p_object.data[i].dtype == np.object_]
            edit_columns = edit_col
            # st.write(edit_col)
            if edit_col is None:
                st.error("The Dataset does'nt contain any categorical columns")
            if edit_col is not None:
                st.info("We found textual data in your dataset, thus we encoded it")
            # '''for col in edit_columns:
            #     label_object = LabelEncoder()
            #     data_p_object.data[col]=label_object.fit_transform(data_p_object.data[col])
            # #     label_encoder_objects[col+"_encoder_object"] = label_object'''
            # global lablevalu
            lablevalu = data_p_object.encode_categorical_columns()
            
            # st.write(lablevalu)
            # st.write(data_p_object.data.head())
            # data_p_object.objects['Label_Encoder'] = label_encoder_objects
            if(st.checkbox("Do you want to preprocess your data?")):
                
                # data_p_object = pd.DataFrame(data_p_object)
                st.write("The correlation matrix for your dataset:")
                corr = data_p_object.data.corr()
                mask = np.ones_like(corr,dtype=np.bool_)
                mask[np.tril_indices_from(mask)]=False
                mask[np.diag_indices_from(mask)]=True
                fig,axes = plt.subplots(nrows=1,ncols=1,figsize=(25,25))
                sns.heatmap(corr,annot=True,mask=mask,ax=axes,annot_kws={'fontsize':8},cbar=False,cmap='ocean_r')
                axes.set_title('\n Correlation Matrix for the dataset\n')
                st.write(fig)

                col_names = dataframe.columns
                col_to_remove = st.multiselect("Select the columns you want to remove: ",col_names)
                if col_to_remove is None:
                    None
                if col_to_remove is not None:
                    data_p_object.data.drop(col_to_remove,axis=1,inplace=True)
                    if type(col_to_remove) == list:
                        data_p_object.features = [i for i in data_p_object.features if i not in col_to_remove]
                    else:
                        data_p_object.features.remove(col_to_remove)

                    st.write("The new correlation matrix for your dataset:")
                    corr = data_p_object.data.corr()
                    mask = np.ones_like(corr,dtype=np.bool_)
                    mask[np.tril_indices_from(mask)]=False
                    mask[np.diag_indices_from(mask)]=True
                    fig,axes = plt.subplots(nrows=1,ncols=1,figsize=(25,25))
                    sns.heatmap(corr,annot=True,mask=mask,ax=axes,annot_kws={'fontsize':8},cbar=False,cmap='ocean_r')
                    axes.set_title('\n Correlation Matrix for the dataset\n')
                    st.write(fig)
                    # st.write(data_p_object.features)


                outliers = st.checkbox("Do you want to handel outliers in your data set? ")
                if outliers==True:
                    lis_for_target = data_p_object.features
                    to_find = st.multiselect("Select the features for outlier handeling: ", list(lis_for_target))
                    to_do = st.selectbox("Do you want to view the outliers or do you want to delete the outliers? ",("View","Handel"))
                    out_graph={}
                    if to_do == "View":
                        for item in to_find:
                            # plt.boxplot(list(float(to_find)))
                            fig,axes = plt.subplots(nrows=math.ceil(len(to_find)/2),ncols=2, sharey=False)
                            plt.boxplot(data_p_object.data[item])
                            st.write(fig)
                    if to_do == "Handel":
                        options = st.selectbox("What do you want to perform on the outliers? ",("Capping","Median Substitution","Deletion"))
                        if options == "Capping":
                            low=st.number_input("Enter the lower threshold percentile: ",1,100)
                            up=st.number_input("Enter the upper threshold percentile: ",1,100)
                            st.caption("Generally lower threshold of 10 and upper threshold of 90 are preferred")
                            for item in to_find:
                                # st.write(data_p_object.data[item])
                                fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))#a matplotlib function to plot multiple plots in different graphs sdjecent to each other.
                                st.write(item)
                                ax1.boxplot(data_p_object.data[item])
                                ax1.set_title('Data before capping')
                                low_threshold = np.percentile(data_p_object.data[item],low)#lower threshold
                                up_threshold = np.percentile(data_p_object.data[item],up)#upper threshold
                                capped_data = np.clip(data_p_object.data[item], low_threshold, up_threshold)#a simple numpy function to replace the outlier values with corresponding values.
                                data_p_object.data[item] = capped_data
                                # st.write(data_p_object.data[item])
                                # Plot the boxplots on the subplots                      
                                ax2.boxplot(capped_data)
                                ax2.set_title('Data after capping')

                                # Show the plot
                                plt.show()
                                st.write(fig)
                        if options == "Median Substitution":
                            for item in to_find:
                                # data_p_object.data[item]
                                fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
                                st.write(item)
                                ax1.boxplot(data_p_object.data[item])
                                ax1.set_title('Data before replacing')
                                median=data_p_object.data[item].median()
                                data_p_object.data[item] = np.where( data_p_object.data[item] >90, median,data_p_object.data[item])
                                ax2.boxplot(data_p_object.data[item])
                                ax2.set_title('Data after replacing')
                                plt.show()
                                st.write(fig)

                        # if options == "Deletion":





                null_val=st.selectbox("How would you like to handle you null data: ", ("drop them","replace with mean"))
                if(null_val=="drop them"):
                    data_p_object.data.dropna(axis=0,inplace=True)
                else:
                    data_p_object.data=data_p_object.data.apply(lambda x:x.fillna(x.mean()))
                

                st.table(data_p_object.data.isnull().any())
                
                lis_for_target = data_p_object.features
                target = st.selectbox("Select the target Variable: ",list(lis_for_target))
                if target is not None:
                    try:
                        data_p_object.input =  data_p_object.data.drop(target,axis=1)
                        data_p_object.output = data_p_object.data[target]
                        data_p_object.features.remove(target)
                        data_p_object.output_name = target
                    except:
                        st.error("Please enter a valid target variable")
                

                tst_percent = st.text_input("Enter the testing percentage","0.2")
                st.caption("an optimal value will be between 20-30%")
                data_p_object.split(float(tst_percent),validation_percent=0.1,rs=42)
                st.write("Dimensions of the training data: ",data_p_object.train_features.shape)
                st.write("Dimensions of the testing data: ",data_p_object.test_features.shape)
                st.write("Dimensions of the validation data: ",data_p_object.val_features.shape)
                st.caption("Countvectorizer: Helps transform textual data to matrix type, use this if you have random text in your dataset, make sure the data of that column is non categorical")
                st.caption("Encode Categoricalcolumns: If your datat has categorical columns, this function will convert it to numbers that represent categories this helps reduce errors while training the model.")
                st.caption("SMOT: Synthetic Minority Oversampling Technique is a statistical technique for increasing the number of cases in your dataset in a balanced way.")
                wut_to_do = st.multiselect("What all preprocessing you want on your data set?",("Apply countvectorizer","Encode categorical columns","Apply SMOTE","Standardize","Normalize"))

                if "Apply countvectorizer" in wut_to_do:
                    to_vectorize = st.text_input("Enter the column which you want to vectorize: ")
                    if to_vectorize is not None:
                        try:
                            from sklearn.feature_extraction.text import CountVectorizer
                            data_p_object.objects['Countvec_'+to_vectorize] = CountVectorizer()#instantiating the count vectorizer
                            data_p_object.data[to_vectorize] = data_p_object.objects['Countvec_'+to_vectorize].fit_transform(data_p_object.data[to_vectorize])
                        except:
                            st.error("Please enter a valid column name")


                if "Encode categorical columns" in wut_to_do:
                    
                    from sklearn.preprocessing import LabelEncoder
                    label_encoder_objects ={}
                    # st.write(data_p_object.features)
                    # st.write(data_p_object.data['diagnosis'].dtype)
                    edit_col = [i for i in dataframe.columns if data_p_object.data[i].dtype == np.object_]
                    edit_columns = edit_col
                    # st.write(edit_col)
                    if edit_columns is None:
                        st.error("The Dataset does'nt contain any categorical columns")
                    for col in edit_columns:
                        label_object = LabelEncoder()
                        data_p_object.data[col]=label_object.fit_transform(data_p_object.data[col])
                        label_encoder_objects[col+"_encoder_object"] = label_object
                    data_p_object.objects['Label_Encoder'] = label_encoder_objects
                    
                if "Apply SMOTE" in wut_to_do:
                    from imblearn.over_sampling import SMOTE
                    smote_object = SMOTE()
                    data_p_object.train_features,data_p_object.train_target = smote_object.fit_resample(data_p_object.train_features,data_p_object.train_target)
                    data_p_object.objects['Smote object'] = smote_object

                if "Standardize" in wut_to_do:
                    from pandas import DataFrame as df
                    scale_object  = data_p_object.objects['Standard scaler']()
                    # st.write( data_p_object.shape())
                    data_p_object.train_features=df(data = scale_object.fit_transform(data_p_object.train_features),columns = data_p_object.features)
                    data_p_object.test_features = df(data = scale_object.fit_transform(data_p_object.test_features),columns = data_p_object.features)
                    data_p_object.val_features = df(data = scale_object.fit_transform(data_p_object.val_features),columns = data_p_object.features)
                
                if "Normalize" in wut_to_do:
                    from pandas import DataFrame as df
                    scale_object  = data_p_object.objects['Min Max Scalar']()
                    data_p_object.train_features=df(data = scale_object.fit_transform(data_p_object.train_features),columns = data_p_object.features)
                    data_p_object.test_features = df(data = scale_object.fit_transform(data_p_object.test_features),columns = data_p_object.features)
                    data_p_object.val_features = df(data = scale_object.fit_transform(data_p_object.val_features),columns = data_p_object.features)
                
                st.write(":partying_face: WOILAAA you have finished the first step i.e., preprocessing :partying_face:")

    #########################################################################################################################################################################
    #REGRESSION
    with tab2:
        if(st.checkbox("Do you want to perform regression on your data set?")):
                # lis_for_target = data_p_object.features
                # target = st.selectbox("Select the target Variable : ",list(lis_for_target))
                # if target is not None:
                #     try:
                #         data_p_object.input =  data_p_object.data.drop(target,axis=1)
                #         data_p_object.output = data_p_object.data[target]
                #         data_p_object.features.remove(target)
                #         data_p_object.output_name = target
                #     except:
                #         st.error("Please enter a valid target variable")
                

                # tst_percent = st.text_input("Enter the testing percentage","0.2")
                # st.caption("an optimal value will be between 20-30%")
                # data_p_object.split(float(tst_percent),validation_percent=0.1,rs=42)
                # st.write("Dimensions of the training data: ",data_p_object.train_features.shape)
                # st.write("Dimensions of the testing data: ",data_p_object.test_features.shape)
                # st.write("Dimensions of the validation data: ",data_p_object.val_features.shape)
                model_obj = MachineLearningRegression(data_p_object)
                for model,dic in model_obj.model_evaluvation_dict.items():
                    model_obj.model_evaluvation_dict[model]['model_object'].fit(model_obj.train_features,model_obj.train_target)
                    model_obj.trained_models.append(model_obj.model_evaluvation_dict[model]['model_object'])
                    model_obj.model_prediction[model] = model_obj.model_evaluvation_dict[model]['model_object'].predict(model_obj.test_features)
                
                from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,mean_absolute_percentage_error
                metrics = {'r2 score':r2_score,'MAE':mean_absolute_error,'MSE':mean_squared_error,'MAPE':mean_absolute_percentage_error}
                for model,dic in model_obj.model_evaluvation_dict.items():
                    for metric,obj in metrics.items():
                        model_obj.model_evaluvation_dict[model][metric] = obj(model_obj.model_prediction[model],model_obj.test_target)
                        if model_obj.model_evaluvation_dict[model]['r2 score']>model_obj.best_r2_score:
                            model_obj.best_model = {'Name':model,
                                            'r2 score':model_obj.model_evaluvation_dict[model]['r2 score'],
                                                'model_obj':model_obj.model_evaluvation_dict[model]['model_object']}
                            model_obj.best_r2_score = model_obj.model_evaluvation_dict[model]['r2 score']
            
                # model_obj.fit()
                # model_obj.Score_test_dataset()
                if type(model_obj.prediction_array)==np.ndarray:
                    model_obj.model_evaluvation_dict['prediction']=model_obj.best_model['model_obj'].predict(np.array([model_obj.prediction_array]))[0]
                for model in model_obj.model_evaluvation_dict:
                    if model!='prediction':
                        del model_obj.model_evaluvation_dict[model]['model_object']
                # model_obj.best_model_object = model_obj.best_model['model_obj']
                # del model_obj.best_model['model_obj']
                model_obj.model_evaluvation_dict['best model'] = model_obj.best_model
                
                review = model_obj.model_evaluvation_dict
                # st.write(review) 
                
                st.caption("Models that we have:")
                st.caption("LinearRegression")
                st.caption("Ridge")
                st.caption("Lasso")
                st.caption("DecisionTreeRegressor")
                st.caption("RandomForestRegressor")
                st.caption("KNeighborsRegressor")
                choise = st.selectbox("What regression model do you want :", ("All","Select a few","Best"))
                if choise == "All":
                    analyse = st.selectbox("How would you like to analyse the results? ",("Numerically","Graphically"))
                    if analyse == "Numerically":
                        to_check = review['best model']['Name']
                        # st.text(to_check)
                        for model_name,metrik in review.items():
                            st.header(model_name)
                            for param,value in metrik.items():
                                if model_name == "best model":
                                    if param == "r2 score":
                                        st.write(param,":",round(value,3))
                                    else:
                                        st.write(param,":",value)
                                else:
                                    st.write(param,":",round(value,3))
                                

                    if analyse == "Graphically":
                        r2_score_list,MSE_list,MAE_list,MAPE_list={},{},{},{}
                        for model in review:
                            if model!='best model':
                                r2_score_list[model] = round(review[model]['r2 score'],2)
                                MSE_list[model] = round(review[model]['MSE'],2)
                                MAE_list[model] = round(review[model]['MAE'],2)
                                MAPE_list[model] = round(review[model]['MAPE'],2)
                        # group_labels = ['LinearRegression', 'Ridge', 'Lasso',"DecisionTreeRegressor","RandomForestRegressor","KNeighborsRegressor"]
                        fig,axes = plt.subplots(nrows=2,ncols=2,figsize=(35,45))
                        # Figure 
                        a = sns.barplot(x=list(r2_score_list.keys()),y=list(r2_score_list.values()),ax=axes[0][0])
                        a.set_title("R2 SCORE PLOT\n",fontsize=20)
                        a.set_xticklabels(a.get_xticklabels(),rotation=90,fontsize=15)
                        for size in a.containers:
                            a.bar_label(size,fontsize=15)

                        # Figure 
                        b = sns.barplot(x=list(MSE_list.keys()),y=list(MSE_list.values()),ax=axes[0][1])
                        b.set_title("MSE PLOT\n",fontsize=20)
                        b.set_xticklabels(a.get_xticklabels(),rotation=90,fontsize=15)
                        for size in b.containers:
                            b.bar_label(size,fontsize=15)

                        # Figure 
                        c = sns.barplot(x=list(MAE_list.keys()),y=list(MAE_list.values()),ax=axes[1][0])
                        c.set_title("MAE PLOT\n",fontsize=20)
                        c.set_xticklabels(a.get_xticklabels(),rotation=90,fontsize=15)
                        for size in c.containers:
                            c.bar_label(size,fontsize=15)

                        # Figure 
                        d = sns.barplot(x=list(MAPE_list.keys()),y=list(MAPE_list.values()),ax=axes[1][1])
                        d.set_title("MAPE PLOT\n",fontsize=20)
                        d.set_xticklabels(a.get_xticklabels(),rotation=90,fontsize=15)
                        for size in d.containers:
                            d.bar_label(size,fontsize=15)
                        st.write(fig)

                if choise=="Best":
                    best_model = review['best model']['Name']
                    analyse = st.selectbox("How would you like to analyse the results? ",("Numerically","Graphically"))
                    if analyse == "Numerically":
                        for model_name,metrik in review.items():
                            if model_name==best_model:
                                st.header(model_name)
                                for param,value in metrik.items():
                                    st.write(param,":",round(value,3))
                    
                    if analyse == "Graphically":
                        r2_score_list,MSE_list,MAE_list,MAPE_list={},{},{},{}
                        for model in review:
                            if model == best_model :
                                r2_score_list[model] = round(review[model]['r2 score'],3)
                                MSE_list[model] = round(review[model]['MSE'],3)
                                MAE_list[model] = round(review[model]['MAE'],3)
                                MAPE_list[model] = round(review[model]['MAPE'],3)
                        # group_labels = ['LinearRegression', 'Ridge', 'Lasso',"DecisionTreeRegressor","RandomForestRegressor","KNeighborsRegressor"]
                        fig,axes = plt.subplots(nrows=2,ncols=2,figsize=(35,45))
                        # Figure 
                        a = sns.barplot(x=list(r2_score_list.keys()),y=list(r2_score_list.values()),ax=axes[0][0])
                        a.set_title("R2 SCORE PLOT\n",fontsize=20)
                        a.set_xticklabels(a.get_xticklabels(),rotation=90,fontsize=15)
                        for size in a.containers:
                            a.bar_label(size,fontsize=15)

                        # Figure 
                        b = sns.barplot(x=list(MSE_list.keys()),y=list(MSE_list.values()),ax=axes[0][1])
                        b.set_title("MSE PLOT\n",fontsize=20)
                        b.set_xticklabels(a.get_xticklabels(),rotation=90,fontsize=15)
                        for size in b.containers:
                            b.bar_label(size,fontsize=15)

                        # Figure 
                        c = sns.barplot(x=list(MAE_list.keys()),y=list(MAE_list.values()),ax=axes[1][0])
                        c.set_title("MAE PLOT\n",fontsize=20)
                        c.set_xticklabels(a.get_xticklabels(),rotation=90,fontsize=15)
                        for size in c.containers:
                            c.bar_label(size,fontsize=15)

                        # Figure 
                        d = sns.barplot(x=list(MAPE_list.keys()),y=list(MAPE_list.values()),ax=axes[1][1])
                        d.set_title("MAPE PLOT\n",fontsize=20)
                        d.set_xticklabels(a.get_xticklabels(),rotation=90,fontsize=15)
                        for size in d.containers:
                            d.bar_label(size,fontsize=15)
                        st.write(fig)
                if choise=="Select a few":
                    best_model = review['best model']['Name']
                    model_selected = st.multiselect("Select the models you want to work with:",('LinearRegression', 'Ridge', 'Lasso','DecisionTreeRegressor','RandomForestRegressor','KNeighborsRegressor'))
                    analyse = st.selectbox("How would you like to analyse the results? ",("Numerically","Graphically"))
                    if best_model not in model_selected:
                        st.warning("YOUR SELECTED MODELS DOES'NT INCLUDE THE BEST MODEL, YOU MIGHT WANT TO CONSIDER IT")
                    
                    if analyse == "Numerically":
                        for model_name,metrik in review.items():
                            if model_name in model_selected:
                                st.header(model_name)
                                for param,value in metrik.items():
                                    st.write(param,":",round(value,3))
                    
                    if analyse == "Graphically":
                        r2_score_list,MSE_list,MAE_list,MAPE_list={},{},{},{}
                        for model in review:
                            if model in model_selected :
                                r2_score_list[model] = round(review[model]['r2 score'],3)
                                MSE_list[model] = round(review[model]['MSE'],3)
                                MAE_list[model] = round(review[model]['MAE'],3)
                                MAPE_list[model] = round(review[model]['MAPE'],3)
                        # group_labels = ['LinearRegression', 'Ridge', 'Lasso',"DecisionTreeRegressor","RandomForestRegressor","KNeighborsRegressor"]
                        fig,axes = plt.subplots(nrows=2,ncols=2,figsize=(35,45))
                        # Figure 
                        a = sns.barplot(x=list(r2_score_list.keys()),y=list(r2_score_list.values()),ax=axes[0][0])
                        a.set_title("R2 SCORE PLOT\n",fontsize=20)
                        a.set_xticklabels(a.get_xticklabels(),rotation=90,fontsize=15)
                        for size in a.containers:
                            a.bar_label(size,fontsize=15)

                        # Figure 
                        b = sns.barplot(x=list(MSE_list.keys()),y=list(MSE_list.values()),ax=axes[0][1])
                        b.set_title("MSE PLOT\n",fontsize=20)
                        b.set_xticklabels(a.get_xticklabels(),rotation=90,fontsize=15)
                        for size in b.containers:
                            b.bar_label(size,fontsize=15)

                        # Figure 
                        c = sns.barplot(x=list(MAE_list.keys()),y=list(MAE_list.values()),ax=axes[1][0])
                        c.set_title("MAE PLOT\n",fontsize=20)
                        c.set_xticklabels(a.get_xticklabels(),rotation=90,fontsize=15)
                        for size in c.containers:
                            c.bar_label(size,fontsize=15)

                        # Figure 
                        d = sns.barplot(x=list(MAPE_list.keys()),y=list(MAPE_list.values()),ax=axes[1][1])
                        d.set_title("MAPE PLOT\n",fontsize=20)
                        d.set_xticklabels(a.get_xticklabels(),rotation=90,fontsize=15)
                        for size in d.containers:
                            d.bar_label(size,fontsize=15)
                        st.write(fig)
                validate = st.selectbox("Do you want to validate your model using validation data?",("No","Yes"))
                if validate == "Yes":
                    score_list={}
                    for i in model_obj.trained_models:
                        score_list[str(i).replace("()","")] = round(i.score(data_p_object.val_features,data_p_object.val_target),3)
                    fig,axes = plt.subplots(figsize=(15,8))
                    a = sns.barplot(x = list(score_list.keys()),y = list(score_list.values()),ax=axes)
                    for i in a.containers:
                        a.bar_label(i)
                    st.write(fig)

                for_kfold = st.selectbox("Do you want to perform k-Fold to improve model accuracy?",("No","Yes"))
                st.info("This can change the best model to some other model")
                st.caption("This will take some time to process, since its itterative process")
                if for_kfold == "Yes":
                    fig,axes = plt.subplots(figsize=(14,2))
                    splits = st.number_input("Enter a value of splits for k fold: ",10)
                    kf = KFold(n_splits=splits)
                    # from sklearn.linear_model import LinearRegression,Lasso
                    models = [LinearRegression(),Lasso(),Ridge(),DecisionTreeRegressor(),RandomForestRegressor(),KNeighborsRegressor()]
                    score_list = {'LinearRegression':[],'Lasso':[], 'Ridge':[],"DecisionTreeRegressor":[],"RandomForestRegressor":[],"KNeighborsRegressor":[]}
                    for model in models:
                        for train,test in kf.split(data_p_object.train_features,data_p_object.train_target):
                            model.fit(data_p_object.train_features.iloc[train],data_p_object.train_target.iloc[train])
                            score_list[str(model).replace("()","")].append(round(model.score(data_p_object.train_features.iloc[train],data_p_object.train_target.iloc[train]),3))
                        score_list[str(model).replace("()","")] = max(score_list[str(model).replace("()","")])*100
                    a = sns.barplot(y = list(score_list.keys()),x = list(score_list.values()),ax=axes)
                    for i in a.containers:
                        a.bar_label(i)
                    st.write(fig)
                
                knn = st.selectbox("Want to know the best K value for KNN algo?",("No","Yes"))
                if knn == "Yes":
                    min_k = int(st.number_input("Enter min value of K",1))
                    if (min_k) is 0:
                        st.warning("k cannot be 0")
                    st.caption("The max value of k will be ten more than min value of k.")
                    model_list = {str(i):KNeighborsRegressor(i) for i in range(min_k,min_k+11)}
                    accuracy_list={}
                    for model_name,model in model_list.items():
                        model.fit(data_p_object.train_features,data_p_object.train_target)
                        accuracy_list[model_name] = round(model.score(data_p_object.val_features,data_p_object.val_target),3)
                    fig,axes = plt.subplots(figsize=(10,5))
                    a = sns.barplot(x=list(accuracy_list.keys()),y=list(accuracy_list.values()),ax=axes)
                    for i in a.containers:
                        a.bar_label(i)
                    st.write(fig)
                    print("The best value of K is",list(accuracy_list.keys())[list(accuracy_list.values()).index(max(list(accuracy_list.values())))])
                predict = st.selectbox("Do you want to predict any sample's output?",("No","Yes"))
                if predict == "Yes":
                    # predict_models = st.multiselect("Select the regression models you want to predict with: ",('LinearRegression', 'Ridge', 'Lasso',"DecisionTreeRegressor","RandomForestRegressor","KNeighborsRegressor"))
                    # st.write(data_p_object.features)
                    categories = data_p_object.features
                    prediction_array={}
                    feature = []
                    for value in categories:
                        # st.write("enter value for ",value)
                        # num = st.text_input('exter the value of')
                        prediction_array[value]=None
                    for k,v in prediction_array.items():
                        prediction_array[k]=st.number_input(k,v)
                    # st.write(prediction_array)
                    for k,v in prediction_array.items():
                        feature.append(v)
                    # if type(self.prediction_array)==np.ndarray:
                    model_obj.model_evaluvation_dict['prediction']=model_obj.best_model['model_obj'].predict(np.array([feature]))[0]
                    st.write(int(model_obj.model_evaluvation_dict['prediction']))
                    outval = int(model_obj.model_evaluvation_dict['prediction'])
                    # ans = lablevalu[outval]
                    # st.write(lablevalu)
                    # for name, age in lablevalu.items():
                    #     if str(age) == str(outval):
                    #         final = name
                    # st.write("The final outcome: ",final)

            #"""i have to add code for taking input of columns with object """
    #########################################################################################################################################################################
    #CLASSIFICATION
    with tab3:
        if(st.checkbox("Do you want to perform Classification on your data set?")):
            # lis_for_target = data_p_object.features
            # target = st.selectbox("Select the target Variable : ",list(lis_for_target))
            # if target is not None:
            #     try:
            #         data_p_object.input =  data_p_object.data.drop(target,axis=1)
            #         data_p_object.output = data_p_object.data[target]
            #         data_p_object.features.remove(target)
            #         data_p_object.output_name = target
            #     except:
            #         st.error("Please enter a valid target variable")
            st.caption("Models that we have:")
            st.caption("LogisticRegression")
            st.caption("GaussianNB")
            st.caption("SVC")
            st.caption("DecisionTreeClassifier")
            st.caption("RandomForestClassifier")
            st.caption("KNeighborsClassifier")
            
            model_obj_classification = MacineLearningClassification(data_p_object)
            #fitting model
            def draw_plot(review,str):
                fig,axes = plt.subplots(nrows=1,ncols=1,figsize=(18,10))
                accuracy = {model:review[model][str] if model!='best model' else None for model in review}
                del accuracy['best model']
                a = sns.barplot(y=list(accuracy.keys()),x=list(accuracy.values()),palette="ocean_r",ax=axes)
                return a
            def draw_heatmap(review):
                fig,axes = plt.subplots(nrows=3,ncols=2,figsize=(15,20))
                confusion_matrixes = {model:review[model]['confusion matrix for test data'] if model!='best model' else None for model in review}
                del confusion_matrixes['best model']
                row = col = 0
                for model in confusion_matrixes:
                    if col in [2,4]:
                        col=0
                        row+=1
                        a = sns.heatmap(confusion_matrixes[model],ax = axes[row][col],cmap='ocean',annot=True,cbar=False,annot_kws={'fontsize':15})
                        a.set_title("\n"+model+"\n")
                        col+=1
                    else:
                        a = sns.heatmap(confusion_matrixes[model],ax = axes[row][col],cmap='ocean',annot=True,cbar=False,annot_kws={'fontsize':15})
                        col+=1
                        a.set_title("\n"+model+"\n")
                return fig
            for model,dic in model_obj_classification.model_evaluvation_dict.items():
                model_obj_classification.model_evaluvation_dict[model]['model_object'].fit(model_obj_classification.train_features,model_obj_classification.train_target)
                model_obj_classification.trained_models.append(model_obj_classification.model_evaluvation_dict[model]['model_object'])
                model_obj_classification.model_prediction[model] = model_obj_classification.model_evaluvation_dict[model]['model_object'].predict(model_obj_classification.test_features)
            #noting test data values:
            for model,dic in model_obj_classification.model_evaluvation_dict.items():
                model_obj_classification.model_evaluvation_dict[model]['score on test data'] = model_obj_classification.model_evaluvation_dict[model]['model_object'].score(model_obj_classification.test_features,model_obj_classification.test_target)*100
                if (model_obj_classification.model_evaluvation_dict[model]['score on test data']>model_obj_classification.best_accuracy):
                    model_obj_classification.best_model = {'Model_obj':model_obj_classification.model_evaluvation_dict[model]['model_object'],
                                    'Name':model,
                                    'Accuracy':model_obj_classification.model_evaluvation_dict[model]['score on test data']}
                    model_obj_classification.best_accuracy = model_obj_classification.model_evaluvation_dict[model]['score on test data']
            #evaluate function
            # self.fit()
            # self.Score_test_data()
            model_obj_classification.create_confusion_matrix()
            model_obj_classification.create_f1_precision_recall()
            if type(model_obj_classification.prediction_array)==np.ndarray:
                model_obj_classification.model_evaluvation_dict['prediction']=model_obj_classification.best_model['Model_obj'].predict(np.array([model_obj_classification.prediction_array]))[0]
            for model in model_obj_classification.model_evaluvation_dict:
                if model!='prediction':
                    del model_obj_classification.model_evaluvation_dict[model]['model_object']
            # model_obj_classification.best_model_object = model_obj_classification.best_model['Model_obj']
            # del model_obj_classification.best_model['Model_obj']
            model_obj_classification.model_evaluvation_dict['best model'] = model_obj_classification.best_model
            review_classification =  model_obj_classification.model_evaluvation_dict
            
            choose = st.selectbox("What regression model do you want:", ("All","Select some","Best"))
            if choose == "All":
                analyse = st.selectbox("How would you like to analyse the results?",("Numerically","Graphically"))
                if analyse == "Numerically":
                    to_check = review_classification['best model']['Name']
                    # st.text(to_check)
                    for model_name,metrik in review_classification.items():
                        st.header(model_name)
                        for param,value in metrik.items():
                            if param != "confusion matrix for test data":
                                if model_name == "best model":
                                    if param == "Accuracy":
                                        st.write(param,":",round(value,3))
                                    else:
                                        st.write(param,":",value)
                                else:
                                    st.write(param,":",round(value,3))
                if analyse == "Graphically":
                    def draw_plot(review,str):
                        figs,axes = plt.subplots(nrows=1,ncols=1,figsize=(18,10))
                        accuracy = {model:round(review[model][str],2) if model!='best model' else None for model in review}
                        del accuracy['best model']
                        a = sns.barplot(y=list(accuracy.keys()),x=list(accuracy.values()),palette="ocean_r",ax=axes)
                        a.set_title('\nAccuracy\n')
                        for s in a.containers:
                            a.bar_label(s,fontsize=10)
                        st.write(figs)
                    a = draw_plot(review_classification,'score on test data')    
            corel = st.selectbox("Do you want to visualise the confusion matrix?",("No","Yes"))
            if corel == "Yes":
               figure =  draw_heatmap(review_classification)
               st.write(figure)
            kfold = st.selectbox("Do you want to perform k-Fold to improve model accuracy? ",("No","Yes"))
            st.info("This can change the best model to some other model")
            if kfold == "Yes":
                fig,axes = plt.subplots(figsize=(14,2))
                splits = st.number_input("Enter a value of splits for k fold:",10)
                kf = KFold(n_splits=splits)
                # from sklearn.linear_model import LinearRegression,Lasso
                models = [LogisticRegression(),DecisionTreeClassifier(),RandomForestClassifier(),GaussianNB(),KNeighborsClassifier(),SVC()]
                score_list = {'LogisticRegression':[],'DecisionTreeClassifier':[], 'RandomForestClassifier':[],"GaussianNB":[],"KNeighborsClassifier":[],"SVC":[]}
                for model in models:
                    for train,test in kf.split(data_p_object.train_features,data_p_object.train_target):
                        model.fit(data_p_object.train_features.iloc[train],data_p_object.train_target.iloc[train])
                        score_list[str(model).replace("()","")].append(round(model.score(data_p_object.train_features.iloc[train],data_p_object.train_target.iloc[train]),3))
                    score_list[str(model).replace("()","")] = max(score_list[str(model).replace("()","")])*100
                a = sns.barplot(y = list(score_list.keys()),x = list(score_list.values()),ax=axes)
                for i in a.containers:
                    a.bar_label(i)
                st.write(fig)
            st.caption("Have a confussion selecting the best hyperameters? Try using GridSearchCV... ")
            grid = st.selectbox("Do you want to know the best parameters for your models using GridSearchCV? ",("No","Yes"))
            if grid == "Yes":
                C = st.text_input("Enter the different values of C for SVC seperated by a comma',' :")
                C = C.split(',')
                
                kernel = st.multiselect("Select the kernels for SVC: ",("rbf","linear",'sigmoid'))
                criterion = st.multiselect("Select the function to measure the quality of a split for Decision Tree",('gini', 'entropy'))
                depth = st.text_input("Enter the different maximum depths of the tree seperated by a comma ',' for Decision Tree:")
                depth = depth.split(',')
                
                trees = st.text_input("Enter the different number of trees for Random Forest regression seperated by a comma',' :")
                trees = trees.split(',')
                
                k = st.text_input("Enter the different number of neighbors for KNN seperated by a comma',' :")
                k = k.split(',')
                for i in range(0, len(trees)):
                    trees[i] = int(trees[i])
                for i in range(0, len(depth)):
                    depth[i] = int(depth[i])
                for i in range(0, len(C)):
                    C[i] = int(C[i])
                for i in range(0, len(k)):
                    k[i] = int(k[i])
                k_fld = int(st.number_input("Enter the number of folds: "))
                from sklearn.model_selection import GridSearchCV
                hypr_parm = {

                'svc':{'model': SVC(gamma='auto'),
                        'parameters':{
                            'C':C,
                            'kernel':kernel
                        }
                },
                'Decission_tree':{'model':DecisionTreeClassifier(),
                        'parameters':{
                            'criterion': criterion,
                            'max_depth':depth
                        }},
                'Random_forest':{'model':RandomForestClassifier(),
                        'parameters':{
                            'n_estimators':trees
                        }},
                    
                'KNN':{'model':KNeighborsClassifier(),
                        'parameters':{
                            'n_neighbors':k
                        }}

                }
                score=[]
                for mod,feature in hypr_parm.items():
                    bestmodl = GridSearchCV(feature['model'],feature['parameters'],cv=k_fld,return_train_score=False)
                    bestmodl.fit(data_p_object.train_features, data_p_object.train_target)
                    score.append({
                        'model':mod,
                        'best_params':bestmodl.best_params_,
                        'best_score':bestmodl.best_score_
                    })

                valuess= pd.DataFrame(score, columns = ['model','best_params','best_score'])
                st.write(valuess)
            classify = st.selectbox("Do you want to predict any sample's output? ",("No","Yes"))
            if classify == "Yes":
                type_input = st.selectbox("Do you want to enter a set of testcases as csv file or give individual values?",("Give a CSV file","Individual value"))
                # predict_models = st.multiselect("Select the regression models you want to predict with: ",('LinearRegression', 'Ridge', 'Lasso',"DecisionTreeRegressor","RandomForestRegressor","KNeighborsRegressor"))
                # st.write(data_p_object.features)
                if type_input == "Give a CSV file":
                #    st.write(data_p_object.features)
                   uploaded_file = st.file_uploader("Enter the dataset you want to test: ")
                   st.caption("Ensure that the column names for test set are the same as the training set.")
                   if uploaded_file is not None:
                        # Can be used wherever a "file-like" object is accepted:
                        dataframe = pd.read_csv(uploaded_file)
                        # st.write(dataframe.head())
                        data_to_predict = dataframe[list(data_p_object.features)]
                        # st.write(data_to_predict.shape[0])
                        for i in range (0,data_to_predict.shape[0]):
                            st.write(data_to_predict.iloc[i])


                if type_input == "Individual value":
                    categories = data_p_object.features
                    st.write(data_p_object.features)
                    predict_array={}
                    feature = []
                    for value in categories:
                        # st.write("enter value for ",value)
                        # num = st.text_input('exter the value of')
                        predict_array[value]=None
                    for key,val in predict_array.items():
                        predict_array[key]=st.number_input(key,val)
                    # st.write(prediction_array)
                    for k,v in predict_array.items():
                        feature.append(v)
                    # st.write(feature)
                    # if type(self.prediction_array)==np.ndarray:
                    model_obj_classification.model_evaluvation_dict['prediction']=model_obj_classification.best_model['Model_obj'].predict_proba(np.array([feature]))[0]
                    # model_obj_classification.model_evaluvation_dict['prediction']=model_obj_classification.best_model['Model_obj'].predict(np.array([feature]))[0]
                    class_pred = model_obj_classification.model_evaluvation_dict['prediction']
                    # labels = LE.
                    st.write(pd.DataFrame(class_pred))
                    # st.write(int(model_obj.model_evaluvation_dict['prediction']))

        # else:
        #   st.info("Please upload a dataset to continue")