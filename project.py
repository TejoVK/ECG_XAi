import streamlit as st
import pandas as pd
from io import StringIO
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
import imblearn
import streamlit_authenticator as stauth

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
        for col in edit_columns:
            label_object = LabelEncoder()
            self.data[col]=label_object.fit_transform(self.data[col])
            label_encoder_objects[col+"_encoder_object"] = label_object
        self.objects['Label_Encoder'] = label_encoder_objects
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





uploaded_file = st.file_uploader("Enter the dataset you want to work on: ")
if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe.head(5))

    if(st.checkbox("Do you want to preprocess your data?")):
        data_p_object = DataPreprocessing(dataframe)
        # data_p_object = pd.DataFrame(data_p_object)
        st.write("The correlation matrix for your dataset:")
        corr = data_p_object.data.corr()
        mask = np.ones_like(corr,dtype=np.bool_)
        mask[np.tril_indices_from(mask)]=False
        mask[np.diag_indices_from(mask)]=True
        fig,axes = plt.subplots(nrows=1,ncols=1,figsize=(8,8))
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
            fig,axes = plt.subplots(nrows=1,ncols=1,figsize=(8,8))
            sns.heatmap(corr,annot=True,mask=mask,ax=axes,annot_kws={'fontsize':8},cbar=False,cmap='ocean_r')
            axes.set_title('\n Correlation Matrix for the dataset\n')
            st.write(fig)



        null_val=st.selectbox("How would you like to handle you null data: ", ("drop them","replace with mean"))
        if(null_val=="drop them"):
            data_p_object.data.dropna(axis=0,inplace=True)
        else:
            data_p_object.data=data_p_object.data.apply(lambda x:x.fillna(x.mean()))
        

        st.table(data_p_object.data.isnull().any())
        

        target = st.text_input("Enter the target Variable")
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
            edit_columns = data_p_object.get_object_column()
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
        # st.caption("take a sneekpeek of your data: ")
        # data_p_object = float(data_p_object)
        # data_p_object = pd.DataFrame(data_p_object)
        # st.write(type(data_p_object))
else:
    st.info("Please upload a dataset to continue")

#########################################################################################################################################################################
#REGRESSION
model_obj = MachineLearningRegression(data_p_object)
review = model_obj.evaluvate()

