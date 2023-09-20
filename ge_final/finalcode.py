
import scipy.io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.transforms as mtransforms
import pandas as pd
#add a logo of amrita near the title. done
#name the models in the select box. done
#add the names of the channels of pre-view graphs. done
#add cumulative matter below heat maps. done
# in an additional tab add model.summary.
#rename the chbot as: curenet ai chatbot done



st.markdown(
    """
    <style>
    /* Style the heading with an alternative color */
    .custom-heading {
        font-size: 36px;
        color: #0074D9; /* Light blue color for text */
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }

    /* Style the subtitle */
    .custom-subtitle {
        font-size: 18px;
        color: #BBB; /* Light gray color for text */
        text-align: center;
        margin-top: 10px;
    }
    </style>
    """
    , unsafe_allow_html=True
)

# Render the styled heading and subtitle
logo_image = 'logos/amritalogo.png'
st.image(logo_image, use_column_width=True)

st.markdown(
    """
    <h1 class="custom-heading">CureNet AI</h1>
    <p class="custom-subtitle">AI Unveiled: Deciphering ECG Abnormalities Through Visual Analysis</p>
    """, unsafe_allow_html=True
)

# Upload the .mat file
tab1, tab2 = st.tabs(["Xai", "Model summary"])

# from streamlit_option_menu import option_menu

# selected = option_menu(
#     menu_title = None,
#     options = ["XAI", "CureNet AI Chatbot"],
#     menu_icon = "cast",
#     default_index = 0,
#     orientation = "horizontal"
# )

with tab1:
    uploaded_file = st.file_uploader("Upload the ECG signal file of single patient: ")

    def heatmap_plot(attributions , signal_data , prediction ):
        fig, ax = plt.subplots(figsize=(10, 6))
        tt = np.array(signal_data).reshape(2500,)
    
        # Assuming only one attribution method
        #[abs(i) for i in attt]
        heatmap_data = attributions
        signal = tt
    
        ax.plot(tt, color="black")
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')
        ax.grid(False)
    
        num_of_colors = 40
    
        min_attr = min(heatmap_data)
        max_attr = max(heatmap_data)
        states_min = list(np.linspace(min_attr, 0, num=int(num_of_colors/2), endpoint=False))
        states_max = list(np.linspace(0, max_attr, num=int(num_of_colors/2)))
        states_int = states_min + states_max
    
        # Generate unique states for binning
        states_int_unique = np.unique(states_int)
    
        # Bin the heatmap data
        heatmap_data_series = pd.Series(heatmap_data)
        state = pd.cut(
            heatmap_data_series, bins=states_int_unique, labels=range(len(states_int_unique) - 1)
        )
    
        # Get the colormap
        cmap = plt.get_cmap('RdBu_r')
    
        # Create a transform for the fill_between operation
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    
        # Iterate over colormap colors and draw the heatmap graph
        j = 0
        for i, color in enumerate(cmap(np.linspace(0, 1, (num_of_colors-1)+2)[1:-1])):
            ax.fill_between(np.arange(0, 2500), 0, 1, where=state == i,
                            facecolor=color, transform=trans, linewidth=6,
                            edgecolor=color, alpha=0.6)
            ax.minorticks_on()
            # ax.xaxis.set_ticks(np.arange(0, len(signal), 50))
            ax.grid(which='minor')
            ax.grid(which='major')
    
        ax.set_ylabel("amplitude")
        st.pyplot(fig)
        # st.write(prediction)
    
    
    
    
    def gradients_method(xs,model):
        X_input = tf.convert_to_tensor(xs)  # Replace with your input data
    
    
        # Use tf.GradientTape to compute the gradients
        with tf.GradientTape() as tape:
            tape.watch(X_input)
            prediction = model(X_input)
        gradients = tape.gradient(prediction, X_input)
        return gradients
    
    
    
    
    class IntegratedGradients():
    
        def __init__(self, xs, model,steps=100, baseline=None):
            self.steps = steps
            self.xs = xs
            self.model = model
            self.baseline = np.zeros_like(xs, dtype=np.float32)
    
            
            #super(IntegratedGradients, self).__init__(T, X, session, keras_learning_phase)
    
        def run(self,  ys=None, batch_size=None):
            self.has_multiple_inputs = type(self.xs) is list or type(self.xs) is tuple
            
            
    
            gradient = None
            for alpha in list(np.linspace(1. / self.steps, 1.0, self.steps)):
                xs_mod = [b + (x - b) * alpha for x, b in zip(self.xs, self.baseline)] if self.has_multiple_inputs \
                    else self.baseline + (self.xs - self.baseline) * alpha
                attr = gradients_method(xs_mod,self.model)
                
                if gradient is None: gradient = attr
                else: gradient = [g + a for g, a in zip(gradient, attr)]
    
            results = [g * (x - b) / self.steps for g, x, b in zip(
                gradient,
                self.xs if self.has_multiple_inputs else [self.xs],
                self.baseline if self.has_multiple_inputs else [self.baseline])]
    
            return results[0][0] if not self.has_multiple_inputs else results
    
    
    
    def gradients_method1(xs,model,eps):
        X_input = tf.convert_to_tensor(xs)  # Replace with your input data
    
    
        # Use tf.GradientTape to compute the gradients
        with tf.GradientTape() as tape:
            tape.watch(X_input)
            prediction = model(X_input)
        grad = tape.gradient(prediction, X_input)
        print(grad.shape)
        print(prediction)
        print(eps)
    
        #er = grad * prediction[0][0] / (X_input + eps *tf.where(X_input >= 0, tf.ones_like(X_input), -1 * tf.ones_like(X_input)))
        return grad
    
    
    class EpsilonLRP():
        eps = None
    
        def __init__(self, xs, model,epsilon=1e-4):
            assert epsilon > 0.0, 'LRP epsilon must be greater than zero'
            global eps
            self.eps = epsilon
            eps = epsilon
            self.model = model
            self.xs = xs
            self.has_multiple_inputs = type(self.xs) is list or type(self.xs) is tuple
    
        def get_symbolic_attribution(self):
            attributions = [g * x for g, x in zip(
                # gradients_method1(self.xs, self.model,self.eps),
                gradients_method1(self.xs, self.model,self.eps),
                self.xs if self.has_multiple_inputs else [self.xs])]
            return np.array(attributions[0][0] * eps)
    
    
    @tf.function
    def deep_lift(model, signal_data, target_class_index):
        signal_data = tf.convert_to_tensor(signal_data, dtype=tf.float32)
        signal_data = tf.expand_dims(signal_data, axis=0)
        print(signal_data.shape)
        
        reference_data = tf.ones_like(signal_data)  # Define your reference data
        print(reference_data.shape)
        
        with tf.GradientTape() as tape:
            tape.watch(signal_data)
            model_out = model(signal_data)
            reference_out = model(reference_data)
            
        
            delta_out = model_out - reference_out
            #print(np.array(diff))
        attribution = tape.gradient(delta_out, signal_data)
        
        delta_in = signal_data - reference_data
        instant_grad = (0.5 * (reference_data + signal_data))
        return delta_in * instant_grad * attribution
    
    
 
    if uploaded_file is not None:
    
        
        st.write("File uploaded")
        mat_data = scipy.io.loadmat(uploaded_file)
        variable_names = mat_data.keys()
        st.write("Variable names in the .mat file:", variable_names)
        selected_variable = st.selectbox("Select a variable to display:", variable_names)

        if selected_variable in variable_names:
            # Convert the data to a Pandas DataFrame
            df = pd.DataFrame(mat_data[selected_variable])
            channel_names = ["I","II", "III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
            eeg_data = loadmat(uploaded_file)
            eeg_data = np.asarray(eeg_data['val'], dtype=np.float64)
            eeg_signal = eeg_data[:,:2500]
            channels = eeg_signal.shape[0]
            st.write(' ')
            st.write(' ')
            on = st.checkbox("Do you want to pre-view the graphs?")
            st.write(' ')
            st.write(' ')
            if on:
            
                sampling_rate = 100  # Sample rate in Hz
                duration = eeg_signal.shape[1] / sampling_rate
                time = np.arange(0, duration, 1 / sampling_rate)
                channels = eeg_signal.shape[0]

                # Determine the number of rows and columns for the subplots
                num_rows = channels  # One plot per row
                num_cols = 1  # Fixed: One signal per row

                # Calculate the subplot height and overall figure height
                subplot_height = 300  # Adjust the height as needed
                fig_height = subplot_height * num_rows

                # Create a subplot figure with subplots for each channel
                fig = make_subplots(rows=num_rows, cols=num_cols, shared_xaxes=True, shared_yaxes=True,
                                    vertical_spacing=0.02, row_heights=[subplot_height] * num_rows)

                # Set a default color for all plots
                default_color = '#3776d7'  # Use the color '3776d7'

                for i in range(channels):
                    # Calculate the row index for the current subplot
                    row_idx = i + 1

                    # Create a mask for the time interval you want to highlight
                    mask = np.logical_and(time >= 30, time <= 40)

                    # Add a trace for the EEG signal
                    fig.add_trace(go.Scatter(x=time, y=eeg_signal[i], mode='lines', name=f'Channel {channel_names[i]}', showlegend=False,
                                             line=dict(color=default_color)),
                                  row=row_idx, col=num_cols)

                    # Add a trace for the highlighted interval in red
                    fig.add_trace(go.Scatter(x=time[mask], y=eeg_signal[i][mask], mode='markers', line=dict(color='red'), showlegend=False),
                                  row=row_idx, col=num_cols)

                    # Customize subplot titles
                    fig.update_xaxes(title_text='Time (s)', row=row_idx, col=num_cols)
                    fig.update_yaxes(title_text=channel_names[i], row=row_idx, col=num_cols)
                fig.update_layout(
                    title_text='ECG Signal with Highlighted Intervals',
                    showlegend=False,
                    height=fig_height,
                    margin=dict(t=20, b=20, l=20, r=20),  # Reduce spacing with margin adjustments
                )
                fig.update_xaxes(tickvals=list(range(0, len(time), 50)))
                st.plotly_chart(fig)
    
            models = ['FCN- ConvNet','Model- ResidualConvNet', 'ResNet']
            model_name = st.selectbox('Select the model you want to use: ', models)
            if 'FCN- ConvNet' == model_name:
                model_path = 'models/best_model4_fcn.h5'
                fcn_model = tf.keras.models.load_model(model_path)
            if 'Model- ResidualConvNet' == model_name:
                model_path = 'models/best_model.h5'
                fcn_model = tf.keras.models.load_model(model_path)
            if 'ResNet' == model_name:
                model_path = 'models\best_model2_res.h5'
                fcn_model = tf.keras.models.load_model(model_path)
    
    
    
            y_pred = fcn_model.predict(eeg_signal.reshape(1,12,2500))
            threshold = 0.3
            predicted_classes = tf.argmax(y_pred, axis=1)
            predicted_classes = np.array(predicted_classes)
            y_binary = tf.where(y_pred > threshold,1,0)
            Abnormalities = ['Myocardial Infarction','Left Axis Deviation']
            pred_abno = Abnormalities[predicted_classes[0]].rstrip()
            y_pred = y_pred.tolist()
            # st.write(y_pred)
            rounded_y_pred = np.round(y_pred[0], 3)
            bar_g_data = pd.DataFrame({'Abnormality': Abnormalities, 'Predict_proba': rounded_y_pred})
            
            fig = px.bar(bar_g_data, x = 'Abnormality', y='Predict_proba', title='Prediction Probability of different Abnormalities')
            st.plotly_chart(fig)
            
            y_pred_score = y_pred[0][predicted_classes[0]]
            st.info(f"The Xai prediction of the abnormality is: {pred_abno}")
            st.info(f"The prediction score: {round(y_pred_score,3)}")
            
            #write the name of the abnormality and the predicted score
    
            st.write('The ECG signal has', channels, 'channels.')
            # channel = [i for i in range(channels)]
            channel = {channel_names[i]:i for i in range(12)}
            Xai = ['Integrated Gradient', 'LRP', 'Deep Lift']
    
            XAI_techniques = st.multiselect('Select the Xai techniques you want :',Xai)
            toview = st.multiselect('Select the channels you want to view: ', channel_names)
    
            my_dict = {i: channel_names[i] for i in range(12)}
            if 'Integrated Gradient' in XAI_techniques:
                ds = IntegratedGradients(eeg_signal.reshape(1,12,2500) , fcn_model)
                ax = ds.run()
                ax = np.array(ax)
                for i in toview:
                    st.write("Integrated Gradient: ", "Channel: ", i)
                    i = channel[i]
                    heatmap_plot(ax[i],eeg_signal[i],y_binary)
            if 'LRP' in XAI_techniques:
                sdf = np.array(eeg_signal,dtype=np.float32).reshape(1,12,2500)
                ds = EpsilonLRP(sdf,fcn_model)
                ax1 = ds.get_symbolic_attribution() 
                # print("EpsilonLRP","   ---- " , ax1.shape)
                lrp = np.array(ax1)
                for i in toview:
                    st.write("LRP: ", "Channel: ", i)
                    i = channel[i]
                    heatmap_plot(lrp[i],eeg_signal[i],y_binary)
            if 'Deep Lift' in XAI_techniques:
                attribution = deep_lift(fcn_model, eeg_signal.astype(np.float32), 0)
                # print("DeepLIFT shape:", attribution.shape)
                for i in toview:
                    st.write("Deep Lift: ", "Channel: ", i)
                    i = channel[i]
                    dfg = attribution[0][i].numpy().reshape(2500,) 
                    heatmap_plot(dfg, eeg_signal[i], y_binary)
            st.caption("Accentuating the critical traits acquired by the model for abnormality classification, these are showcased through overlaid red-colored strips, offering a clear and interpretable representation of the AI-driven diagnosis.")
            # st.markdown("""<a id="top"></a>""", unsafe_allow_html=True)


# st.components.v1.html(js)    
        else:
            st.write("Variable not found in the .mat file")



with tab2:

    # ("do you want to know about the models used?")
    modelll = ["RESNET MODEL", "ConvNet Model", "Residual Convolutional Model"]
    ansss = st.multiselect("Select the models you want to know about: ", modelll)
    
    if "RESNET MODEL" in ansss:
            st.header("RESNET Model Description:")
            st.write(""" Our "RESNET MODEL" is an advanced neural network tailored for the precise classification of Electrocardiogram (ECG) signals. It excels at deciphering complex ECG patterns, making it a vital tool for diagnosing heart-related conditions.""")
            st.header("Key Features:")
            
            st.write("""Residual Blocks: Utilizing residual blocks, our model efficiently learns intricate ECG patterns, ensuring robust performance.
        Batch Normalization: Batch normalization stabilizes training and accelerates convergence by normalizing inputs after each convolutional layer.
        ReLU Activation: Employing ReLU activation, the model maintains information flow, allowing for efficient learning.
        Dropout: Dropout layers mitigate overfitting, enhancing model generalization.
        Global Average Pooling: Global Average Pooling optimizes model efficiency by reducing spatial information while preserving effectiveness.
        Softmax Output: The output layer uses softmax activation, making it ideal for multi-class classification tasks.
        Binary Crossentropy Loss: The model employs binary cross-entropy loss, suitable for multi-label classification of ECG signals.
        Optimization with Adam: Adam optimizer with a learning rate of 0.001 ensures efficient model training.
        Comprehensive Metrics: Evaluation metrics include Binary Accuracy, Recall, Precision, and Area Under the ROC Curve (AUC).""")
            st.header("Training and Saving:")
            st.write("""Our model is trained for 100 epochs with a batch size of 32, employing a class weight strategy to handle data imbalances. After training, both the model and its training history are saved for future reference.
        ResNet""")
            # model1_image = 'logos/output5.png'
            # st.image(model1_image, use_column_width=True)
            history_resnet = np.load('npy/res_historyp.npy', allow_pickle=True).item()
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(history_resnet['Recall'], label='Training Recall')
            ax.plot(history_resnet['val_Recall'], label='Validation Recall')
            ax.plot(history_resnet['Precision'], label='Training Precision')
            ax.plot(history_resnet['val_Precision'], label='Validation Precision')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss')
            ax.legend()
            st.pyplot(fig)




    if "ConvNet Model" in ansss:
        st.header("ConvNet Model Description:")
        st.write("""
    Our ConvNet Model is a specialized neural network designed for the precise classification of Electrocardiogram (ECG) signals. It excels at capturing complex patterns within ECG data, making it a valuable tool for diagnosing heart-related conditions.""")
        st.header("Key Features:")
        
        st.write("""Convolutional Layers: The model employs multiple convolutional layers to learn hierarchical features from ECG signals effectively.
    Batch Normalization: Batch normalization layers are integrated after each convolutional layer to enhance training stability and convergence speed.
    ReLU Activation: ReLU activation functions are used throughout the network, maintaining strong information flow during training.
    Global Average Pooling: Global Average Pooling condenses spatial information in the final convolutional layer, enabling more efficient and computationally lightweight processing.
    Sigmoid Output: The output layer utilizes a sigmoid activation function, making it suitable for multi-label classification tasks.
    Binary Crossentropy Loss: The model uses binary cross-entropy loss, a well-suited choice for multi-label ECG signal classification.
    Optimization with SGD: Stochastic Gradient Descent (SGD) optimizer with a learning rate of 0.001 is employed for efficient model training.
    Comprehensive Metrics: Model evaluation includes Binary Accuracy, Recall, Precision, and Area Under the ROC Curve (AUC), providing a thorough assessment of classification performance.""")
        st.header("Training and Saving:")
        st.write("""Our FCN model is trained with 100 epochs, a batch size of 32, and class weight consideration to address data imbalances. After training, the model is saved for future use.
    """)
        # model2_image = 'logos/6.png'
        # st.image(model2_image, use_column_width=True)
        history_convonet = np.load('npy/fcn_historyp.npy', allow_pickle=True).item()
        fig_conv, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history_convonet['Recall'], label='Training Recall')
        ax.plot(history_convonet['val_Recall'], label='Validation Recall')
        ax.plot(history_convonet['Precision'], label='Training Precision')
        ax.plot(history_convonet['val_Precision'], label='Validation Precision')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        st.pyplot(fig_conv)




    if "Residual Convolutional Model" in ansss:
        st.header("Residual Convolutional Model Description:")
        st.write("""The Residual Convolutional Model" is a specialized deep neural network designed for the classification of Electrocardiogram (ECG) signals. This model efficiently processes ECG data and categorizes it into different classes, aiding in the diagnosis of various heart-related conditions.""")
        st.header("Key Features:")
        
        st.write("""LeakyReLU Activation: The model employs the Leaky Rectified Linear Unit (LeakyReLU) activation function with a small gradient for negative values. This helps prevent the vanishing gradient problem and enhances the training of deep networks.
    Residual Blocks: The model is structured around a series of residual blocks. Each residual block consists of convolutional layers followed by average pooling with a window size of 2. This architecture is inspired by ResNet and enables the efficient extraction of complex features from ECG signals.
    Global Average Pooling: After the residual blocks, Global Average Pooling is applied to the feature maps. This technique reduces the spatial dimensions and parameters while retaining essential information.
    Flattened Features: The output from the previous step is flattened to prepare it for fully connected layers.
    Dropout Regularization: Dropout layers are integrated into the fully connected layers (fc1 and fc2) to prevent overfitting by randomly deactivating a portion of neurons during training.
    Softmax Output: The output layer employs a softmax activation function, making it suitable for multi-class classification tasks. The model provides probability distributions over different ECG classes.
    Binary Crossentropy Loss: The model uses binary cross-entropy loss, which is suitable for multi-label classification problems where multiple classes may be present in a single ECG signal.
    Optimization and Metrics: The model is trained using the Adam optimizer with a learning rate of 0.001. It is evaluated using metrics such as Binary Accuracy, Recall, Precision, and Area Under the Receiver Operating Characteristic Curve (AUC).""")
        st.header("Usage:")
        st.write("""The ECG Classification Model is a valuable tool for healthcare professionals and researchers. It can be seamlessly integrated into web applications, clinical systems, or research platforms to automate the classification of ECG signals, making the diagnosis of heart-related conditions more efficient and accurate.
    Please feel free to use this model in your application or system. If you require further customization or have specific requirements, do not hesitate to reach out, and we will be happy to assist you.
    """)
        # model3_image = 'logos/7.png'
        # st.image(model3_image, use_column_width=True)
        history_Residual = np.load('npy/historyp.npy', allow_pickle=True).item()
        fig_residual, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history_Residual['Recall'], label='Training Recall')
        ax.plot(history_Residual['val_Recall'], label='Validation Recall')
        ax.plot(history_Residual['Precision'], label='Training Precision')
        ax.plot(history_Residual['val_Precision'], label='Validation Precision')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        st.pyplot(fig_residual)


        

# with tab3:
#     import pickle
#     import streamlit as st
#     from streamlit_chat import message

#     import os , random 
#     from PIL import Image
    
#     with open("chain_falcon_7b_xai.bin", "rb") as f:
#         # Deserialize the object from the file.
#         chain = pickle.load(f)

#     # Close the file.
#     f.close()
    
    
#     st.title("CureNet AI ChatBot 🧑🏽‍⚕️")
    
#     def conversation_chat(query):
#         result = chain({"question": query, "chat_history": st.session_state['history']})
#         st.session_state['history'].append((query, result["answer"]))
#         return result["answer"]
    
#     def initialize_session_state():
        
        
#         if 'history' not in st.session_state:
#             st.session_state['history'] = []
    
#         if 'generated' not in st.session_state:
#             st.session_state['generated'] = ["Hello! Ask me about anything 🤗"]
#             st.session_state['images_path'] = []
#             st.session_state['dict_path'] = {}
#             st.session_state["count1"] = 0
    
#         if 'past' not in st.session_state:
#             st.session_state['past'] =  ["Hey! 👋"] 
#         if "image" not in st.session_state:
#             st.session_state["image"] = []
    
#     def display_chat_history():
#         reply_container = st.container()
#         container = st.container()
#         path = "https://drive.google.com/drive/folders/1RGOMaIcWv3wB0yn0B3xQcGJi8r5p4F-5?usp=sharing"
#         ll = []
        
#         question = st.session_state['past']
        
        
#         for dir in os.listdir(path):
#             ll.append(dir)
        

#         with container:
#             with st.form(key='my_form', clear_on_submit=True):
#                 user_input = st.text_input("Question:", placeholder="Your Query", key='input')
#                 submit_button = st.form_submit_button(label='Send')
              
                

#             # if submit_button and user_input:
#             #     if any(keyword in str(question[count]).lower() for keyword in ["plot", "show", "picture"]):
#             #         for keyword in ["plot", "show", "picture"]:
#             #             if keyword in str(question[count]).lower():
#             #                 user_input1 = user_input.lower().replace(keyword, "")
#             #                 print(user_input1)
#             #                 user_input1 = user_input.strip()
#             if submit_button and user_input:
#                 images_path = []
#                 user_input1 = user_input.replace("show", "")
#                 st.write(user_input1)
#                 output = conversation_chat(user_input1)
#                 st.session_state['past'].append(user_input)
#                 st.session_state['generated'].append(output)
#                 question = st.session_state['past']
                


                
#                 if any(keyword in str(user_input).lower() for keyword in ["plot", "show", "picture"]):
#                             for i in ll:
                                
#                                 if i.lower() in str(user_input).lower():
#                                     directory_path = path + i
#                                     image_files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
#                                     random_image_file = random.choice(image_files)
                                    

#                                     # Load and display the random image
#                                     random_image_path = os.path.join(directory_path, random_image_file)
#                                     images_path.append(random_image_path)
#                             st.session_state['dict_path'][len(question) - 1] = images_path
                                    


                
                            
                            
                        
             
                
                
            
                
#             st.session_state["count1"]  += 1
            
                
                
                

#         if st.session_state['generated']:
            
#             with reply_container:
#                 count = 0
                
                
               
                
#                 for k in range(len(st.session_state['generated'])):

#                     if isinstance(st.session_state["generated"][k], str):
                        
#                         message(st.session_state["past"][k], is_user=True, key=str(k) + '_user', avatar_style="thumbs")
#                         message(st.session_state["generated"][k], key=str(k), avatar_style="fun-emoji")
                    
#                     # st.write(k,count,len(question))

#                     if any(keyword in question[count].lower() for keyword in ["plot", "show", "picture"]):
#                                 dict_path = st.session_state["dict_path"]
#                                 # st.write(dict_path.keys())
                                
#                                 j = dict_path[count]
#                                 for kl in j:
                                
                    
                    
#                                     img = Image.open(kl)
                                    

#                                     st.image(img,use_column_width=True)
#                     count += 1
                                
                                
                                
                                

    
#     # Initialize session state
#     initialize_session_state()
#     # Display chat history
#     display_chat_history()