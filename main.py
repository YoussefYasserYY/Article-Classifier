import streamlit as st
import pickle
import os
from Text_class import loading_data , class_prob , preprocessing,conditional_prob,get_titles_from_folders,process_uploaded_files
st.set_page_config(
    page_title="Article Classifier",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

st.title("Article Classifier📑")
st.image('0_6nPN6naiH7lApcvg.png',width=550)

choice = st.sidebar.radio('',["Load" , 'Train' ,"Test"],horizontal=True)


if choice == 'Load':

    st.header('load and Train Your Naive Bayes Model')
    c = st.radio('Choose your Data',['Generate Data','Upload Data'])
    st.subheader('Enter Topic Names')
    col1,col2 = st.columns(2) 
    # Text input widget
    if c == 'Generate Data':
        label1 = col1.text_input("",placeholder='Class 1')
        label2= col2.text_input("",placeholder='Class 2')
        if st.button('Generate',type='primary'):
            if label1 and label2:
                    try:
                        with st.status('Loading Data...'):
                            domains =[label1,label2] 
                            Titles = loading_data(domains)
                            DOC_LABELS = []
                            for title in Titles:
                                if  label1.casefold() in title.casefold() :
                                    DOC_LABELS.append(label1)
                                if  label2.casefold() in title.casefold() :
                                    DOC_LABELS.append(label2)
                            CLASS_PROBS = class_prob(DOC_LABELS)
                            st.session_state['CLASS_PROBS'] = CLASS_PROBS
                            st.session_state['DOC_LABELS'] = DOC_LABELS
                    except:
                        st.warning('Error Occurred Please try again')
    else:
        st.subheader("Upload your Data")
        label1 = col1.text_input("",placeholder='Class 1')
        label2= col2.text_input("",placeholder='Class 2')
        cl1,cl2 = st.columns(2)
        data_folder = os.path.join(os.getcwd(), 'data')
        os.makedirs(data_folder, exist_ok=True)
        
        if label1:
            folder_name1 = label1.strip()  # Remove leading/trailing spaces
            folder_path1 = os.path.join(data_folder, folder_name1)
            if not os.path.exists(folder_path1):
                os.makedirs(folder_path1)
                cl1.success(f"Folder '{folder_name1}' created successfully!")
            uploaded_files1 = cl1.file_uploader("Upload text files (Ctrl+Click to select multiple)", type=["txt"], accept_multiple_files=True,key=('11'))
            if uploaded_files1 is not None:
                process_uploaded_files(folder_path1, uploaded_files1)
        if label2:
            folder_name2 = label2.strip()
            folder_path2 = os.path.join(data_folder, folder_name2)
            if not os.path.exists(folder_path2):
                os.makedirs(folder_path2)
                cl2.success(f"Folder '{folder_name2}' created successfully!")
            uploaded_files2 = cl2.file_uploader("Upload text files (Ctrl+Click to select multiple)", type=["txt"], accept_multiple_files=True,key=('22'))
            if uploaded_files2 is not None:
                process_uploaded_files(folder_path2, uploaded_files2)
            
            if st.button('Load Data',type='primary'):
                with st.status('Loading Data...'):
                    domains =[label1,label2]
                    Titles = get_titles_from_folders(label1,label2)
                    # Titles = loading_data(domains)
                    DOC_LABELS = []
                    # with open("titles.txt", "w") as file:
                    for title in Titles:
                            # file.write(title + "\n")
                        if  label1.casefold() in title.casefold() :
                            DOC_LABELS.append(label1)
                        if  label2.casefold() in title.casefold() :
                            DOC_LABELS.append(label2)
                    CLASS_PROBS = class_prob(DOC_LABELS)
                    st.session_state['CLASS_PROBS'] = CLASS_PROBS
                    st.session_state['DOC_LABELS'] = DOC_LABELS
    st.session_state.c = c
    st.session_state.label1= label1
    st.session_state.label2= label2

                    


elif choice=='Train':
    if st.button('Train',type='primary'):
        if st.session_state.c == 'Generate Data':
            # try:
                with st.status('Training...'):
                    import glob
                    class1 = glob.glob(f"data/{st.session_state.label1}/*.txt")
                    class2 = glob.glob(f"data/{st.session_state.label2}/*.txt")
                    train_docs = glob.glob(f"Train_*.txt")
                    folder1 = f'data/{st.session_state.label1}'
                    folder2 = f"data/{st.session_state.label2}"
                    train_docs = glob.glob(f"{folder1}/*.txt") + glob.glob(f"{folder2}/*.txt")
                    train_docs =preprocessing(train_docs)
                    class1 =preprocessing(class1)
                    class2 =preprocessing(class2) 

                    prob = {}
                    class_label1 = st.session_state.label1
                    class_label2 = st.session_state.label2
                    for i in train_docs: 
                        words = i.split()
                        for j in words:
                            prob[(class_label1,j)] = conditional_prob(class_label1, j, train_docs,class1,class2,st.session_state.label1,st.session_state.label2)
                            prob[(class_label2,j)] = conditional_prob(class_label2, j, train_docs,class1,class2,st.session_state.label1,st.session_state.label2)
                
                    with open('trained.pickle', 'wb') as handle:
                        pickle.dump(prob, handle, protocol=pickle.HIGHEST_PROTOCOL)
                st.balloons()
                st.success('Training Completed!')
            # except Exception as ex:
                # st.error('Sorry, Please try again')
        else:
            st.header('Training')
            with st.status('Training...'):
                import glob
                class1 = glob.glob(f"data/{st.session_state.label1}/*.txt")
                class2 = glob.glob(f"data/{st.session_state.label2}/*.txt")
                train_docs = glob.glob(f"Train_*.txt")
                folder1 = f'data/{st.session_state.label1}'
                folder2 = f"data/{st.session_state.label2}"
                train_docs = glob.glob(f"{folder1}/*.txt") + glob.glob(f"{folder2}/*.txt")
                train_docs =preprocessing(train_docs)
                class1 =preprocessing(class1)
                class2 =preprocessing(class2) 

                prob = {}
                class_label1 = st.session_state.label1
                class_label2 = st.session_state.label2
                for i in train_docs: 
                    words = i.split()
                    for j in words:
                        prob[(class_label1,j)] = conditional_prob(class_label1, j, train_docs,class1,class2,st.session_state.label1,st.session_state.label2)
                        prob[(class_label2,j)] = conditional_prob(class_label2, j, train_docs,class1,class2,st.session_state.label1,st.session_state.label2)
            
                with open('trained.pickle', 'wb') as handle:
                    pickle.dump(prob, handle, protocol=pickle.HIGHEST_PROTOCOL)
            st.balloons()
            st.success('Training Completed!')
                        
            
            
        
        
        
else:
    label1 = st.session_state.label1 
    label2 = st.session_state.label2
    CLASS_PROBS = st.session_state['CLASS_PROBS'] 
    DOC_LABELS = st.session_state['DOC_LABELS'] 
    st.header('Test Your Model')
    text = st.text_area('Enter Text to Classify')
    if st.button('Classify',type = 'primary'):
        # try:
            CLASS_PROBS = st.session_state['CLASS_PROBS'] 
            DOC_LABELS = st.session_state['DOC_LABELS']  
            with open('trained.pickle', 'rb') as handle:
                Trained_model = pickle.load(handle)
            prob1 = 1
            prob2 = 1
            words = text.split()
            for word in words:
                if (label1, word) in Trained_model:
                    prob1 *= Trained_model[(label1, word)]
                if (label2, word) in Trained_model:
                    prob2 *= Trained_model[(label2, word)]
            print(CLASS_PROBS)
            prob1 *= CLASS_PROBS[label1]
            prob2 *= CLASS_PROBS[label2]
            score1 = int(prob1/(prob1+prob2)*100)
            score2 = int(prob2/(prob1+prob2)*100)
            # Add a check to handle the case when prob_data is zero
            if prob2 == 0:
                st.info(f"{label1}).")

            elif prob1 / prob2 > 1:
                st.info(f"{label1} Topic")
            else:
                st.info(f"{label2} Topic")
            col1,col2 = st.columns(2) 
            col1.subheader(f'{label1} {score1}%')
            col2.subheader(f'{label2} {score2}%')
            
        # except Exception as ex:
            # Handle exceptions at the domain level here
            # print(f"Error processing domain '{domain}': {ex}")
            # st.warning('Please Train Your Model First')

        # st.write(predict(text))

