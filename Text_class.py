import random
import wikipedia
import nltk
# from nltk.tokenize import word_tokenize
# from string import punctuation
import glob
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import os 
import streamlit as st
# List of domains for which you want to download text documents

# Function to get the most relevant page from a list of options
def get_most_relevant_page(options, domain):
    for option in options:
        if domain.lower() in option.lower():
            return option
    return options[0]


def get_titles_from_folders(folder_name1, folder_name2, base_folder='data'):
    folder_path1 = os.path.join(base_folder, folder_name1)
    folder_path2 = os.path.join(base_folder, folder_name2)

    # Use glob to find all .txt files in both folders
    files_in_folder1 = glob.glob(os.path.join(folder_path1, "*.txt"))
    files_in_folder2 = glob.glob(os.path.join(folder_path2, "*.txt"))

    # Extract titles from file paths
    titles1 = [os.path.splitext(os.path.basename(file))[0] for file in files_in_folder1]
    titles2 = [os.path.splitext(os.path.basename(file))[0] for file in files_in_folder2]

    # Concatenate titles from both folders into a single list
    all_titles = titles1 + titles2

    return all_titles

def loading_data(domains, num_documents_per_domain=5, base_folder='data'):
    titles = []
    
    for domain in domains:
        # Create a folder for each domain
        domain_folder = os.path.join(base_folder, domain)
        os.makedirs(domain_folder, exist_ok=True)
        
        wiki_titles = wikipedia.search(domain, results=6)
        if wiki_titles:
            # Shuffle the search results to ensure randomness
            random.shuffle(wiki_titles)
            for i, title in enumerate(wiki_titles[:num_documents_per_domain]):
                try:
                    page = wikipedia.WikipediaPage(title=title)
                    page_content = page.summary
                    if len(page_content.split()) >= 100:
                        # Determine whether it's a training or test document
                        if i < num_documents_per_domain - 1:
                            with open(os.path.join(domain_folder, f"Train_{title}.txt"), "w", encoding="utf-8") as temp_doc:
                                temp_doc.write(page_content)
                        else:
                            with open(os.path.join(domain_folder, f"Test_{title}.txt"), "w", encoding="utf-8") as temp_doc:
                                temp_doc.write(page_content)
                        titles.append(title)
                except wikipedia.exceptions.DisambiguationError as e:
                    # Handle disambiguation errors here (choose one option or skip)
                    print(f"Disambiguation error for '{domain}': {e.options}")
                    # You can choose one option randomly or manually select one
                    selected_option = random.choice(e.options)
                    print(f"Selected option: {selected_option}")
                    page = wikipedia.WikipediaPage(title=selected_option)
                    page_content = page.summary
                    if len(page_content.split()) >= 100:
                        with open(os.path.join(domain_folder, f"Train_{selected_option}.txt"), "w", encoding="utf-8") as temp_doc:
                            temp_doc.write(page_content)
                        titles.append(selected_option)
                except Exception as ex:
                    # Handle other exceptions here
                    print(f"Error processing '{title}': {ex}")
    return titles



def class_prob(doc_labels:list) -> dict:
    """
    Calculates the probability of each class given a list of document labels.
    
    Args:
    - doc_labels: a list of document labels
    
    Returns:
    - A dictionary where the keys are the unique labels in doc_labels and the values are the corresponding probabilities.
    """
    class_counts = {}
    for label in doc_labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    total_docs = len(doc_labels)
    class_probs = {label: count / total_docs for label, count in class_counts.items()}
    return class_probs

def tokenize(text):
    """
    Tokenizes text by splitting it into words.

    Args:
    - text: a string of text

    Returns:
    - A list of tokens.
    """
    # Define a list of characters to split the text on
    delimiters = [' ', '\n', '\t', '.', ',', ';', ':', '!', '?', '"', "'", '(', ')', '[', ']', '{', '}', '<', '>', '/', '\\', '|', '@', '#', '$', '%', '^', '&', '*', '-', '_', '+', '=', '`', '~']
    
    # Split text into words
    words = []
    word = ''
    for char in text:
        if char in delimiters:
            if word:
                words.append(word)
                word = ''
        else:
            word += char
    if word:
        words.append(word)

    return words

def preprocessing(documents:list) -> list:
    """
    Preprocesses a list of training documents by removing stop words, punctuation, lowercasing all text, 
    and applying stemming and lemmatization.

    Args:
    - documents: a list of training documents

    Returns:
    - A list of preprocessed documents.
    """
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    preprocessed_documents = []
    for file in documents:
        # Tokenize text into individual words
        with open(file,"r",encoding='utf-8') as doc:
            tokens = tokenize(doc.read())
            # Remove stop words and punctuation from the text
            filtered_tokens = [token.lower() for token in tokens if (token not in stop_words) and (token not in string.punctuation)]
            # Stem and lemmatize the filtered tokens
            stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
            lemmatized_tokens = [lemmatizer.lemmatize(token) for token in stemmed_tokens]
            # Join the lemmatized tokens back into a string
            preprocessed_doc = ' '.join(lemmatized_tokens)
            preprocessed_documents.append(preprocessed_doc)
    return preprocessed_documents


def count_words(preprocessed_doc):
    """Counts the number of words in a preprocessed document.

    Args:
      - preprocessed_doc: The preprocessed document.

    Returns:
      - The number of words in the preprocessed document.
    """
    
    # Split the document into words.
    words = preprocessed_doc.split()

    # Count the number of words.
    num_words = len(words)

    return num_words
def count_string_in_list(lst, string):
    count = 0
    words = lst.split()
    for i in words:
        if i == string:
            count += 1
    return count

def conditional_prob(class_label, word, train_docs,class1,class2,label1,label2):
    
    word_num = []
    if(class_label == label1):
        train = class1
        string = ' '.join(class1)
    elif (class_label == label2):
        train = class2
        string = ' '.join(class2)
        
    string1 = ' '.join(train_docs)
    word_counter=0
    for i in train:
        word_num.append(count_words(i))
        word_counter += count_string_in_list(i, word)
            
    k = count_words(string1)
    count_words_class= count_words(string)
    
    conditional_probability = (word_counter + 1) / (k + count_words_class)
    
    return conditional_probability



def process_uploaded_files(folder_path, uploaded_files):
    for uploaded_file in uploaded_files:
        file_contents = uploaded_file.read()
        file_name = uploaded_file.name
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "wb") as f:
            f.write(file_contents)
