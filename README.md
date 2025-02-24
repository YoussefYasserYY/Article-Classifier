# Topic Classifier

## Overview
The **Topic Classifier** is a Streamlit-based application that allows users to classify text into two distinct topics using a Naive Bayes classification approach. The model can be trained on either automatically generated datasets or user-uploaded text files. 

## Features
- **Load Data**: Users can either generate dataset labels or upload their own labeled text files.
- **Train Model**: Trains a Naive Bayes classifier on the provided dataset.
- **Test Model**: Allows users to input text and classify it into one of the predefined topics.
- **User-friendly UI**: Interactive widgets for easy data loading, training, and classification.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repository/topic-classifier.git
   ```
2. Navigate to the project directory:
   ```bash
   cd topic-classifier
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
2. Navigate through the sidebar to either Load Data, Train, or Test the model.
3. Follow the instructions on each page to generate or upload data, train the model, and classify text.

## Project Structure
```
topic-classifier/
│── app.py                   # Main Streamlit application
│── text_class.py            # Supporting functions for text processing and classification
│── requirements.txt         # Dependencies
│── trained.pickle           # Serialized trained model (generated after training)
│── data/                    # Directory containing user-uploaded text files
│── README.md                # Project documentation
```

## Dependencies
- Streamlit
- Pickle
- OS
- Glob
- Python Standard Libraries

## How It Works
1. **Data Loading**: Users provide topic names, and the app either generates titles based on predefined sources or allows manual data uploads.
2. **Training**: A Naive Bayes model is trained using word frequencies from the dataset.
3. **Classification**: The trained model predicts the most probable class for a given text input.

## Future Improvements
- Extend support for multiple classes.
- Implement an advanced text preprocessing pipeline.
- Improve the UI with enhanced visualization for classification results.
