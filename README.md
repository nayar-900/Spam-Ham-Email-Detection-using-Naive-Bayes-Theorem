# Spam Email Detection using Naive Bayes Theorem
This project demonstrates a Spam vs Ham email classifier built using the Naive Bayes Theorem. It combines both Python (GUI + ML) and Excel implementations to provide a comprehensive understanding of text classification using probabilities.
## Project Highlights
- Naive Bayes Classifier using Scikit-learn
- Spam/Ham Data Visualization: Bar Chart, Pie Chart, Confusion Matrix
- Real-time Classification GUI built with Tkinter
- Excel Implementation using formulas for probability-based prediction
- Preprocessing using NLTK (tokenization + stopword removal)
- Custom CSV File Support (Message, Category format)
## Tools and Libraries
- Python 3.x
- Pandas
- Scikit-learn
- NLTK
- Matplotlib
- Tkinter
## Folder Structure
Spam-Detection-NaiveBayes/

├── spam_classifier_gui.py      # GUI-based Python application

├── sample_dataset.csv          # Example dataset (Message, Category)

├── spam_excel_implementation.xlsx   # Excel-based Naive Bayes classifier

├── README.md

## How to Run the Python Project
### 1. Clone this repository:
git clone https://github.com/your-username/spam-email-naive-bayes.git
cd spam-email-naive-bayes
### 2. Install dependencies:
pip install pandas scikit-learn matplotlib nltk
### 3. Run the app:
python spam_classifier_gui.py
### 4. Load your CSV file and begin classification.

## CSV Format
Your dataset should have two columns:
### Message	Category
“You’ve won a free iPhone!”	              spam

“Let’s meet tomorrow.”	                   ham
## Visualization Features
* Bar Chart: Count of spam vs ham
* Pie Chart: Percentage distribution
* Confusion Matrix: Model performance
* Text Classification: Result pop-up for any message
## Excel-based Implementation
* Manual calculation of:
  * Prior probabilities
  * Word frequencies
  * Conditional probabilities
* Predicts category using the Bayes formula in formulas
* Useful for academic presentations and non-coders
## Use Case
* Learning text classification using Naive Bayes
* Presenting a dual-mode (Python + Excel) solution in educational settings
* Understanding basic Natural Language Processing (NLP) tasks
## Credits
Project by *Rayan Badar* 

Special thanks to *Sumaira Majeed* for support in development and documentation.

