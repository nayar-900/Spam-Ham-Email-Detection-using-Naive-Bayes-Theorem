#  Spam Email Detection using Naive Bayes Theorem

This project demonstrates a **Spam vs Ham email classifier** built using the **Naive Bayes Theorem**. It combines both **Python (GUI + ML)** and **Excel** implementations to provide a comprehensive understanding of text classification using probabilities.

---

## Project Highlights

-  **Naive Bayes Classifier** using Scikit-learn
-  **Spam/Ham Data Visualization**: Bar Chart, Pie Chart, Confusion Matrix
-  **Real-time Classification GUI** built with Tkinter
-  **Excel Implementation** using formulas for probability-based prediction
-  **Preprocessing** using NLTK (tokenization + stopword removal)
-  **Custom CSV File Support** (`Message`, `Category` format)

---

## Tools and Libraries

- Python 3.x
- Pandas
- Scikit-learn
- NLTK
- Matplotlib
- Tkinter

---

## Folder Structure

```

 Spam-Detection-NaiveBayes/
│
├── spam\_classifier\_gui.py      # GUI-based Python application
├── sample\_dataset.csv          # Example dataset (Message, Category)
├── spam\_excel\_implementation.xlsx   # Excel-based Naive Bayes classifier
├── README.md

````

---

##  How to Run the Python Project

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/spam-email-naive-bayes.git
   cd spam-email-naive-bayes
````

2. Install dependencies:

   ```bash
   pip install pandas scikit-learn matplotlib nltk
   ```

3. Run the app:

   ```bash
   python spam_classifier_gui.py
   ```

4. Load your `CSV` file and begin classification.

---

## CSV Format

Your dataset should have two columns:

| Message                     | Category |
| --------------------------- | -------- |
| "You’ve won a free iPhone!" | spam     |
| "Let’s meet tomorrow."      | ham      |

---

## Visualization Features

* **Bar Chart:** Count of spam vs ham
* **Pie Chart:** Percentage distribution
* **Confusion Matrix:** Model performance
* **Text Classification:** Result popup for any message
<img width="607" height="659" alt="image" src="https://github.com/user-attachments/assets/d7060cb7-e26b-41ee-b7fa-33ee9b63f5e7" />


---

## Excel-based Implementation

* Manual calculation of:

  * Prior probabilities
  * Word frequencies
  * Conditional probabilities
* Predicts category using Bayes formula in formulas
* Useful for **academic presentations** and non-coders

---

## Use Case

This project is ideal for:

* Learning **text classification** using Naive Bayes
* Presenting a dual-mode (Python + Excel) solution in educational settings
* Understanding basic **Natural Language Processing (NLP)** tasks

---

## Screenshots

<img width="1893" height="850" alt="image" src="https://github.com/user-attachments/assets/7c11b549-c1e0-4fa0-8722-9217675457b0" />
<img width="897" height="644" alt="image" src="https://github.com/user-attachments/assets/72d31027-184e-49d5-8268-4b8cc79a9b78" />
<img width="1345" height="885" alt="image" src="https://github.com/user-attachments/assets/e7a5894d-0652-47b7-b631-31753b6a5c39" />


---

## Credits

Project by \[Your Name - Rayan Badar]
Special thanks to \[Co-partner Sumaira Majeed] for support in development and documentation.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
