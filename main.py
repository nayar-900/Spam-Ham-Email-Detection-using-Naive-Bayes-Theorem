import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')

# Global variables
data = None
vectorizer = None
model = None
X_train, X_test, y_train, y_test = None, None, None, None


# Text preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text.lower())  # Tokenize and lowercase
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    return " ".join(filtered_words)


# Function to load dataset
def load_dataset():
    global data
    try:
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return
        data = pd.read_csv(file_path)

        # Check if 'Category' and 'Message' columns exist
        if 'Category' not in data.columns or 'Message' not in data.columns:
            messagebox.showerror("Error", "Dataset must contain 'Category' and 'Message' columns!")
            data = None
            return

        messagebox.showinfo("Success", "Dataset loaded successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")


# Function to train the classifier
def train_classifier():
    global data, model, vectorizer, X_train, X_test, y_train, y_test
    if data is None:
        messagebox.showerror("Error", "Dataset not loaded!")
        return

    try:
        # Ensure all values in the 'Message' column are strings and handle missing values
        data['Message'] = data['Message'].fillna("").astype(str)

        # Remove rows with missing labels
        data = data[data['Category'].notna()]
        data['Category'] = data['Category'].astype(str)

        # Preprocess text
        data['processed_text'] = data['Message'].apply(preprocess_text)

        # Feature extraction
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(data['processed_text'])

        # Map labels: 'ham' -> 0, 'spam' -> 1
        valid_labels = {'ham': 0, 'spam': 1}
        data['Category'] = data['Category'].map(valid_labels)
        data = data[data['Category'].notna()]  # Remove invalid labels
        y = data['Category']

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Naive Bayes classifier
        model = MultinomialNB()
        model.fit(X_train, y_train)

        # Calculate accuracy on the test set
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)

        # Show success message
        messagebox.showinfo("Success", f"Classifier trained successfully!\nAccuracy: {acc:.2f}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to train classifier: {str(e)}")


# Function to classify a new message
def classify_message():
    global model, vectorizer
    if model is None or vectorizer is None:
        messagebox.showerror("Error", "Model not trained!")
        return

    try:
        message = input_message.get("1.0", tk.END).strip()
        if not message:
            messagebox.showerror("Error", "Message cannot be empty!")
            return

        processed_message = preprocess_text(message)
        features = vectorizer.transform([processed_message])
        prediction = model.predict(features)

        result = "Spam" if prediction[0] == 1 else "Ham"
        messagebox.showinfo("Result", f"The message is classified as: {result}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to classify message: {str(e)}")

# Function to display spam and ham count
from sklearn.metrics import confusion_matrix
import numpy as np

def show_counts():
    global data, model, X_test, y_test
    if data is None:
        messagebox.showerror("Error", "Dataset not loaded!")
        return

    if model is None or X_test is None or y_test is None:
        messagebox.showerror("Error", "Please train the model first!")
        return

    try:
        # Count spam and ham messages
        spam_count = sum(data['Category'] == 1)
        ham_count = sum(data['Category'] == 0)

        # Confusion matrix
        predictions = model.predict(X_test)
        cm = confusion_matrix(y_test, predictions)
        cm_labels = ['Ham', 'Spam']

        # Create plots
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Spam Detection Visualizations", fontsize=16)

        # Bar Chart
        axs[0].bar(['Ham', 'Spam'], [ham_count, spam_count], color=['blue', 'red'])
        axs[0].set_title("Spam vs Ham Count - Bar Chart")
        axs[0].set_xlabel("Category")
        axs[0].set_ylabel("Count")

        # Pie Chart
        axs[1].pie([ham_count, spam_count], labels=['Ham', 'Spam'], colors=['blue', 'red'],
                   autopct='%1.1f%%', startangle=140)
        axs[1].set_title("Spam vs Ham Distribution - Pie Chart")

        # Confusion Matrix
        axs[2].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axs[2].set_title("Confusion Matrix")
        tick_marks = np.arange(len(cm_labels))
        axs[2].set_xticks(tick_marks)
        axs[2].set_yticks(tick_marks)
        axs[2].set_xticklabels(cm_labels)
        axs[2].set_yticklabels(cm_labels)
        axs[2].set_xlabel('Predicted Label')
        axs[2].set_ylabel('True Label')

        # Annotate confusion matrix values
        for i in range(len(cm)):
            for j in range(len(cm[0])):
                axs[2].text(j, i, str(cm[i, j]),
                            ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black")

        plt.tight_layout()
        plt.show()

    except Exception as e:
        messagebox.showerror("Error", f"Failed to display charts: {str(e)}")

from sklearn.metrics import classification_report, confusion_matrix

def show_metrics():
    global model, X_test, y_test
    if model is None or X_test is None or y_test is None:
        messagebox.showerror("Error", "Train the model first!")
        return

    try:
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)
        report = classification_report(y_test, predictions, target_names=["Ham", "Spam"])

        # Create a new window
        metrics_window = tk.Toplevel(root)
        metrics_window.title("Model Evaluation Metrics")
        metrics_window.geometry("600x400")

        # Show accuracy
        lbl_acc = tk.Label(metrics_window, text=f"Accuracy: {acc:.2f}", font=('Arial', 12, 'bold'))
        lbl_acc.pack(pady=10)

        # Show confusion matrix
        lbl_cm = tk.Label(metrics_window, text="Confusion Matrix:", font=('Arial', 12, 'bold'))
        lbl_cm.pack()
        cm_text = tk.Text(metrics_window, height=4, width=40)
        cm_text.insert(tk.END, f"{cm}")
        cm_text.config(state='disabled')
        cm_text.pack()

        # Show classification report
        lbl_cr = tk.Label(metrics_window, text="Classification Report:", font=('Arial', 12, 'bold'))
        lbl_cr.pack(pady=5)
        report_text = tk.Text(metrics_window, height=10, width=70)
        report_text.insert(tk.END, report)
        report_text.config(state='disabled')
        report_text.pack()

    except Exception as e:
        messagebox.showerror("Error", f"Failed to generate metrics: {str(e)}")

def show_statistics():
    global data
    if data is None:
        messagebox.showerror("Error", "Dataset not loaded!")
        return

    try:
        stats_window = tk.Toplevel(root)
        stats_window.title("Descriptive Statistics")
        stats_window.geometry("900x600")

        stats_text = tk.Text(stats_window, wrap='word', font=('Courier New', 10))
        stats_text.pack(expand=True, fill='both', padx=10, pady=10)

        numeric_data = data.select_dtypes(include=['number'])

        if numeric_data.empty:
            stats_text.insert(tk.END, "No numeric columns available for statistical analysis ...\n")
        else:
            # Displaying everything in organized manner
            stats_text.insert(tk.END, "=" * 80 + "\n")
            stats_text.insert(tk.END, "Descriptive Statistics Summary\n")
            stats_text.insert(tk.END, "=" * 80 + "\n\n")

            stats_text.insert(tk.END, ">>> Mean:\n")
            stats_text.insert(tk.END, numeric_data.mean().to_string() + "\n\n")

            stats_text.insert(tk.END, ">>> Median:\n")
            stats_text.insert(tk.END, numeric_data.median().to_string() + "\n\n")

            stats_text.insert(tk.END, ">>> Mode:\n")
            stats_text.insert(tk.END, numeric_data.mode().iloc[0].to_string() + "\n\n")

            stats_text.insert(tk.END, ">>> Variance:\n")
            stats_text.insert(tk.END, numeric_data.var().to_string() + "\n\n")

            stats_text.insert(tk.END, ">>> Standard Deviation:\n")
            stats_text.insert(tk.END, numeric_data.std().to_string() + "\n\n")

            stats_text.insert(tk.END, ">>> 25th, 50th, 75th Percentiles (Quartiles):\n")
            stats_text.insert(tk.END, numeric_data.quantile([0.25, 0.5, 0.75]).to_string() + "\n\n")

            stats_text.insert(tk.END, ">>> Correlation Coefficients:\n")
            stats_text.insert(tk.END, numeric_data.corr().to_string() + "\n\n")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to show statistics: {str(e)}")

def show_probabilities():
    global model, vectorizer
    if model is None or vectorizer is None:
        messagebox.showerror("Error", "Train the model first!")
        return

    try:
        feature_names = vectorizer.get_feature_names_out()
        spam_probs = model.feature_log_prob_[1]
        ham_probs = model.feature_log_prob_[0]

        top_spam_indices = spam_probs.argsort()[-10:][::-1]
        top_ham_indices = ham_probs.argsort()[-10:][::-1]

        print("\nTop indicative words for SPAM and their log probabilities:")
        for idx in top_spam_indices:
            print(f"{feature_names[idx]} -> P(Word|Spam): {spam_probs[idx]:.2f}")

        print("\nTop indicative words for HAM and their log probabilities:")
        for idx in top_ham_indices:
            print(f"{feature_names[idx]} -> P(Word|Ham): {ham_probs[idx]:.2f}")

    except Exception as e:
        print(f"Failed to show word probabilities: {str(e)}")


# Function to exit the application
def exit_app():
    root.destroy()


# Create the GUI
root = tk.Tk()
root.title("Spam Detection with Naive Bayes")

# Buttons
btn_load = tk.Button(root, text="Load Dataset", command=load_dataset, width=25)
btn_load.pack(pady=5)

btn_train = tk.Button(root, text="Train Classifier", command=train_classifier, width=25)
btn_train.pack(pady=5)

btn_counts = tk.Button(root, text="Show Spam/Ham Counts", command=show_counts, width=25)
btn_counts.pack(pady=5)

btn_probs = tk.Button(root, text="Show Word Probabilities", command=show_probabilities, width=25)
btn_probs.pack(pady=5)

btn_metrics = tk.Button(root, text="Show Evaluation Metrics", command=show_metrics, width=25)
btn_metrics.pack(pady=5)

btn_stats = tk.Button(root, text="Show Statistics", command=show_statistics, width=25)
btn_stats.pack(pady=5)

label_input = tk.Label(root, text="Enter a message to classify:")
label_input.pack(pady=5)

input_message = tk.Text(root, height=5, width=50)
input_message.pack(pady=5)

btn_classify = tk.Button(root, text="Classify Message", command=classify_message, width=25)
btn_classify.pack(pady=5)

btn_exit = tk.Button(root, text="Exit", command=exit_app, width=25)
btn_exit.pack(pady=5)

root.mainloop() 