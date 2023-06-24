import sys
import pandas as pd
import ssl
import tkinter as tk
from tkinter import ttk
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from pandastable import Table
import matplotlib.pyplot as plt

# downloading the dataset
ssl._create_default_https_context = ssl._create_unverified_context
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
headers = [
    "class", "Alcohol", "Malicacid", "Ash", "Alcalinity_of_ash", "Magnesium",
    "Total_phenols", "Flavanoids", "Nonflavanoid_phenols", "Proanthocyanins",
    "Color_intensity", "Hue", "0D280_0D315_of_diluted_wines", "Proline"
]

df = pd.read_csv(url, names=headers)
# Represents data from all columns except class
X = df.iloc[:, 1:].values
# Represents class value
y = df.iloc[:, 0].values

# Initialize the model
knn = KNeighborsClassifier()

# Training function
def train_model():
    global knn, X_test, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2137, shuffle=True)
    knn.fit(X_train, y_train)
    create_output_window("Model trained successfully!")

# Testing function
def test_model():
    global knn, X_test, y_test
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    cross_validation = cross_val_score(knn, X, y, cv=50)
    output_text = f"Model tested successfully!\n"
    output_text += f"Accuracy: {accuracy}\n"
    output_text += f"Confusion Matrix:\n{confusion}\n"
    output_text += f"Cross-Validation Scores: {cross_validation}\n"
    create_output_window(output_text)

# Function to predict data passed by user
def predict_data(data):
    global knn
    try:
        data = [float(x) for x in data.split(",")] # Data need to be passed in comma seperated string
        prediction = knn.predict([data])
        create_output_window(f"Prediction: {prediction[0]}")
    except ValueError:
        create_output_window("Invalid input!")

# Adding new data passed by user to the data folder
def add_data(data):
    global df, X, y
    try:
        data = [float(x) for x in data.split(",")]
        new_row = pd.DataFrame([data], columns=headers) # creating new data frame
        df = pd.concat([df, new_row], ignore_index=True) # adds to existing data
        X = df.iloc[:, 1:].values # updates data matrix
        y = df.iloc[:, 0].values # updates class matrix
        create_output_window("New data added successfully!")
    except ValueError:
        create_output_window("Invalid input!")

# Displaying in table format
def show_data():
    global df
    top = tk.Toplevel()
    table = Table(top, dataframe=df)
    table.show()

# Visualization of data
def visualize_data():
    global df
    plt.figure(figsize=(10, 6))

    # Gooing in loop through unique class values
    for class_label in df['class'].unique():
        # Filter values for each class
        class_data = df[df['class'] == class_label]
        # Hardcoding Alcohol parameter as x and second as y
        plt.scatter(class_data['Alcohol'], class_data['Color_intensity'], label=class_label)
    plt.xlabel('Alcohol')
    plt.ylabel('Color Intensity')
    plt.title('Wine Classification')
    plt.legend()
    plt.show()

# Function to display data in seperate window
def create_output_window(output_text):
    window = tk.Toplevel()
    window.title("Output")
    window.geometry("400x200")

    output_label = ttk.Label(window, text=output_text)
    output_label.pack(pady=10)

# Creating main window
window = tk.Tk()
window.title("Wine Classification")
window.geometry("400x400")

# Adding buttons
train_button = ttk.Button(window, text="Train Model", command=train_model)
train_button.pack(pady=10)

test_button = ttk.Button(window, text="Test Model", command=test_model)
test_button.pack(pady=10)

predict_label = ttk.Label(window, text="Enter new comma-separated data :")
predict_label.pack(pady=10)

predict_entry = ttk.Entry(window)
predict_entry.pack(pady=5)

predict_button = ttk.Button(window, text="Predict", command=lambda: predict_data(predict_entry.get()))
predict_button.pack(pady=5)

add_label = ttk.Label(window, text="Enter new comma-separated data :")
add_label.pack(pady=10)

add_entry = ttk.Entry(window)
add_entry.pack(pady=5)

add_button = ttk.Button(window, text="Add Data", command=lambda: add_data(add_entry.get()))
add_button.pack(pady=5)

show_button = ttk.Button(window, text="Show All Data", command=show_data)
show_button.pack(pady=10)

visualize_button = ttk.Button(window, text="Visualize Data", command=visualize_data)
visualize_button.pack(pady=10)

# Starts the loop of app
window.mainloop()