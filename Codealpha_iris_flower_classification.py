# Project Name: Iris Flower Classification

# This script demonstrates a basic machine learning project using the Iris dataset.
# It covers data loading, splitting, model training, and evaluation.

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    """
    Main function to run the Iris classification project.
    """
    try:
        # Step 1: Load the Iris dataset from the provided CSV file
        # The data is loaded into a pandas DataFrame for easy manipulation.
        # Ensure the file is at the correct path.
        print("Step 1: Loading the Iris dataset...")

        # This line uses the environment variable to find the uploaded file.
        # You don't need to manually specify the 'Iris.csv' file path.
        file_path = 'Iris.csv'
        
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"Error: The file '{file_path}' was not found. Please make sure the file is available.")
            return

        df = pd.read_csv(file_path)
        print("Dataset loaded successfully.")
        print("\nFirst 5 rows of the dataset:")
        print(df.head())

        # Step 2: Prepare the data for machine learning
        # We separate the features (X) from the target variable (y).
        # Features are the measurements (Sepal and Petal dimensions).
        # The target is the 'Species' column.
        X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
        y = df['Species']

        # Step 3: Split the data into training and testing sets
        # We'll use 80% of the data for training and 20% for testing.
        # `random_state` ensures that the split is the same every time you run the script,
        # which is useful for reproducibility.
        print("\nStep 2: Splitting data into training (80%) and testing (20%) sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"Training set size: {X_train.shape[0]} samples")
        print(f"Testing set size: {X_test.shape[0]} samples")

        # Step 4: Choose and train a machine learning model
        # We'll use the K-Nearest Neighbors (KNN) classifier.
        # We instantiate the model with a value of k=5 (number of neighbors).
        print("\nStep 3: Training the K-Nearest Neighbors (KNN) model...")
        model = KNeighborsClassifier(n_neighbors=5)

        # The model learns the relationship between features and the target variable
        # by fitting itself to the training data.
        model.fit(X_train, y_train)
        print("Model training complete.")

        # Step 5: Make predictions on the test data
        # Now we use our trained model to predict the species for the test set.
        print("\nStep 4: Making predictions on the test data...")
        y_pred = model.predict(X_test)
        print("Predictions complete.")

        # Step 6: Evaluate the model's performance
        # We compare the predicted species with the actual species from the test set.
        print("\nStep 5: Evaluating the model's performance...")
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        print(f"\nModel Accuracy: {accuracy:.2f}")
        print("\nClassification Report:")
        print(report)

        print("\nConfusion Matrix:")
        print(cm)

        # Plotting the confusion matrix for better visualization
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=model.classes_,
            yticklabels=model.classes_
        )
        plt.title('Confusion Matrix for Iris Classification')
        plt.xlabel('Predicted Species')
        plt.ylabel('Actual Species')
        plt.show()

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# This ensures the main function runs when the script is executed.
if __name__ == '__main__':
    main()
