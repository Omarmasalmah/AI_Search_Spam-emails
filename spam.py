import numpy as np
import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

TEST_SIZE = 0.3
K = 3

class NN:
    def __init__(self, trainingFeatures, trainingLabels):
        self.trainingFeatures = trainingFeatures
        self.trainingLabels = trainingLabels

    def euclidean_distance(self, example1, example2):
        return np.sqrt(np.sum((example1 - example2)**2))

    def predict(self, features, k):
        """
        Given a list of feature vectors of testing examples,
        return the predicted class labels (list of either 0s or 1s)
        using the k nearest neighbors.
        """
        y_pred = []
        for test_example in features:
            distances = []
            for train_example in self.trainingFeatures:
                distance = self.euclidean_distance(test_example, train_example)
                distances.append(distance)
            distances = np.array(distances)
            sorted_indices = np.argsort(distances)
            k_nearest_indices = sorted_indices[:k]
            k_nearest_labels = self.trainingLabels[k_nearest_indices]
            pred_label = np.argmax(np.bincount(k_nearest_labels))
            y_pred.append(pred_label)
        return y_pred


def load_data(filename):
    """
    Load spam data from a CSV file `filename` and convert it into a list of
    feature vectors and a list of target labels. Return a tuple (features, labels).

    Feature vectors should be a list of lists, where each list contains the
    57 feature values.

    Labels should be the corresponding list of labels, where each label
    is 1 if spam, and 0 otherwise.
    """
    data = np.loadtxt(filename, delimiter=',')
    features = data[:, :-1]
    labels = data[:, -1].astype(int)
    return features, labels


def preprocess(features):
    """
    Normalize each feature by subtracting the mean value in each
    feature and dividing by the standard deviation.
    """
    means = np.mean(features, axis=0)  # Calculate mean values for each feature
    stds = np.std(features, axis=0)  # Calculate standard deviations for each feature

    normalized_features = (features - means) / stds  # Normalize the features

    return normalized_features


def train_mlp_model(features, labels):
    """
    Given a list of feature lists and a list of labels, return a
    fitted MLP model trained on the data using sklearn implementation.
    """
    mlp = MLPClassifier(hidden_layer_sizes=(10, 5), activation='logistic', max_iter=2000)
    mlp.fit(features, labels)
    return mlp



def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return (accuracy, precision, recall, f1).

    Assume each label is either a 1 (positive) or 0 (negative).
    """
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return accuracy, precision, recall, f1


def main():

    # Load data from spreadsheet and split into train and test sets
    filename = "spambase.csv"
    features, labels = load_data(filename)
    features = preprocess(features)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=TEST_SIZE)

    # Train a k-NN model and make predictions
    model_nn = NN(X_train, y_train)
    predictions = model_nn.predict(X_test, K)
    accuracy, precision, recall, f1 = evaluate(y_test, predictions)
    cm_knn = confusion_matrix(y_test, predictions)


    # Print results
    print("**** 1-Nearest Neighbor Results ****")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    print("Confusion Matrix for k-NN :")
    print(cm_knn)

    # Train an MLP model and make predictions
    model = train_mlp_model(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy, precision, recall, f1 = evaluate(y_test, predictions)
    cm_mlp = confusion_matrix(y_test, predictions)

    # Print results
    print("**** MLP Results ****")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    print("Confusion Matrix for MLP:")
    print(cm_mlp)

if __name__ == "__main__":
    main()
