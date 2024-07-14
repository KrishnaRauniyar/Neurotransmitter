import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import sys

args= sys.argv[1:]
if len(args) < 2 :
    print("please specify all parameters")
    print("command : python neuro_transmitter_balanced.py full_path csvfile")
    quit(0)

# Full path
Relative_PATH = args[0]
CSVFILE = args[1]

class NeuroTransmitterClassifier:
    def __init__(self, data_path):
        self.data_path = data_path
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.model = None

    def load_data(self):
        df = pd.read_csv(self.data_path)
        return df

    def preprocess_data(self, df):
        class_mapping = {'ASP': 0, 'GLU': 1, '7V7': 2, 'ABU': 3, 'ACH': 4, 'LDP': 5, 'SRO': 6}
        df['label'] = df['group'].map(class_mapping)
        original_counts = df['label'].value_counts().sort_index()
        X = df.drop(['group', 'protein', 'label'], axis=1).values
        y = df['label'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
        scaler = MinMaxScaler()
        self.X_train = scaler.fit_transform(X_train)
        self.X_test = scaler.transform(X_test)
        smote = SMOTE(random_state=42)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, y_train)
        self.X_train, self.y_train = shuffle(self.X_train, self.y_train, random_state=42)
        self.y_train = to_categorical(self.y_train, num_classes=7)
        self.y_test = to_categorical(y_test, num_classes=7)
        balanced_counts = pd.Series(self.y_train.argmax(axis=1)).value_counts().sort_index()
        return original_counts, balanced_counts
    
    def build_model(self):
        self.model = Sequential([
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(units=7, activation='softmax')
        ])
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train_model(self, epochs=25, batch_size=32):
        self.model.fit(x=self.X_train, y=self.y_train, epochs=epochs, batch_size=batch_size, validation_data=(self.X_test, self.y_test), verbose= 2)

    def plot_loss(self):
        plt.plot(self.model.history.history['loss'], label='Training Loss')
        plt.plot(self.model.history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(Relative_PATH+'dnn_balanced/results/loss_plot.png')
        plt.close()

    def plot_accuracy(self):
        plt.plot(self.model.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.model.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(Relative_PATH+'dnn_balanced/results/accuracy_plot.png')
        plt.close()

    def evaluate_model(self):
        y_pred_prob = self.model.predict(self.X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        print(classification_report(np.argmax(self.y_test, axis=1), y_pred))
        conf_matrix = confusion_matrix(np.argmax(self.y_test, axis=1), y_pred)
        print("Confusion Matrix:")
        print(conf_matrix)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['ASP', 'GLU', '7V7', 'ABU', 'ACH', 'LDP', 'SRO'], yticklabels=['ASP', 'GLU', '7V7', 'ABU', 'ACH', 'LDP', 'SRO'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.savefig(Relative_PATH+'dnn_balanced/results/confusion_matrix.png')
        plt.close()

    def plot_roc_curve(self):
        y_test_bin = label_binarize(np.argmax(self.y_test, axis=1), classes=[0, 1, 2, 3, 4, 5, 6])
        y_pred_prob = self.model.predict(self.X_test)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(7):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure(figsize=(8, 6))
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']

        for i, color in zip(range(7), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.savefig(Relative_PATH+'dnn_balanced/results/roc.png')
        plt.close()

if __name__ == "__main__":
    classifier = NeuroTransmitterClassifier(data_path=Relative_PATH+CSVFILE)
    df = classifier.load_data()
    original_counts, balanced_counts = classifier.preprocess_data(df)
    print("Original Counts:")
    print(original_counts)
    print("\nBalanced Counts:")
    print(balanced_counts)
    classifier.build_model()
    classifier.train_model()
    classifier.plot_loss()
    classifier.plot_accuracy()
    classifier.evaluate_model()
    classifier.plot_roc_curve()
