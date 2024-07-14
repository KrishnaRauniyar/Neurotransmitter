import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import sys

args= sys.argv[1:]
if len(args) < 2 :
    print("please specify all parameters")
    print("command : python neuro_transmitter_unbalanced.py full_path csvfile")
    quit(0)

# Full path
Relative_PATH = args[0]
CSVFILE = args[1]

class NeuroTransmitterClassifier:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.y_train_one_hot, self.y_test_one_hot = None, None

    def load_data(self):
        df = pd.read_csv(self.data_path)
        return df

    def preprocess_data(self, df):
        df['label'] = df['group'].map({'ASP': 0, 'GLU': 1, '7V7': 2, 'ABU': 3, 'ACH': 4, 'LDP': 5, 'SRO': 6})
        df = df.drop(['group', 'protein'], axis=1)
        X = df.drop('label', axis=1).values
        y = df['label'].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.20, random_state=101)
        scaler = MinMaxScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        self.y_train_one_hot = to_categorical(self.y_train, num_classes=7)
        self.y_test_one_hot = to_categorical(self.y_test, num_classes=7)

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
        self.model.fit(x=self.X_train, y=self.y_train_one_hot, epochs=epochs, batch_size=batch_size, validation_data=(self.X_test, self.y_test_one_hot))

    def plot_loss(self):
        losses = pd.DataFrame(self.model.history.history)
        print(losses)
        plt.plot(losses['loss'], label='Training Loss')
        plt.plot(losses['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        # plt.show()
        plt.savefig(Relative_PATH+'dnn_unbalanced/results/loss_plot.png')
        plt.close()

    def plot_accuracy(self):
        losses = pd.DataFrame(self.model.history.history)
        plt.plot(losses['accuracy'], label='Training Accuracy')
        plt.plot(losses['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        # plt.show()
        plt.savefig(Relative_PATH+'dnn_unbalanced/results/accuracy_plot.png')
        plt.close()

    def evaluate_model(self):
        y_pred_prob = self.model.predict(self.X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        print(classification_report(self.y_test, y_pred))
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        print("Confusion Matrix:")
        print(conf_matrix)

    def plot_confusion_matrix(self):
        y_pred_prob = self.model.predict(self.X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['ASP', 'GLU', '7V7', 'ABU', 'ACH', 'LDP', 'SRO'], yticklabels=['ASP', 'GLU', '7V7', 'ABU', 'ACH', 'LDP', 'SRO'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        # plt.show()
        plt.savefig(Relative_PATH+'dnn_unbalanced/results/confusion_matrix.png')
        plt.close()

    def plot_roc_curve(self):
        y_test_bin = label_binarize(self.y_test, classes=[0, 1, 2, 3, 4, 5, 6])
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
        # plt.show()
        plt.savefig(Relative_PATH+'dnn_unbalanced/results/roc.png')
        plt.close()

if __name__ == "__main__":
    classifier = NeuroTransmitterClassifier(data_path=Relative_PATH+CSVFILE)
    df = classifier.load_data()
    classifier.preprocess_data(df)
    classifier.build_model()
    classifier.train_model()
    classifier.plot_accuracy()
    classifier.plot_loss()
    classifier.evaluate_model()
    classifier.plot_confusion_matrix()
    classifier.plot_roc_curve()
