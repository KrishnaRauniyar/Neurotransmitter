import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys

args= sys.argv[1:]
if len(args) < 2 :
    print("please specify all parameters")
    print("command : python svm.py full_path csvfile")
    quit(0)

# Full path
Relative_PATH = args[0]
CSVFILE = args[1]

class SVMModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.model = None

    def load_data(self):
        df = pd.read_csv(self.data_path)
        class_mapping = {'ASP': 0, 'GLU': 1, '7V7': 2, 'ABU': 3, 'ACH': 4, 'LDP': 5, 'SRO': 6}
        df['label'] = df['group'].map(class_mapping)
        X = df.drop(['group', 'protein', 'label'], axis=1).values
        y = df['label'].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.30, random_state=101)

    def preprocess_data(self):
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(self.X_train)
        smote = SMOTE(random_state=42)
        self.X_train, self.y_train = smote.fit_resample(X_scaled, self.y_train)

    def train_model(self):
        svm_classifier = SVC(kernel='linear', random_state=42)
        param_grid_svm = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
        }
        grid_search_svm = GridSearchCV(svm_classifier, param_grid_svm, cv=3, scoring='accuracy')
        grid_search_svm.fit(self.X_train, self.y_train)
        self.model = grid_search_svm.best_estimator_

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        print("\nSupport Vector Machine (SVM) Results:")
        print(f"Accuracy: {accuracy_score(self.y_test, y_pred)}")
        print(classification_report(self.y_test, y_pred))
        print(confusion_matrix(self.y_test, y_pred))


if __name__ == "__main__":
    svm_model = SVMModel(data_path=Relative_PATH+CSVFILE)
    svm_model.load_data()
    svm_model.preprocess_data()
    svm_model.train_model()
    svm_model.evaluate_model()
