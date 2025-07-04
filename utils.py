from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def encode_labels(df, categorical_cols):
    df_encoded = df.copy()
    encoders = {}
    for col in categorical_cols:
        enc = LabelEncoder()
        df_encoded[col] = enc.fit_transform(df_encoded[col].astype(str))
        encoders[col] = enc
    return df_encoded, encoders

def get_classifiers():
    return {
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'GBRT': GradientBoostingClassifier(random_state=42)
    }

def train_and_evaluate(X, y):
    results = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    classifiers = get_classifiers()
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        results[name] = {
            'model': clf,
            'train_acc': clf.score(X_train, y_train),
            'test_acc': clf.score(X_test, y_test),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'conf_matrix': confusion_matrix(y_test, y_pred),
            'y_pred': y_pred,
            'y_test': y_test
        }
    return results
