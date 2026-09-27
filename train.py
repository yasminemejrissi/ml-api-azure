from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

# 1. Charger les données (Banknote Authentication, UCI id=267)
banknote = fetch_ucirepo(id=267)
X = banknote.data.features
y = banknote.data.targets.iloc[:, 0]  # l'étiquette : 0 ou 1

print("Taille du dataset :", X.shape)
print("Colonnes :", list(X.columns))
print("Repartition des classes :")
print(y.value_counts())

# 2. Séparer train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Pipeline = prétraitement + modèle dans un seul objet
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000)),
])
pipeline.fit(X_train, y_train)

# 4. Évaluer
print(classification_report(y_test, pipeline.predict(X_test)))

# 5. Sauvegarder
Path("model").mkdir(exist_ok=True)
joblib.dump(pipeline, "model/model.joblib")
print("Modele sauvegarde dans model/model.joblib")

# 6. Vérifier qu'on peut le recharger et prédire
reloaded = joblib.load("model/model.joblib")
print("Test rechargement, prediction du 1er billet :", reloaded.predict(X_test.iloc[[0]]))