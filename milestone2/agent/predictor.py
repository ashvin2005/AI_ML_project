import os
import re
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "milestone1", "svm_model.joblib")
VECTORIZER_PATH = os.path.join(BASE_DIR, "milestone1", "tfidf_vectorizer.joblib")


class CredibilityPredictor:
    def __init__(self):
        if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
            raise FileNotFoundError(
                "Model files not found. Train the model first (see milestone1/model.ipynb)."
            )
        self.model = joblib.load(MODEL_PATH)
        self.vectorizer = joblib.load(VECTORIZER_PATH)

    def _clean(self, text):
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+", " ", text)
        text = re.sub(r"<.*?>", " ", text)
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def predict(self, text: str) -> dict:
        vec = self.vectorizer.transform([self._clean(text)])
        raw = self.model.predict(vec)[0]

        proba_real, proba_fake = 0.5, 0.5
        try:
            proba = self.model.predict_proba(vec)[0]
            classes = list(self.model.classes_)
            if 1 in classes:
                proba_real = float(proba[classes.index(1)])
            if 0 in classes:
                proba_fake = float(proba[classes.index(0)])
        except Exception:
            pass

        label = "REAL" if raw == 1 else "FAKE"
        return {
            "label": label,
            "confidence": round(max(proba_real, proba_fake) * 100, 1),
            "proba_real": round(proba_real * 100, 1),
            "proba_fake": round(proba_fake * 100, 1),
        }