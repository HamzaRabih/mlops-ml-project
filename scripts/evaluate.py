# scripts/evaluate.py
import json
from pathlib import Path
import joblib
import yaml
from sklearn.metrics import classification_report
from src.data import load_dataset

def load_cfg(path="config/train.yaml"):
    return yaml.safe_load(open(path, "r", encoding="utf-8"))

def main():
    cfg = load_cfg()
    art_dir = Path(cfg.get("artifacts_dir", "artifacts"))
   
    model = joblib.load(art_dir / "model.joblib")
    X, y = load_dataset(cfg)
   
    pred = model.predict(X)
    report = classification_report(y, pred, output_dict=True)
   
    json.dump(report, open(art_dir / "report.json", "w"), indent=2)
    print("Evaluate OK: artifacts/report.json")

if __name__ == "__main__":
    main()