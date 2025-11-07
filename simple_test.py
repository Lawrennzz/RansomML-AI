#!/usr/bin/env python3
import time
import requests


BASE_URL = "http://localhost:5000"


def main():
    print("Checking feature columns...")
    r = requests.get(f"{BASE_URL}/api/feature-columns")
    data = r.json()
    features = data.get('features') or []
    if not features:
        print("No features loaded yet. Training model to initialize...")
        tr = requests.post(f"{BASE_URL}/api/train")
        print(tr.json())
        time.sleep(1)
        r = requests.get(f"{BASE_URL}/api/feature-columns")
        features = r.json().get('features') or []

    print(f"Feature count: {len(features)}")

    # Build a dummy sample of zeros
    sample = {f: 0 for f in features}

    print("Predicting dummy sample...")
    pr = requests.post(f"{BASE_URL}/api/predict", json=sample)
    print(pr.json())


if __name__ == "__main__":
    main()


