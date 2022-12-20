## Decision Transformer

1. Create a fresh environment (to avoid version conflicts) and install requirements.
```
pip install -r requirements.txt
```

2. Generate random dataset for offline training (outputs  a file named `mario-v0-offline-random-dataset.pkl`)
```
python generate_offline_data.py
```

3. Train and Run the model (uses the file genereted by step 1)
```
python train_and_run.py
```

4. Check Some output frames
```
output frames are stored in ./output
```
