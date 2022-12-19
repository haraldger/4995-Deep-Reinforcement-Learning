## Decision Transformer

1. Generate random dataset for offline training (outputs  a file named `mario-v0-offline-random-dataset.pkl`)
```
python generate_offline_data.py
```

2. Train and Run the model (uses the file genereted by step 1)
```
python train_and_run.py
```

3. Check Some output frames
```
output frames are stored in ./output
```
