# טיפים לביצועים - Speckle CNN Training

## אופטימיזציות GPU

### 1. הגדרות זיכרון GPU
```json
"gpu": {
    "memory_fraction": 0.9,  // השתמש ב-90% מהזיכרון
    "allow_growth": true     // הגדל זיכרון דינמית
}
```

### 2. Mixed Precision Training
```json
"gpu": {
    "mixed_precision": true  // הפעל mixed precision
}
```
- חוסך ~50% זיכרון
- מהיר יותר ב-1.5-2x
- דורש GPU עם Tensor Cores (RTX 20xx+)

### 3. Model Compilation (PyTorch 2.0+)
```json
"gpu": {
    "compile_model": true  // קומפל את המודל
}
```
- מהיר יותר ב-20-30%
- דורש PyTorch 2.0+

## אופטימיזציות DataLoader

### 1. הגדרות Worker
```json
"training": {
    "num_workers": 8,           // מספר workers
    "pin_memory": true,         // pin memory
    "persistent_workers": true  // שמור workers
}
```

### 2. Prefetch Factor
```json
"performance": {
    "prefetch_factor": 2  // טען 2 batches מראש
}
```

## אופטימיזציות אימון

### 1. Gradient Accumulation
```json
"training": {
    "gradient_accumulation_steps": 2  // צבור gradients
}
```
- מאפשר batch size גדול יותר
- חוסך זיכרון

### 2. Learning Rate Scheduling
```json
"training": {
    "scheduler": "CosineAnnealingLR",
    "scheduler_params": {
        "T_max": 20,
        "eta_min": 0.0001
    }
}
```

### 3. Early Stopping
```json
"performance": {
    "early_stopping_patience": 5  // עצור אחרי 5 epochs ללא שיפור
}
```

## הגדרות לפי סוג GPU

### RTX 4090 / RTX 3090 Ti (24GB+)
```json
{
    "training": {
        "batch_size": 32,
        "num_workers": 12
    },
    "gpu": {
        "memory_fraction": 0.95,
        "mixed_precision": true,
        "compile_model": true
    }
}
```

### RTX 3080 / RTX 4070 (10-12GB)
```json
{
    "training": {
        "batch_size": 16,
        "num_workers": 8
    },
    "gpu": {
        "memory_fraction": 0.9,
        "mixed_precision": true,
        "compile_model": true
    }
}
```

### RTX 3060 / RTX 4060 (8GB)
```json
{
    "training": {
        "batch_size": 8,
        "gradient_accumulation_steps": 2,
        "num_workers": 6
    },
    "gpu": {
        "memory_fraction": 0.8,
        "mixed_precision": true,
        "compile_model": false
    }
}
```

### GPU עם זיכרון נמוך (4-6GB)
```json
{
    "training": {
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "num_workers": 4
    },
    "gpu": {
        "memory_fraction": 0.7,
        "mixed_precision": true,
        "compile_model": false
    }
}
```

## טיפים כלליים

### 1. ניטור זיכרון
```python
# בדוק שימוש זיכרון
print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
print(f"Peak Memory: {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")
```

### 2. ניקוי זיכרון
```python
# נקה זיכרון בין folds
torch.cuda.empty_cache()
```

### 3. הגדרות מערכת
- סגור תוכנות אחרות
- השתמש ב-SSD לאחסון נתונים
- ודא שיש מספיק RAM (16GB+)

### 4. בדיקת ביצועים
```bash
# בדוק שימוש GPU
nvidia-smi

# הרץ בדיקת ביצועים
python check_gpu.py
```

## פתרון בעיות נפוצות

### 1. Out of Memory
- הקטן batch_size
- הפעל gradient_accumulation_steps
- הקטן memory_fraction
- השתמש ב-mixed_precision

### 2. איטיות
- הגדל num_workers
- הפעל pin_memory
- השתמש ב-persistent_workers
- הפעל model compilation

### 3. שגיאות CUDA
- בדוק גרסת CUDA
- התקן PyTorch עם CUDA support
- בדוק תאימות GPU

## מדידת ביצועים

### 1. זמן אימון
```python
import time
start_time = time.time()
# ... training code ...
print(f"Training took {time.time() - start_time:.2f} seconds")
```

### 2. שימוש זיכרון
```python
import psutil
print(f"RAM usage: {psutil.virtual_memory().percent}%")
print(f"GPU usage: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
```

### 3. Throughput
```python
# דוגמאות לשנייה
samples_per_second = len(dataset) / training_time
print(f"Throughput: {samples_per_second:.2f} samples/sec")
```
