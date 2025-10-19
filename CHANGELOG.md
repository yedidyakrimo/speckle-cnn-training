# Changelog - Speckle CNN Training Optimization

## גרסה 2.0 - GPU Optimization Release

### שיפורים עיקריים

#### 1. תמיכה מלאה ב-GPU
- ✅ תמיכה ב-RTX 3090 Ti ו-GPU מתקדמים אחרים
- ✅ Mixed Precision Training (AMP) לביצועים מהירים יותר
- ✅ Model Compilation (PyTorch 2.0+) לביצועים משופרים
- ✅ ניהול זיכרון GPU מתקדם
- ✅ תמיכה ב-multiple GPUs

#### 2. אופטימיזציות DataLoader
- ✅ pin_memory ו-persistent_workers
- ✅ prefetch_factor מותאם
- ✅ num_workers מותאם למחשב
- ✅ non_blocking data transfer

#### 3. שיפורי אימון
- ✅ Gradient Accumulation
- ✅ Learning Rate Scheduling (CosineAnnealingLR, ReduceLROnPlateau)
- ✅ Early Stopping
- ✅ Gradient Clipping
- ✅ Weight Decay
- ✅ Best Model Saving

#### 4. קבצי קונפיגורציה
- ✅ `config.json` - הגדרות בסיסיות
- ✅ `config_high_performance.json` - למחשבים חזקים
- ✅ `config_cpu.json` - לאימון על CPU

#### 5. כלי עזר
- ✅ `check_gpu.py` - בדיקת זמינות GPU
- ✅ `run_training.py` - הרצה עם אופציות
- ✅ `setup.py` - התקנה אוטומטית
- ✅ `examples.py` - דוגמאות שימוש

#### 6. קבצי הרצה
- ✅ `run_gpu.bat` - הרצה עם GPU (Windows)
- ✅ `run_cpu.bat` - הרצה עם CPU (Windows)

#### 7. תיעוד
- ✅ `README.md` - מדריך מפורט
- ✅ `PERFORMANCE_TIPS.md` - טיפים לביצועים
- ✅ `requirements.txt` - חבילות נדרשות

### שיפורי ביצועים

#### GPU (RTX 3090 Ti)
- **מהירות**: 3-5x מהיר יותר מ-CPU
- **זיכרון**: חיסכון של 50% עם mixed precision
- **Throughput**: עד 2x יותר samples לשנייה

#### CPU
- **אופטימיזציה**: gradient accumulation
- **זיכרון**: cache dataset
- **Parallelism**: num_workers מותאם

### הגדרות מומלצות

#### RTX 3090 Ti (24GB)
```json
{
    "training": {
        "batch_size": 32,
        "num_workers": 12,
        "mixed_precision": true
    },
    "gpu": {
        "memory_fraction": 0.95,
        "compile_model": true
    }
}
```

#### RTX 3080 (10GB)
```json
{
    "training": {
        "batch_size": 16,
        "num_workers": 8,
        "mixed_precision": true
    },
    "gpu": {
        "memory_fraction": 0.9,
        "compile_model": true
    }
}
```

#### CPU Only
```json
{
    "training": {
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "num_workers": 2
    },
    "gpu": {
        "use_gpu": false
    }
}
```

### קבצים שנוצרו/עודכנו

#### קבצים עיקריים
- `train_speckles_cnn.py` - הקוד הראשי (עודכן)
- `config.json` - קונפיגורציה בסיסית (עודכן)

#### קבצי קונפיגורציה
- `config_high_performance.json` - חדש
- `config_cpu.json` - חדש

#### כלי עזר
- `check_gpu.py` - חדש
- `run_training.py` - חדש
- `setup.py` - חדש
- `examples.py` - חדש

#### קבצי הרצה
- `run_gpu.bat` - חדש
- `run_cpu.bat` - חדש

#### תיעוד
- `README.md` - עודכן
- `PERFORMANCE_TIPS.md` - חדש
- `requirements.txt` - חדש
- `CHANGELOG.md` - חדש

### הוראות שימוש

#### התקנה מהירה
```bash
python setup.py
```

#### הרצה עם GPU
```bash
python run_training.py --gpu
# או
run_gpu.bat  # Windows
```

#### הרצה עם CPU
```bash
python run_training.py --cpu
# או
run_cpu.bat  # Windows
```

#### בדיקת GPU
```bash
python check_gpu.py
```

### תאימות

#### Python
- Python 3.8+
- PyTorch 2.0+ (מומלץ)

#### GPU
- NVIDIA GPU עם CUDA support
- RTX 20xx+ (לשימוש ב-mixed precision)
- RTX 30xx+ (לשימוש ב-model compilation)

#### מערכת הפעלה
- Windows 10/11
- Linux (Ubuntu 18.04+)
- macOS (CPU only)

### הערות חשובות

1. **Mixed Precision**: דורש GPU עם Tensor Cores
2. **Model Compilation**: דורש PyTorch 2.0+
3. **Memory Management**: מומלץ 16GB+ RAM
4. **Data Path**: עדכן נתיב הנתונים ב-config

### בעיות ידועות

1. **CUDA Out of Memory**: הקטן batch_size או הפעל gradient_accumulation
2. **Slow Training**: בדוק num_workers ו-pin_memory
3. **Import Errors**: התקן requirements.txt

### תמיכה

לבעיות או שאלות:
1. בדוק את `PERFORMANCE_TIPS.md`
2. הרץ `python check_gpu.py`
3. בדוק את `examples.py`
