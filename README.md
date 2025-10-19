# Speckle CNN Training with GPU Optimization

פרויקט לאימון מודל CNN לזיהוי דפוסי speckle עם אופטימיזציות GPU מתקדמות.

## התקנה

### התקנה אוטומטית (מומלץ)
```bash
python setup.py
```

### התקנה ידנית
1. התקן את החבילות הנדרשות:
```bash
pip install -r requirements.txt
```

2. בדוק את זמינות ה-GPU:
```bash
python check_gpu.py
```

3. ודא שיש לך CUDA מותקן על המחשב (לשימוש ב-GPU).

### דרישות מערכת
- Python 3.8+
- NVIDIA GPU עם CUDA support (לשימוש ב-GPU)
- 16+ GB RAM (מומלץ 32+ GB)
- 10+ GB שטח דיסק פנוי

## טיפים לביצועים
למדריך מפורט לאופטימיזציות ביצועים, ראה [PERFORMANCE_TIPS.md](PERFORMANCE_TIPS.md)

## קונפיגורציה

הקובץ `config.json` מכיל את כל ההגדרות:

### הגדרות GPU
- `use_gpu`: הפעל/כבה שימוש ב-GPU
- `gpu_id`: מזהה GPU (0, 1, וכו')
- `mixed_precision`: שימוש ב-mixed precision training
- `compile_model`: קומפילציה של המודל לביצועים טובים יותר
- `memory_fraction`: חלק מהזיכרון הזמין לשימוש (0.9 = 90%)
- `allow_growth`: הגדלת זיכרון GPU דינמית

### הגדרות אימון
- `batch_size`: גודל batch (16)
- `epochs`: מספר epochs (20)
- `learning_rate`: קצב למידה (0.001)
- `optimizer`: אופטימייזר (AdamW)
- `num_workers`: מספר worker processes לטעינת נתונים (8)
- `gradient_accumulation_steps`: צעדי הצטברות gradient (2)
- `weight_decay`: דעיכת משקלים (0.01)

### הגדרות ביצועים
- `pin_memory`: שימוש ב-pin memory
- `persistent_workers`: שמירת workers בין epochs
- `prefetch_factor`: גורם prefetch
- `gradient_clip_norm`: גזירת gradient
- `early_stopping_patience`: סבלנות לעצירה מוקדמת
- `save_best_model`: שמירת המודל הטוב ביותר

## הרצה

### הרצה בסיסית
```bash
python train_speckles_cnn.py
```

### הרצה עם קונפיגורציות שונות

#### Windows (Double-click)
- `run_gpu.bat` - הרצה עם GPU
- `run_cpu.bat` - הרצה עם CPU

#### Command Line
```bash
# הרצה עם הגדרות GPU מתקדמות
python run_training.py --gpu

# הרצה עם הגדרות CPU
python run_training.py --cpu

# הרצה עם קובץ קונפיג ספציפי
python run_training.py --config config_high_performance.json
```

### קבצי קונפיגורציה זמינים
- `config.json`: הגדרות בסיסיות עם GPU
- `config_high_performance.json`: הגדרות מתקדמות למחשבים חזקים
- `config_cpu.json`: הגדרות לאימון על CPU בלבד

### דוגמאות שימוש
```bash
# הרץ דוגמאות שונות
python examples.py

# בדוק את זמינות ה-GPU
python check_gpu.py
```

## אופטימיזציות שהוספו

1. **Mixed Precision Training**: שימוש ב-AMP לביצועים מהירים יותר
2. **DataLoader Optimization**: pin_memory, persistent_workers, prefetch_factor
3. **Gradient Accumulation**: אימון עם batches גדולים יותר
4. **Learning Rate Scheduling**: CosineAnnealingLR או ReduceLROnPlateau
5. **Early Stopping**: עצירה מוקדמת למניעת overfitting
6. **Model Compilation**: קומפילציה של המודל (PyTorch 2.0+)
7. **Memory Management**: ניהול זיכרון GPU מתקדם
8. **Best Model Saving**: שמירת המודל עם הvalidation loss הנמוך ביותר

## פלטים

- `predictions.csv`: תחזיות המודל
- `speckle_cnn.pth`: המודל הטוב ביותר
- `normalized_confusion_matrix.png`: מטריצת בלבול מנורמלת

## דרישות מערכת

- GPU עם CUDA support (מומלץ RTX 3090 Ti או דומה)
- 16+ GB RAM
- Python 3.8+
- PyTorch עם CUDA support
