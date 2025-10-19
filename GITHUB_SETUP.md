# 🚀 העלאה לגיטהאב - הוראות מפורטות

## שלב 1: יצירת Repository בגיטהאב

1. **לך לגיטהאב**: https://github.com
2. **לחץ על "New"** או "+" בפינה הימנית העליונה
3. **מלא פרטים**:
   - **Repository name**: `speckle-cnn-training`
   - **Description**: `GPU-optimized CNN training for speckle pattern recognition with RTX 3090 Ti support`
   - **Visibility**: בחר Public או Private
   - **❌ אל תסמן** "Add a README file" (כבר יש לנו)
   - **❌ אל תסמן** "Add .gitignore" (כבר יש לנו)
   - **❌ אל תסמן** "Choose a license" (אופציונלי)
4. **לחץ "Create repository"**

## שלב 2: העלאה אוטומטית

### אפשרות A: Windows (Double-click)
```bash
upload_to_github.bat
```

### אפשרות B: Python (כל הפלטפורמות)
```bash
python upload_to_github.py
```

### אפשרות C: ידנית
```bash
# 1. הגדר את ה-URL (החלף YOUR_USERNAME ו-REPO_NAME)
git remote set-url origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# 2. העלה לגיטהאב
git push -u origin main
```

## שלב 3: אימות (Authentication)

אם אתה נתקל בשגיאות אימות, יש לך כמה אפשרויות:

### אפשרות 1: GitHub CLI (מומלץ)
```bash
# התקן GitHub CLI
# Windows: winget install GitHub.cli
# או הורד מ: https://cli.github.com/

# התחבר
gh auth login

# העלה
git push -u origin main
```

### אפשרות 2: Personal Access Token
1. לך ל: https://github.com/settings/tokens
2. לחץ "Generate new token" → "Generate new token (classic)"
3. בחר scopes: `repo`, `workflow`
4. העתק את ה-token
5. השתמש ב-token כסיסמה כשתעלה

### אפשרות 3: SSH Keys
```bash
# צור SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"

# הוסף ל-GitHub
# העתק את התוכן של ~/.ssh/id_ed25519.pub
# הוסף ב: https://github.com/settings/keys

# שנה ל-SSH URL
git remote set-url origin git@github.com:USERNAME/REPO.git
```

## שלב 4: בדיקה

אחרי ההעלאה, בדוק:
1. **Repository URL**: https://github.com/YOUR_USERNAME/speckle-cnn-training
2. **קבצים**: כל הקבצים שלנו אמורים להיות שם
3. **README**: אמור להציג את התיעוד שלנו

## קבצים שיועלו

- ✅ `train_speckles_cnn.py` - הקוד הראשי
- ✅ `config.json` - קונפיגורציה בסיסית
- ✅ `config_high_performance.json` - למחשבים חזקים
- ✅ `config_cpu.json` - לאימון על CPU
- ✅ `check_gpu.py` - בדיקת GPU
- ✅ `run_training.py` - הרצה עם אופציות
- ✅ `setup.py` - התקנה אוטומטית
- ✅ `examples.py` - דוגמאות שימוש
- ✅ `run_gpu.bat` / `run_cpu.bat` - קבצי הרצה
- ✅ `README.md` - תיעוד מפורט
- ✅ `PERFORMANCE_TIPS.md` - טיפים לביצועים
- ✅ `requirements.txt` - חבילות נדרשות
- ✅ `.gitignore` - קבצים להתעלמות

## בעיות נפוצות

### "Repository not found"
- בדוק שה-URL נכון
- ודא שיש לך גישה ל-repository

### "Authentication failed"
- השתמש ב-GitHub CLI או Personal Access Token
- בדוק שה-SSH key מוגדר נכון

### "Permission denied"
- ודא שיש לך write access ל-repository
- בדוק שה-repository לא private אם אתה לא מחובר

### "Remote origin already exists"
```bash
git remote remove origin
git remote add origin YOUR_NEW_URL
```

## המשך פיתוח

אחרי ההעלאה הראשונה, תוכל:

```bash
# הוסף שינויים
git add .
git commit -m "Your commit message"
git push

# הורד repository אחר
git clone https://github.com/YOUR_USERNAME/speckle-cnn-training.git

# עדכן repository קיים
git pull origin main
```

## תמיכה

אם אתה נתקל בבעיות:
1. בדוק את הודעות השגיאה
2. השתמש ב-`upload_to_github.py` לבדיקה אוטומטית
3. בדוק את התיעוד של GitHub
