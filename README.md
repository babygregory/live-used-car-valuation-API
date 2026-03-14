# Final production stack should look like this
```text
train_malaysia_xgb_export_ui.py
        ↓
model.pkl
preprocessor.pkl
        ↓
app.py (Flask API)
        ↓
index.html UI
```
# Test the API health endpoint: 
OPen browser and type **http://127.0.0.1:5000/api/health**

# How to run on local machine

## 1. In terminal: 

Ensure requirement.txt has these lines below:- 
- Flask==3.1.0
- pandas==2.2.3
- numpy==2.1.3
- scikit-learn==1.5.2
- xgboost==2.1.2
- joblib==1.4.2
 
pip install -r requirements.txt and after all the libraries in requirement.txt are installed **run python app.py**

## 2. Then open browser: Type http://127.0.0.1:5000
