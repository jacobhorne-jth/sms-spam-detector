# sms-spam-detector
A machine learning web app that classifies SMS messages as spam or ham using TF-IDF vectorization and custom text features.

**Live Demo: https://sms-spam-checker.netlify.app**  

_Note on Loading Time_
```text
Because the backend API is hosted on a free Render service, it automatically goes to sleep after periods of
inactivity. If you visit the app after it has been idle, the first request can take up to 30 seconds to
respond while the server wakes up. Subsequent requests will be much faster.
```

- Frontend: [https://sms-spam-checker.netlify.app](https://sms-spam-checker.netlify.app)
- Backend API: [https://sms-spam-api-3g2w.onrender.com/predict](https://sms-spam-api-3g2w.onrender.com/predict)

---

**Features**
- Logistic Regression classifier trained on real SMS data
- TF-IDF vectorization combined with custom NLP features:
  - Phone number detection
  - Money symbol presence (£, $, €)
  - Spam keywords ("free", "win", "prize", etc.)
  - Uppercase usage patterns
- Clean, accessible UI designed for all audiences
- Real-time prediction with probability score

---

**Dataset**

The core dataset is the [UCI SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset/data), containing 5,574 SMS messages labeled as `ham` or `spam`.  
To improve detection of modern smishing and phishing messages, I augmented this dataset with synthetic examples.

---

**Project Structure**
```text

backend/

  api.py # FastAPI server
  
frontend/

  index.html # Web UI
  
  script.js # API calls
  
  styles.css # Styling
  
model/

  spam_detector_model.pkl # Trained Logistic Regression model
  
  tfidf_vectorizer.pkl # Saved vectorizer
  
datasets/

  spam.csv
  
  synthetic_smishing_messages.csv
  
  ... # Additional training data
  
train.py # Model training script

requirements.txt # Python dependencies
```

---

**How to Run Locally**

1. Clone the repository

```text
git clone https://github.com/jacobhorne-jth/sms_spam_detector.git
cd sms_spam_detector
```
2. Install dependencies
```text
pip install -r requirements.txt
```
3. Download NLTK stopwords (first time only)
```text
import nltk
nltk.download("stopwords")
```
4. Train the model (optional)
```text
python train.py
```
5. Run the API server
```text
uvicorn backend.api:app --reload
```
6. Open the frontend

Use any static server or drag index.html into your browser.

**Deployment**
- Backend: Render (FastAPI)

- Frontend: Netlify (static hosting)

- Auto-deploy on push to main

**Example Prediction**
Input:
```text
Congratulations! You have won £1000 cash! Call to claim your prize.
```
Output:
```text
Prediction: SPAM
Spam Probability: 0.92
```


**License**

This project is licensed under the MIT License.

**Acknowledgments**
- UCI SMS Spam Collection Dataset
- scikit-learn, FastAPI, and Netlify for making deployment accessible
