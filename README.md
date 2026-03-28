# 🌱 LeafIQ — AI-Powered Apple Disease Detection

LeafIQ is an intelligent decision-support system that detects apple leaf diseases using deep learning, while also handling uncertainty to avoid misleading predictions.

---

## 🚀 Features

- 🍃 Detects:
  - Apple Scab
  - Black Rot
  - Rust
  - Healthy

- 📊 Top-2 Predictions with confidence scores  
- ⚠️ Uncertainty Handling (avoids wrong predictions)  
- 🧠 Confidence Reasoning (explains why model is confident)  
- 🩺 Disease Explanation & Severity Estimation  
- 🌾 Farmer-Friendly Treatment Suggestions  
- 🖼️ Input Validation (rejects non-leaf images)

---

## 🧠 How It Works

1. User uploads a leaf image  
2. Image is validated (ensures it's a leaf)  
3. Model predicts disease using deep learning  
4. System returns:
   - Prediction (Top-2)
   - Confidence score
   - Explanation
   - Treatment
   - Severity
   - Confidence reasoning  

---

## 🛠️ Tech Stack

- **Frontend:** HTML, CSS, JavaScript  
- **Backend:** Flask (Python)  
- **Model:** TensorFlow / Keras (MobileNet-based)  
- **Deployment:** Vercel (Frontend), Render (Backend)

---

## ⚙️ Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/luffy69a/smart-orchard-ai.git
cd smart-orchard-ai