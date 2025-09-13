# ✍️ Handwritten Character Recognition  

**Project Overview:**  
This project is a **Handwritten Character Recognition** system built using **Convolutional Neural Networks (CNN)** and deployed as a **Streamlit web application**. Users can draw a character on a canvas, click **Predict**, and the system will identify the character.  

---

**Try Streamlit App:** [Streamlit App](https://handwritten-character-recognition1.streamlit.app/)  

**Dataset:** [Kaggle Dataset](https://www.kaggle.com/datasets/dhruvildave/english-handwritten-characters-dataset)

---

## 🛠️ Features  

- **Interactive Canvas:** Draw characters directly in the browser with a sleek, white canvas.  
- **Real-Time Prediction:** Click a button to get the predicted character instantly.  
- **User-Friendly UI:** Modern dark-themed interface with soft purple accents and smooth hover effects.  
- **Model Training:** Uses a CNN trained on custom dataset with data augmentation to improve accuracy.  
- **Flexible Label Mapping:** Supports custom CSV files with character labels.  

---

## 📁 Project Structure  

project/
│
├─ app.py # Streamlit web app
├─ handwritten_cnn_model.keras # Trained CNN model
├─ english.csv # CSV with 'image' and 'label' columns
├─ dataset/ # Folder containing handwritten character images
└─ README.md # Project documentation

---

## 🖥️ Model Architecture and Explanation  

The CNN model is designed for recognizing handwritten characters from images. Here's a detailed breakdown:  

1. **Input Layer:**  
   - Accepts images resized to a fixed dimension (e.g., 28x28 pixels, grayscale).  
   
2. **Convolutional Layers:**  
   - Extracts spatial features such as edges, corners, and strokes.  
   - Multiple convolutional layers with ReLU activation are used to capture complex patterns.  
   
3. **Pooling Layers:**  
   - MaxPooling layers reduce spatial dimensions while retaining the most important features.  
   - Helps in reducing computation and overfitting.  
   
4. **Dropout Layers:**  
   - Added to prevent overfitting by randomly ignoring some neurons during training.  

5. **Fully Connected (Dense) Layers:**  
   - Combines extracted features to make predictions.  
   - The final dense layer uses **softmax activation** to output a probability distribution over all possible character classes.  

6. **Output Layer:**  
   - Returns the **predicted character** with the **highest probability**.  

**Training Details:**  
- **Dataset:** Custom handwritten character images, labeled in `english.csv`.  
- **Data Augmentation:** Rotation, zoom, horizontal/vertical flipping to make the model robust.  
- **Loss Function:** Categorical Cross-Entropy.  
- **Optimizer:** Adam optimizer for fast convergence.  
- **Metrics:** Accuracy monitored during training.  

---

## 📊 Model Performance  

After training on the custom dataset, the model achieved the following approximate performance:  

- **Training Accuracy:** 86%  
- **Validation Accuracy:** 83%  
- **Test Accuracy:** 83%  

- **Loss:**  
  - Training Loss: 0.32
  - Validation Loss: 0.5

**Observations:**  
- Most misclassifications occur for visually similar characters.  
- Data augmentation improved generalization.  
- The model is lightweight and fast, suitable for real-time predictions in Streamlit.  

---

## 📊 Model Output  

- The model outputs the **predicted character label** along with a **confidence score** (probability).  
- Example:  
Predicted Character: 'A'
Confidence: 92%


- The prediction is dynamically mapped from the `english.csv` file so that the displayed character matches the label of the highest probability class.  
- The output is shown in a clear, white prediction box below the canvas.  

---

## 🚀 Running the App  

streamlit run app.py

- Open the app in your browser.
- Draw a character in the canvas.
- Click ✨ Predict.
- View the predicted character in the stylish prediction box.

## 📌 Notes
- The canvas box in the app is white, as per design specifications.

- Hover effects and UI styling are consistent across the app.

- Labels are read from english.csv and mapped to predictions dynamically.

