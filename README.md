# ✋ Hand Recognition using CNN
CNN-based static hand gesture classification  
📁 GitHub: [`rehaan1603/SCT_ML_4`](https://github.com/rehaan1603/SCT_ML_4)

---

## 📌 Project Overview

- **Objective**: Build a CNN model to classify static hand gestures from grayscale images.
- **Model**: Convolutional Neural Network (CNN)
- **Dataset**: CSV files (`sign_mnist_train.csv`, `sign_mnist_test.csv`) with 28x28 grayscale images and 25 gesture labels.
- **Platform Used**: Google Colab (Jupyter Notebook environment)

---

## 🛠️ Technologies Used

- Python  
- TensorFlow / Keras  
- NumPy  
- Pandas  
- Matplotlib  
- scikit-learn  

---

## 📁 Project Structure

```bash
SCT_ML_4/
├── sign_mnist_train.csv          # Training dataset
├── sign_mnist_test.csv           # Testing dataset
├── hand_gesture_model.ipynb      # CNN training notebook
├── images/
│   ├── sample_colored.png
│   ├── sample_grayscale.png
│   └── gesture_reference.png
├── requirements.txt
└── README.md
```

---

## 🧠 Model Architecture

```text
Input Layer: (28 x 28 x 1)
↓
Conv2D (32 filters, 3x3) → ReLU → MaxPooling2D (2x2)
↓
Conv2D (64 filters, 3x3) → ReLU → MaxPooling2D (2x2)
↓
Flatten
↓
Dense (128) → ReLU
↓
Dropout (0.2)
↓
Dense (25) → Softmax
```

- **Loss Function**: Sparse Categorical Crossentropy  
- **Optimizer**: Adam  
- **Evaluation Metric**: Accuracy

---

## 🔠 Label Mapping

| Label | Gesture |
|-------|---------|
| 0     | A       |
| 1     | B       |
| 2     | C       |
| 3     | D       |
| 4     | E       |
| 5     | F       |
| 6     | G       |
| 7     | H       |
| 8     | I       |
| 9     | K       |
| 10    | L       |
| 11    | M       |
| 12    | N       |
| 13    | O       |
| 14    | P       |
| 15    | Q       |
| 16    | R       |
| 17    | S       |
| 18    | T       |
| 19    | U       |
| 20    | V       |
| 21    | W       |
| 22    | X       |
| 23    | Y       |
| 24    | Z       |

⚠️ Gesture `J` is excluded because it requires motion.

---

## 📊 Results

- ✅ **Training Accuracy**: ~99%  
- ✅ **Test Accuracy**: ~95%  
- 📈 Accuracy and loss graphs plotted over epochs  
- 🖼️ Sample test predictions visualized using `matplotlib`

---

## 🖼️ Sample Visuals

| Colored Input | Grayscale | Gesture Chart |
|---------------|-----------|----------------|
| ![Color](images/sample_colored.png) | ![Gray](images/sample_grayscale.png) | ![Chart](images/gesture_reference.png) |

---

## 🔧 Installation

Use a virtual environment or Google Colab to install:

```bash
pip install -r requirements.txt
```

---

## 📦 requirements.txt

```txt
tensorflow
numpy
pandas
matplotlib
scikit-learn
```

---

## 🚀 Future Scope

- Real-time detection using OpenCV and webcam  
- Web interface with Streamlit or Flask  
- More robust datasets and multi-label support

---

## 📄 License

MIT License — for academic and educational use.

---

## 🙌 Acknowledgments

- Dataset Source: Kaggle CSV (Static gesture data)
- Training Done on: Google Colab

---

> 🔗 GitHub Repo: [`rehaan1603/SCT_ML_4`](https://github.com/rehaan1603/SCT_ML_4)
