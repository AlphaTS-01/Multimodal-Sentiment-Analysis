# Multimodal Sentiment Analysis

A comprehensive machine learning project that combines text and image data for **multiclass sentiment analysis** using **early fusion** techniques. This project implements a multimodal deep learning model to analyze memes and social media content across multiple dimensions: humor, sarcasm, offensiveness, motivational content, and overall sentiment.

---

## 🎯 Project Overview

This project uses **early fusion** for multimodal sentiment analysis, where text (OCR-extracted and corrected) and image features are **concatenated and processed jointly**. It combines the strength of **BERT** for text and **Vision Transformer (ViT)** for images.

### 🔑 Key Features

- **Multimodal Architecture:** Combines text and image data  
- **Multi-task Learning:** Predicts 5 sentiment dimensions simultaneously  
- **State-of-the-Art Models:** BERT for text, ViT for images  
- **Real-Time Web App:** Deployable with Streamlit  
- **Comprehensive Analysis:** Data exploration, visualization, and evaluation

---

![Sample](https://github.com/user-attachments/assets/45c68031-5146-4a82-8358-c0716dc37086)

## 📊 Tasks & Classes

| Task | Classes | Labels |
|------|---------|--------|
| **Humor** | 4 | `not_funny`, `funny`, `very_funny`, `hilarious` |
| **Sarcasm** | 4 | `not_sarcastic`, `general`, `twisted_meaning`, `very_twisted` |
| **Offensive** | 4 | `not_offensive`, `slight`, `very_offensive`, `hateful_offensive` |
| **Motivational** | 2 | `not_motivational`, `motivational` |
| **Overall Sentiment** | 5 | `very_negative`, `negative`, `neutral`, `positive`, `very_positive` |

---

## 🏗️ Architecture

**Model Components:**  
1. **Text Encoder:** BERT (`bert-base-uncased`), output reduced to 128-d  
2. **Image Encoder:** Vision Transformer (ViT), output 128-d  
3. **Fusion:** Concatenation → 256-d combined features  
4. **Heads:** 5 task-specific classifiers

**Training Details:**  
- **Optimizer:** Adam (`lr=1e-3`)  
- **Loss:** CrossEntropy per task  
- **Device:** CUDA if available

---

## 📁 Dataset Structure

```plaintext
Multimodal_dataset_assignment3/
├── labels.csv         # Text & labels
└── images/            # All meme images
    ├── image_1.jpg
    ├── image_2.jpeg
    └── ...
```
### Dataset Features

- `text_ocr`: Raw OCR text  
- `text_corrected`: Manually corrected text  
- Image paths + sentiment labels in CSV  

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision transformers pandas numpy pillow scikit-learn matplotlib seaborn streamlit
```

### 2. Prepare Dataset

- Match the directory structure above  
- Update `data_path` and `image_folder` in the notebook or script  

### 3. Train the Model

```bash
jupyter notebook train.ipynb
```

### 4. Run Inference

```python
model = MultimodalModel(num_classes_list=[4, 4, 4, 2, 5])
model.load_state_dict(torch.load("Multimodal_Model.pth"))
model.eval()
```

---

## 🌐 Streamlit Web App

The trained model is deployed using **Streamlit**, allowing real-time predictions directly from the browser.

### 🔍 Features

- Upload memes or social media images  
- View OCR-extracted and corrected text  
- Instant predictions for:
  - Humor level
  - Sarcasm intensity
  - Offensive tone
  - Motivational nature
  - Overall sentiment  
- Visual display of predictions alongside the uploaded image  

### ▶️ Run Locally

```bash
streamlit run app.py
```

### 📦 App File Structure

```
app.py                           # Main Streamlit app
Multimodal_Model.pth             # Trained model weights
utils/
├── ocr_utils.py                 # Text extraction and correction
├── preprocessing.py             # Image transforms and tokenization
├── inference.py                 # Model loading and prediction logic
```

---

## 📈 Model Performance

- **Humor**: Multi-class accuracy  
- **Sarcasm**: Multi-class accuracy  
- **Offensive**: Multi-class accuracy  
- **Motivational**: Binary accuracy  
- **Sentiment**: Multi-class accuracy  

### Training Configuration

- **Epochs**: 10  
- **Batch Size**: 32  
- **Train/Test Split**: 70/30  
- **Image Size**: 224x224  
- **Text Max Length**: BERT default  

---

## 🔧 Code Structure

| Component | Description |
|----------|-------------|
| `MemotionDataset` | Custom PyTorch dataset class for loading and preprocessing |
| `MultimodalModel` | Core model architecture (BERT + ViT + fusion) |
| `train.ipynb` | Training loop with multi-task loss |
| `app.py` | Streamlit web application |
| `utils/` | Utility scripts for preprocessing, OCR, and inference |

---

## 📊 Data Analysis Features

- Class distribution plots  
- Word count per sentiment class  
- Missing value handling  
- Categorical to numerical label conversion  

---

## 🎨 Visualization

- Sentiment class frequencies  
- Box plots for text length  
- Training loss and accuracy curves  
- Per-task evaluation metrics  

---

## 💾 Model Persistence

- **File**: `Multimodal_Model.pth`  
- **Format**: PyTorch `state_dict`  
- **Usage**: Load via `MultimodalModel` class  

---

## 🔄 Future Enhancements

- [ ] Attention-based fusion mechanisms  
- [ ] Larger pre-trained models (e.g., ViT, RoBERTa)  
- [ ] Image and text data augmentation  
- [ ] Hyperparameter optimization  
- [ ] K-fold cross-validation  
- [ ] Additional metrics: F1, precision, recall  

---

## 📚 References

- Sanh, V. et al. *BERT: A distilled version of BERT* (2019)  
- He, K. et al. *Deep Residual Learning for Image Recognition* (2016)  
- Baltrusaitis, T. et al. *Multimodal Machine Learning: A Survey and Taxonomy* (2018)  

---

*This project represents a full-stack solution to multimodal sentiment analysis — from data preprocessing and model training to real-time deployment via Streamlit.*
