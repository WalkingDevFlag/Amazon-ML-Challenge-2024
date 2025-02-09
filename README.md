# Amazon ML Challenge 2024 

A comprehensive pipeline for extracting product attributes from images using OCR and deep learning. This system combines PaddleOCR for text extraction, BERT for text encoding, and a fine-tuned BART model for attribute prediction.

## Key Features
- Dual Conda environment setup for OCR and ML workflows
- Automated image preprocessing with OpenCV CLAHE enhancement
- Hybrid text processing with PaddleOCR and BERT embeddings
- BART-based sequence-to-sequence model for attribute prediction
- Fuzzy matching post-processing for unit standardization
- GPU-accelerated processing with checkpointing

## Installation

### Prerequisites
- NVIDIA GPU with CUDA 11.8+
- Conda package manager
- Kaggle API credentials

```
# PaddleOCR Environment
conda create -n paddle_ocr python=3.10
conda activate paddle_ocr
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install paddlepaddle-gpu==2.5.1.post117 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
pip install pandas tqdm requests opencv-python

# Main ML Environment
conda create -n amazon_ml python=3.10
conda activate amazon_ml
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers pandas tqdm scikit-learn nltk fuzzywuzzy kagglehub
python -m nltk.downloader punkt
```

## Usage

### Workflow Pipeline
1. **Data Preparation**
   ```
   python Download_kagglehub.py
   python Linking.py
   ```

2. **OCR Processing**
   ```
   python Preprocessing.py
   ```

3. **Text Encoding**
   ```
   python Encoding.py
   python Cleaning.py
   ```

4. **Model Training**
   ```
   python Fine_Tuning.py
   ```

5. **Prediction & Correction**
   ```
   python Prediction.py
   python Unit_Correction.py
   ```

## Code Structure

| File | Purpose | Key Technologies |
|------|---------|-------------------|
| `Preprocessing.py` | Image enhancement & OCR | PaddleOCR, OpenCV |
| `Encoding.py` | Text embedding generation | BERT, PyTorch |
| `Fine_Tuning.py` | Model training | BART, HuggingFace |
| `Unit_Correction.py` | Output standardization | FuzzyWuzzy |

## Results

**Evaluation Metrics** (20 Epochs):
- Exact Match Accuracy: 78.42%
- BLEU-4 Score: 0.851

**Example Prediction:**
```
Input:  width | Product Width: 15.5inc
Output: 15.5 inch
```

## Directory Structure
```
Amazon ML Challenge 2024/
├── archive/
│   ├── images/           # Raw product images
│   └── dataset/         # CSV metadata files
├── outputs/
│   ├── processed/       # Cleaned datasets
│   └── predictions/     # Model outputs
└── model/               # Saved BART models
```

## License
MIT License
