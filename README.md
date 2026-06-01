# PawClassify 🐾
Cat & Dog Image Classifier using CNN (MobileNetV2)

## Setup
pip install -r requirements.txt

## Download Model
Download cat_dog_best_model.h5 from:
[Google Drive link here]
Place it in the same folder as app.py.

## Run
streamlit run app.py

## Model
- Base: MobileNetV2 (ImageNet pretrained)
- Test Accuracy: 87.50%
- K-Fold Val: 91.51% ± 6.29%
- Dataset: 210 images (105 cats, 105 dogs)