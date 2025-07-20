# UTK-face

# **Age Detection System**

## **Overview**

This project predicts a person’s **age group** from their facial image using **deep learning**. It uses a **pre-trained MobileNetV2 model** with transfer learning for fast and accurate predictions. The model classifies faces into **five age groups**:

* **0:** 0–9 years (Child)
* **1:** 10–19 years (Teen)
* **2:** 20–39 years (Young Adult)
* **3:** 40–59 years (Middle-aged)
* **4:** 60+ years (Senior)

The system is designed for high performance and can be extended for real-time applications or integrated into analytics dashboards.

## **Features**

* Predicts the **age group** of a person from a facial image.
* Uses **transfer learning** with MobileNetV2 for efficient training.
* Works with the **UTKFace (Aligned) Dataset**.
* Achieves **83% accuracy** on validation data.
* Evaluates performance with:

  * **Accuracy**
  * **Precision, Recall, F1-Score**
  * **Confusion Matrix**
  * **Mean Absolute Error (MAE)**

## **requirements.txt**

```
torch
torchvision
opencv-python
numpy
scikit-learn
matplotlib
tqdm
```

## **Model Details**

* **Base Model**: MobileNetV2 (pre-trained on ImageNet)
* **Custom Layers**: Final classifier adjusted for 5 age groups
* **Dataset**: UTKFace (Cropped and Aligned)
* **Image Preprocessing**:

  * Resized to **160×160**
  * Normalized to \[-1, 1]

### **Performance**

* **Accuracy**: 83%
* **MAE**: 0.175 (very close predictions)

## **Project Files**

* `Age_Detection_UTK_5bins.ipynb` → Model training and evaluation
* `mobilenet_age_classifier_5bins.pth` → Trained model weights
* `requirements.txt` → Required libraries

## **Outputs**

* **Confusion Matrix** for 5 age groups
* **Classification Report** with precision, recall, F1-score
* **MAE** for error analysis

## **Future Improvements**

* Add **gender detection** along with age.
* Support **real-time video streams** using OpenCV.
* Optimize model for **edge devices and mobile deployment**.
* Expand to **exact age prediction** using regression.

