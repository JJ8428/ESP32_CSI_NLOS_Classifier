# 🚀 ESP32 NLOS Classifier for Indoor Localization

Indoor localization continues to gain traction as smart home applications and resource management technologies grow. A common approach is to use **WiFi Fine Timing Measurement (FTM)** for indoor localization through multiple access points. However, **WiFi FTM-based localization** often suffers from multipath-induced signal distortions, significantly degrading performance, especially in environments with obstructions.

**This project introduces a robust Line-of-Sight (LOS) detector** that leverages WiFi Channel State Information (CSI) and Channel Impulse Response (CIR) data to determine whether a user has a clear LOS to an access point. 🌐

---

## 🔍 Motivation

WiFi FTM localization works well when there are no obstructions between users and access points. However, **multipath signal distortions** can degrade accuracy in complex environments. To address this, it's crucial to determine the LOS condition for each access point, which is where the need for a robust **LOS/NLOS (Non-Line-of-Sight) classifier** comes in. 🎯

---

## 📊 Dataset Overview

The repository contains a **comprehensive CSI dataset** with examples of both LOS and NLOS cases. This dataset serves as the foundation for training and evaluating the ESP32 NLOS Classifier.

### Key Features:
- 📡 WiFi CSI and CIR data
- 🏷️ Labeled LOS and NLOS examples

---

## 🤖 Model Overview

Two models are used to classify whether the connection between a user and a router is LOS or NLOS based on WiFi CSI and CIR data:

1. **Support Vector Machine (SVM)**  
   - 📈 Accuracy: 79%

2. **Neural Network**  
   - 💯 Accuracy: 86%

For more detailed information on model performance and dataset features, check out the [project report](https://docs.google.com/document/d/1cXvi47HVLpnSG2Ms7i4oNRthOpzG6jbhIWm3wank2Gg/edit?tab=t.0). 📄

---

## 🛠️ Usage

The project folders `NLOS_classification` and `board_program/ftm_ESP` contain the necessary scripts to run the classification.

### Steps to run the project:
1. Set up two EspressIf boards with proper WiFi antennas.
2. Upload the **FTM Initiator** and **FTM Responder** code using the EspressIf IDE.
3. To perform real-time classification, run `realtime_NLOS_clf.py` on a computer or laptop connected to the board with the FTM initiator code.

### ⚠️ Things to be aware of:
- **Potential Issue**: The real-time classifier currently runs on a single thread. Ideally, two threads should be implemented: 
  - One thread should handle reading and updating the CSI/CIR data.
  - The other should handle classification using the model.
  
  With one thread, WiFi CSI data frames can be missed during inference, causing potential data hiccups. This is an improvement that can be easily addressed. 🔄

---

## 🌱 Future Extensions

Future improvements to the project could include:
- Recording a dataset with distance...
- Going back and adding more comments...
- Developing a **second classifier** to predict the **distance** between two WiFi routers based on WiFi CSI data. 📏📶
