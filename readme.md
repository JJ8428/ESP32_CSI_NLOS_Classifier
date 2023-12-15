# ESP32 NLOS Classifier

## Overview

ESP32 NLOS Classifier is a binary classifier designed to analyze WiFi Channel State Information (CSI) data in the time, fast time, and frequency domains. The primary objective is to extract features that effectively distinguish between Line-of-Sight (LOS) and Non-Line-of-Sight (NLOS) scenarios. This project incorporates adapted features from previous NLOS CSI classifiers, experiments with features used for Ultra-Wideband (UWB), and introduces new features inspired by intuition gained during research.

## Features

- Analysis of WiFi CSI data in time, fast time, and frequency domains.
- Adaptation of features from previous NLOS CSI classifiers.
- Experimentation with features used for UWB.
- Introduction of new features developed from research intuition.

## Dataset

The repository includes a comprehensive CSI dataset featuring numerous examples of both LOS and NLOS cases. This dataset serves as the foundation for training and evaluating the ESP32 NLOS Classifier.

## Classifiers

### Support Vector Machine (SVM) Binary Classifier

An implementation of an SVM binary classifier is provided, leveraging the extracted features to distinguish between LOS and NLOS scenarios.

### Neural Network (NN) Binary Classifier

The repository includes a Neural Network (NN) binary classifier that employs deep learning techniques for analyzing CSI data and making predictions regarding NLOS classification.

## Real-Time Application

Explore a real-time application of the NN classifier, demonstrating its functionality in practical scenarios.

## Results

The results of the classifiers, including accuracy, precision, recall, and other relevant metrics, can be found [here](insert-link-here).

## Plots

Discover a collection of plots illustrating the effectiveness of each feature in distinguishing between LOS and NLOS scenarios.

## Data Collection

The CSI data is collected using the EspressIf CSI script, available [here](https://github.com/espressif/esp-csi/tree/master/examples/get-started).

## Espressif IDF

Ensure you have the Espressif IoT Development Framework (IDF) installed before using this repository. You can find installation instructions for IDF [here](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/).

## Research Paper

For a detailed paper on the research behind this repository, please refer to the following [Google Document](https://docs.google.com/document/d/1cXvi47HVLpnSG2Ms7i4oNRthOpzG6jbhIWm3wank2Gg/edit?usp=sharing).
