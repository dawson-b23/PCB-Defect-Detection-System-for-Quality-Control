# PCB Defect Detection System for Quality Control

**Author:** Dawson Burgess, Computer Science Department, University of Idaho, Moscow, ID, United States  
**Email:** [burg1648@vandals.uidaho.edu](mailto:burg1648@vandals.uidaho.edu), [dawsonsburgess@gmail.com](mailto:dawsonsburgess@gmail.com)

## Overview

This repository contains the code, proposal, and research paper for the project "PCB Defect Detection System for Quality Control," developed as part of CS555: Machine Vision at the University of Idaho. The system uses computer vision and deep learning to detect manufacturing defects in Printed Circuit Boards (PCBs), such as missing holes, broken traces, and soldering issues. It compares Convolutional Neural Networks (CNNs) with traditional machine learning models (SVM, Random Forest, Logistic Regression), achieving up to 96% accuracy with a deeper CNN.

### Key Features

- **Data Preprocessing**: Parses XML annotations, preprocesses images with histogram equalization and Gaussian blur.
- **Models**: Baseline CNN, Deeper CNN, and classical ML models for defect classification.
- **Visualization**: Confusion matrices, ROC/PR curves, PCA feature space, and misclassified samples.
- **Dataset**: Utilizes the PCB Defect Dataset from Kaggle.

## Project Structure

- **`final_project_cs555_computer_vision.py`**: Main script for data preparation, model training, and evaluation.
- **`PCB_Defect_Detection_System_for_Quality_Control.pdf`**: Research paper detailing methodology and results.
- **`Final_Project_Proposal_1.pdf`**: Initial project proposal outlining goals and approach.

## Dataset

The **PCB Defect Dataset** is sourced from Kaggle and available [here](https://www.kaggle.com/datasets/akhatova/pcb-defects). It includes high-resolution PCB images and XML annotations for six defect types: missing hole, mouse bite, open circuit, short, spur, and spurious copper. The dataset was parsed into a CSV file for streamlined processing, with bounding box coordinates used to crop defect regions.

**Preprocessing Steps**:

- Parsed XML annotations into a pandas DataFrame.
- Cropped defect regions, resized to 224x224 pixels, and normalized to [0, 1].
- Applied histogram equalization and Gaussian blur to enhance defect visibility.

## Methodology

### Data Preparation

- Converted XML annotations to a CSV format with bounding box coordinates and class labels.
- Split data into 80% training and 20% testing sets.
- Preprocessed images for CNNs (TensorFlow) and feature extraction for classical ML (scikit-learn).

### Models

1. **Baseline CNN**:
   - Architecture: 2 Conv2D layers, 2 MaxPooling2D layers, Dense layers (128, 6).
   - Performance: 94% accuracy.
2. **Deeper CNN**:
   - Architecture: 6 Conv2D layers, 3 MaxPooling2D layers, Dense layers (256, 6).
   - Performance: 96% accuracy.
3. **Classical ML Models**:
   - SVM (RBF Kernel): 90% accuracy.
   - Random Forest: 91% accuracy.
   - Logistic Regression: 82% accuracy.

### Visualization Techniques

- **Confusion Matrices**: Assessed class-wise performance (e.g., `confusion_matrix_deeper_cnn.png`).
- **ROC/PR Curves**: Evaluated model robustness (`roc_curves_deeper_cnn.png`).
- **Training History**: Plotted accuracy/loss trends (`training_history_baseline_cnn.png`).
- **PCA Visualization**: Explored feature space separability (`sklearn_results_pca.png`).
- **Misclassified Samples**: Highlighted challenging cases (`misclassified_examples_deeper_cnn.png`).

## Results

- **Deeper CNN**: Top performer with 96% accuracy, high precision/recall across all classes.
- **Baseline CNN**: Achieved 94% accuracy, slightly less robust than the deeper model.
- **Classical ML**: Random Forest (91%) outperformed SVM (90%) and Logistic Regression (82%), but lagged behind CNNs.
- **Key Insight**: CNNs excelled at learning subtle defect patterns, with the deeper model showing balanced performance across classes like spur and spurious copper.

## Challenges and Limitations

- **Visual Similarity**: Defects like spur vs. spurious copper were harder to distinguish.
- **Data Variability**: Limited exploration of lighting/design variations.
- **Time Constraints**: Shifted focus from traditional image processing to end-to-end deep learning.

## Future Work

- Enhance preprocessing with advanced augmentation (e.g., rotations, brightness adjustments).
- Integrate attention mechanisms or real-time detection with live feeds.
- Expand dataset with diverse PCB designs for better generalization.

## Installation and Usage

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/pcb-defect-detection.git
   ```

2. Download the dataset:

    Access the PCB Defect Dataset [here](https://www.kaggle.com/datasets/akhatova/pcb-defects).
    Place it in the data/ directory or update file paths in final_project_cs555_computer_vision.py.

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the script:

   ```bash
   python final_project_cs555_computer_vision.py
   ```

5. View outputs: Check figures/ for visualizations and model_output_summary for metrics.

## License

This project is licensed under the MIT License.

## Contact

For inquiries, contact Dawson Burgess at <burg1648@vandals.uidaho.edu> or <dawsonsburgess@gmail.com>.

## Citation

If you use this project, please cite:
Dawson Burgess. (2024). PCB Defect Detection System for Quality Control. University of Idaho.
