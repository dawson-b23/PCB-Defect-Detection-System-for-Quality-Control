# Dawson Burgess
# final project cs555 computer vision
# Final Project code 12/9
# topic - PCB board defect detection for maufacturing
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from tensorflow.keras import layers, models

## ## ## ## ##
## DATA PREP #
## ## ## ## ##


## importing and formatting data / making CSV for usage in program
def preparing_annotation_data():
    # this is how the data is structured
    dataStructure = {
        "xmin": [],
        "xmax": [],
        "ymin": [],
        "ymax": [],
        "class": [],
        "file": [],
        "width": [],
        "height": [],
    }
    # load all of the files
    data_files = []
    for path, subdirectory, file in os.walk("../../../PCB_DATASET/Annotations/"):
        for name in file:
            data_files.append(os.path.join(path, name))

    # extract relevant information needed for training from file
    # this i had to look up how to do, i've never worked with xml before
    for file in data_files:
        # print(file) # for debugging to make sure all Annotations are loading....

        # Parse the XML file
        tree = ET.parse(str(file))  # loads the file
        root = tree.getroot()  # gets the root or top of the file for parsing through it

        # Extract information from each object
        for xmlObject in root.findall("object"):
            # append repeated values
            dataStructure["file"].append(root.find("filename").text)
            dataStructure["width"].append(int(root.find(".//size/width").text))
            dataStructure["height"].append(int(root.find(".//size/height").text))

            # class name
            dataStructure["class"].append(xmlObject.find("name").text)

            # Extract bounding box coordinates
            xmin = int(xmlObject.find(".//bndbox/xmin").text)
            xmax = int(xmlObject.find(".//bndbox/xmax").text)
            ymin = int(xmlObject.find(".//bndbox/ymin").text)
            ymax = int(xmlObject.find(".//bndbox/ymax").text)

            dataStructure["xmin"].append(xmin)
            dataStructure["xmax"].append(xmax)
            dataStructure["ymin"].append(ymin)
            dataStructure["ymax"].append(ymax)

    # print for debug to make sure everything is parsed correctly
    # print(dataStructure)

    # now load the parsed data into a pandas file
    annotation_df = pd.DataFrame(dataStructure)
    print(annotation_df)

    annotation_df.to_csv("annotation_data_df.csv", sep=",")

    return annotation_df


## preps data for test/train
def preparing_train_test_data(annotation_df):
    # map classes to integers
    anomaly_type = {
        "missing_hole": 0,
        "mouse_bite": 1,
        "open_circuit": 2,
        "short": 3,
        "spur": 4,
        "spurious_copper": 5,
    }
    # split into train and test
    train, test = train_test_split(annotation_df, shuffle=True, test_size=0.2)
    print("Train: ", train.shape)
    print("Test: ", test.shape)

    # map classes to ints
    train["class"] = train["class"].map(anomaly_type)
    test["class"] = test["class"].map(anomaly_type)

    # handle NaN values
    train = train.dropna(subset=["class"])
    test = test.dropna(subset=["class"])

    # conver to int
    train["class"] = train["class"].astype(int)
    test["class"] = test["class"].astype(int)

    print(train.head())
    return train, test


# this is a helper function for getting the name out of the file name
#   e.g. -- 09_mouse_bite_90.jpg  >> Mouse_bite
#   essentially this is needed for getting the filepath name
def extract_and_format_name(filename):
    # extract needed parts with regex
    match = re.search(r"\d+_(.*?)_\d+\.", filename)
    if match:
        words = match.group(1)
        # split the words by underscores, capitalize the first letter, join words
        parts = words.split("_")
        formatted_name = parts[0].capitalize() + (
            "_" + "_".join(parts[1:]) if len(parts) > 1 else ""
        )
        return formatted_name

    return None  # return none if no pattern found


## preprocess images to highlight defects more clearly
def preprocess_image(image):
    # convert to HSV
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    # histogram equalization on the V channel
    v = cv.equalizeHist(v)
    hsv = cv.merge([h, s, v])
    # convert back to BGR
    preprocessed = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    # GB  to reduce noise
    preprocessed = cv.GaussianBlur(preprocessed, (3, 3), 0)
    return preprocessed


## ## ## ## ## ## ## ## ##
## DATA LOAD FOR MODELS ##
## ## ## ## ## ## ## ## ##


## tensorflow dataset preparation
def load_and_preprocess_image_tf(file_path, xmin, ymin, xmax, ymax):
    # load and decode image
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    # crop region of interest
    image = image[ymin:ymax, xmin:xmax]
    # resize and normalize
    image = tf.image.resize(image, [224, 224])
    image = image / 255.0
    return image


## data prep for tensorflow
def df_to_tf_dataset(df):
    # create tensors for file paths, labels, and boxes
    file_paths, labels, boxes = [], [], []
    for _, row in df.iterrows():
        directory_subtype = extract_and_format_name(row["file"])
        directory_path_name = f"../../../PCB_DATASET/images/{directory_subtype}/"
        image_path = os.path.join(directory_path_name, row["file"])
        file_paths.append(image_path)
        labels.append(row["class"])
        boxes.append([row["xmin"], row["ymin"], row["xmax"], row["ymax"]])

    file_paths = tf.convert_to_tensor(file_paths, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    boxes = tf.convert_to_tensor(boxes, dtype=tf.int32)

    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels, boxes))

    # map function to load and preprocess images
    def _load_fn(file_path, label, box):
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
        image = load_and_preprocess_image_tf(file_path, xmin, ymin, xmax, ymax)
        return image, label

    ds = ds.map(_load_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return ds


## loading images for sklearn usage - extracting features for classical ml models
def load_images_for_sklearn(df, target_size=(64, 64)):
    X, y = [], []
    for _, row in df.iterrows():
        directory_subtype = extract_and_format_name(row["file"])
        directory_path_name = f"../../../PCB_DATASET/images/{directory_subtype}/"
        image_path = os.path.join(directory_path_name, row["file"])
        image = cv.imread(image_path)
        if image is None:
            continue

        # crop and preprocess image
        xmin, xmax = row["xmin"], row["xmax"]
        ymin, ymax = row["ymin"], row["ymax"]
        cropped = image[ymin:ymax, xmin:xmax]
        cropped = preprocess_image(cropped)
        cropped = cv.cvtColor(cropped, cv.COLOR_BGR2RGB)
        cropped = cv.resize(cropped, target_size)
        cropped = cropped / 255.0
        features = cropped.flatten()
        X.append(features)
        y.append(row["class"])

    return np.array(X), np.array(y)


## ## ## ##
## MODELS #
## ## ## ##

## KERAS MODELS ##


def cnn_model(num_classes=6):
    model = models.Sequential(
        [
            layers.Conv2D(
                16, (3, 3), activation="relu", padding="same", input_shape=(224, 224, 3)
            ),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.summary()
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def build_deeper_cnn(num_classes=6):
    # a deeper cnn with more convolutional layers
    model = models.Sequential(
        [
            layers.Conv2D(
                32, (3, 3), activation="relu", padding="same", input_shape=(224, 224, 3)
            ),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.summary()
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


## ## ## ## ## ## ## ## ## ##
## VISUALIZATION FUNCTIONS ##
## ## ## ## ## ## ## ## ## ##


# print classification report only for classes present
def print_classification_metrics(y_true, y_pred, class_names):
    labels_present = sorted(list(set(y_true) | set(y_pred)))
    used_class_names = [class_names[i] for i in labels_present]
    print("Classification Report:")
    print(
        classification_report(
            y_true, y_pred, labels=labels_present, target_names=used_class_names
        )
    )


# plot confusion matrix for sklearn models
def plot_confusion_matrix(y_true, y_pred, class_names, model_name="model"):
    labels_present = sorted(list(set(y_true) | set(y_pred)))
    used_class_names = [class_names[i] for i in labels_present]
    cm = confusion_matrix(y_true, y_pred, labels=labels_present)
    plt.figure(figsize=(6, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=used_class_names,
        yticklabels=used_class_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"confusion_matrix_{model_name}.png")
    plt.close()


## collect predictions from the keras model
def collect_predictions(model, test_ds):
    y_true = []
    y_pred = []
    y_pred_proba = []
    for images, labels in test_ds:
        preds = model.predict(images)
        preds_cls = np.argmax(preds, axis=1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds_cls)
        y_pred_proba.extend(preds)
    return np.array(y_true), np.array(y_pred), np.array(y_pred_proba)


## evaluate model and visualize results
def evaluate_and_visualize_results(model, test_ds, class_names, model_name="model"):
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test Accuracy: {test_acc:.2f}")

    y_true, y_pred, y_pred_proba = collect_predictions(model, test_ds)

    labels_present = sorted(list(set(y_true) | set(y_pred)))
    used_class_names = [class_names[i] for i in labels_present]

    print("Classification Report:")
    print(
        classification_report(
            y_true, y_pred, labels=labels_present, target_names=used_class_names
        )
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels_present)
    print("Confusion Matrix:")
    print(cm)

    plt.figure(figsize=(6, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=used_class_names,
        yticklabels=used_class_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"confusion_matrix_{model_name}.png")
    plt.close()

    return y_true, y_pred, y_pred_proba, labels_present, used_class_names


## plot training history for keras models
def plot_training_history(history, model_name="model"):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, "bo-", label="Training accuracy")
    plt.plot(epochs, val_acc, "ro-", label="Validation accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, "bo-", label="Training loss")
    plt.plot(epochs, val_loss, "ro-", label="Validation loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.savefig(f"training_history_{model_name}.png")
    plt.close()


## show misclassified examples
def show_misclassified_samples(model, test_ds, class_names, model_name="model"):
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []

    for images, labels in test_ds:
        preds = model.predict(images)
        preds_cls = np.argmax(preds, axis=1)
        for img, true_label, pred_label in zip(images, labels.numpy(), preds_cls):
            if true_label != pred_label:
                misclassified_images.append(img)
                misclassified_labels.append(true_label)
                misclassified_preds.append(pred_label)
        if len(misclassified_images) > 9:
            break

    if len(misclassified_images) == 0:
        print("No misclassified samples found.")
        return

    plt.figure(figsize=(10, 10))
    for i in range(min(9, len(misclassified_images))):
        plt.subplot(3, 3, i + 1)
        img = misclassified_images[i].numpy()
        plt.imshow(img)
        plt.title(
            f"True: {class_names[misclassified_labels[i]]}, Pred: {class_names[misclassified_preds[i]]}"
        )
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"misclassified_examples_{model_name}.png")
    plt.close()


## plot roc curves
def plot_roc_curves(y_true, y_pred_proba, class_names, model_name="model"):
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title("One-vs-Rest ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(f"roc_curves_{model_name}.png")
    plt.close()


## plot precision-recall curves
def plot_precision_recall_curves(y_true, y_pred_proba, class_names, model_name="model"):
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(
            y_true_bin[:, i], y_pred_proba[:, i]
        )
        ap = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])
        plt.plot(recall, precision, label=f"{class_names[i]} (AP = {ap:.2f})")
    plt.title("One-vs-Rest Precision-Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.savefig(f"precision_recall_curves_{model_name}.png")
    plt.close()


## visualize feature space using pca
def visualize_feature_space(X, y, class_names, model_name="model"):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="Set1", alpha=0.7)
    plt.title("PCA Visualization of Feature Space")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    handles, _ = scatter.legend_elements()
    plt.legend(handles, class_names)
    plt.savefig(f"{model_name}_pca.png")
    plt.close()


## ## ## ## ## ## ## ##
## PIPELINE FUNCTIONS #
## ## ## ## ## ## ## ##


## run a keras model pipeline: train, evaluate, visualize, save
def run_keras_model(model_fn, train, test, class_names, model_name="model"):
    # create tf datasets
    train_ds = df_to_tf_dataset(train).shuffle(500).batch(32).prefetch(tf.data.AUTOTUNE)
    test_ds = df_to_tf_dataset(test).batch(32).prefetch(tf.data.AUTOTUNE)

    # build and train model
    model = model_fn(num_classes=len(class_names))
    history = model.fit(train_ds, validation_data=test_ds, epochs=10)
    plot_training_history(history, model_name=model_name)

    # evaluate and visualize
    y_true, y_pred, y_pred_proba, labels_present, used_class_names = (
        evaluate_and_visualize_results(
            model, test_ds, class_names, model_name=model_name
        )
    )

    # Remap classes if needed
    label_map = {lb: i for i, lb in enumerate(labels_present)}
    y_true_remap = np.array([label_map[yt] for yt in y_true])
    y_pred_proba_remap = (
        y_pred_proba[:, labels_present]
        if y_pred_proba.shape[1] > len(labels_present)
        else y_pred_proba
    )

    # if all classes present, plot roc and pr curves
    if len(labels_present) == len(class_names):
        # If all classes present, plot ROC and PR curves
        plot_roc_curves(
            y_true_remap, y_pred_proba_remap, used_class_names, model_name=model_name
        )
        plot_precision_recall_curves(
            y_true_remap, y_pred_proba_remap, used_class_names, model_name=model_name
        )

    show_misclassified_samples(model, test_ds, class_names, model_name=model_name)

    # Save model and training history
    model.save(f"{model_name}.h5")
    with open(f"{model_name}_history.json", "w") as f:
        json.dump(history.history, f)

    return model


## run sklearn models for comparison
def run_sklearn_models(train, test, class_names):
    # load features for sklearn
    X_train, y_train = load_images_for_sklearn(train)
    X_test, y_test = load_images_for_sklearn(test)

    if X_train.size == 0 or X_test.size == 0:
        print(
            "No training or testing data loaded. Check image paths and data integrity."
        )
        return

    # define models to run
    models_dict = {
        "SVM (RBF Kernel)": SVC(kernel="rbf", C=1.0),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "Logistic Regression": LogisticRegression(max_iter=1000),
    }

    # train and evaluate each model
    for model_name, model in models_dict.items():
        print(f"- - - - - {model_name} - - - - - ")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Accuracy: {acc:.2f}")
        print_classification_metrics(y_test, preds, class_names)
        plot_confusion_matrix(y_test, preds, class_names, model_name=model_name)
        print("\n")

    # visualize feature space with pca
    visualize_feature_space(X_train, y_train, class_names, model_name="sklearn results")


## ## ## #
## MAIN ##
## ## ## #
if __name__ == "__main__":
    # open a file named model_output for saving output to
    # remove comment to write to an output file
    sys.stdout = open("model_output_summary", "w")

    # prepare annotation data
    df = preparing_annotation_data()
    class_names = [
        "missing_hole",
        "mouse_bite",
        "open_circuit",
        "short",
        "spur",
        "spurious_copper",
    ]
    # train/test split
    train, test = preparing_train_test_data(df)

    # run baseline keras cnn
    print("- - - - - - Running Baseline Keras CNN Model - - - - - - ")
    run_keras_model(cnn_model, train, test, class_names, model_name="baseline_cnn")

    print("- - - - - -  Running Deeper CNN Model - - - - - - ")
    run_keras_model(build_deeper_cnn, train, test, class_names, model_name="deeper_cnn")

    print("- - - - - - Running Scikit-Learn Models - - - - - - ")
    run_sklearn_models(train, test, class_names)

    # close and save output to file
    # remove comment to write to an output file
    sys.stdout.close()
