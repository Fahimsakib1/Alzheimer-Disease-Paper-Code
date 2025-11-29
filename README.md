# Alzheimer’s Disease Stage Prediction — Research
Alzheimer’s Disease (AD) is a progressive neurological disorder that leads to declining memory and cognitive function. Early diagnosis is essential, as timely intervention can slow disease progression and improve patient outcomes. This study reviews and experiments with unimodal and multimodal Machine Learning (ML) and Deep Learning (DL) approaches for predicting AD stages.

## 1. Abstract
Alzheimer's Disease (AD) is a brain disorder causing gradual memory and cognition problems, where early diagnosis is crucial for effective intervention. The objective of this study is to review the unimodal and multimodal approaches applying Machine Learning (ML) and Deep Learning (DL) for prediction of AD stages. In our experiments, unimodal models analyzing single data types showed varying accuracy, with Electronic Health Record (EHR) and Single Nucleotide Polymorphism (SNP) classifiers achieving 56\% to 76\% while CNN-based models reached 98\%. Multimodal models integrating EHR, SNP, and imaging data significantly improved classification performance. Among ML models, Random Forest achieved 93\% accuracy while XGBoost and LightGBM performed slightly better at 95\% and 96\%, respectively. CNN-based architectures showed superior results, with Deep CNN achieving 95\% accuracy. Despite integration challenges, multimodal fusion enhances AD diagnosis, showing the impact of deep learning models in healthcare data analysis.

## 2. Introduction
Alzheimer’s Disease (AD) is a progressive neurodegenerative disorder that leads to memory loss, cognitive decline, and behavioral changes, affecting millions worldwide. Early symptoms often include difficulties in reasoning, word finding, and spatial awareness, eventually progressing to severe impairment in daily activities. Since AD is influenced by genetic, environmental, and lifestyle factors, early and accurate diagnosis is essential for improving patient outcomes.

Recent research shows that multimodal approaches—integrating neuroimaging, genetic data, clinical records, and cognitive assessments—significantly improve prediction performance, particularly when identifying the transition from Mild Cognitive Impairment (MCI) to AD. However, these methods come with challenges such as high computational cost, complex preprocessing, and reduced interpretability. In contrast, unimodal models are simpler and computationally efficient but often fail to capture the full complexity of AD, leading to lower accuracy.

To address these gaps, this study proposes a framework that enhances both interpretability and computational efficiency in multimodal AD diagnosis. By incorporating data from IoT sensors, neuroimaging, and additional clinical features, the framework supports robust binary and multi-class classification while reducing model complexity. This approach aims to provide more accurate, scalable, and clinically actionable diagnostic tools for real-world healthcare applications.

![1](https://github.com/user-attachments/assets/073a3066-b8b8-492b-97bc-129dcb6f2c43)
*Figure 1: Brain changes in AD: Normal vs Alzheimer's brain.*

## 3. Research Background
Recent research on Alzheimer’s Disease (AD) prediction has focused heavily on **multimodal machine learning and deep learning models**, integrating imaging, genetic, and clinical data. Studies using ADNI and OASIS datasets demonstrate that hybrid approaches combining MRI, PET, SNPs, retinal biomarkers, and cognitive scores consistently outperform unimodal models. Techniques such as DenseNet3D with SVR, multimodal CNNs, weighted SVM fusion, and ensemble classifiers have achieved strong results, including accuracies between **89% and 97%**, AUC values up to **0.97**, and significant reductions in sample size requirements for clinical trials. Many works also highlight the benefit of explainable AI (XAI) using methods like LIME, SHAP, and GradCAM to improve medical trust and transparency.

Additional studies show the growing value of **speech-based biomarkers**, longitudinal data modeling, and fusion frameworks that combine neuroimaging with genomic or clinical features. CNN and transfer-learning models trained on MRI often reach above **95% accuracy**, while multimodal models using PET, MRI, and demographic data improve conversion prediction from MCI to AD. Despite promising performance, common limitations include small datasets, restricted generalizability, imbalance issues, and high computational cost. Overall, the literature indicates that **multimodal fusion and deep learning architectures significantly enhance AD diagnostic accuracy**, surpassing traditional machine learning and single-modality approaches.

| **Dataset** | **Models Used**              | **Key Results**                                                           |
|-------------|------------------------------|---------------------------------------------------------------------------|
| ADNI        | 3D CNN                       | Accuracy: **94%** (AD vs NC), **74.5%** (multi-class)                     |
| ADNI        | SVM, RF, KNN, LR, DT         | RF: **90.51%** accuracy, 90.69% precision, 90.51% recall, 90.41% F1-score |
| ADNI        | Decision Tree, Fuzzy Systems | Accuracy: **93.95%** for AD prediction, **87.09%** for transition         |
| ADNI        | RNN                          | MCI → AD accuracy **81%**, AUC **0.86**                                   |
| ADNI        | MHS, Brain Atrophy           | Reduced clinical trial sample size by **67%**                             |
| OASIS       | RF, SVM, DT, etc.            | RF achieved **92.14%** accuracy                                           |
| OASIS-3     | CNN                          | Image-based accuracy **71.43%**, hybrid model **74.89%**                  |
| OASIS       | LR, DT, RF, SVM, AdaBoost    | RF achieved **86.8%** accuracy, 80% recall, AUC **0.872**                 |
| ADNI        | Deep Learning Models         | Highest accuracy **89%**                                                  |


## 4. Methodological Framework
Our framework integrates multimodal data from IoT devices and clinical sources, including imaging and text-based data, to enhance Alzheimer's Disease (AD) prediction using machine learning (ML) and deep learning (DL) techniques. The study utilizes the OASIS-3 and ADNI datasets, which provide comprehensive neuroimaging, cognitive, and biomarker data for individuals across different stages of AD.

![Framework_Workflow](https://github.com/user-attachments/assets/f6d2dd3a-3e0c-4e97-a676-6a5d547c69a7)
*Figure 2: Step-by-step workflow for detecting Alzheimer’s disease progression using a multimodal fusion-
based data integration and analysis framewor.*

### 4.1 Data Acquisition and Preparation
We collect clinical metrics (e.g., heart rate, blood pressure), imaging data (MRI, PET), and genetic data. Test and training sets are generated from ADNI and OASIS-3, ensuring a comprehensive view of disease progression.

### 4.2 Data Preprocessing and Augmentation
* Image Conversion: MRI images converted from DICOM to JPEG, resized, and converted to 3-channel RGB.
* Augmentation: Flipping, rotation, zoom, shearing, elastic deformations, and contrast adjustments increase dataset diversity.
* Optimization: Histogram equalization and combined augmentation improve model performance, robustness, and generalization.
  
### 4.3 Feature Extraction and Classification
* CNNs and multimodal deep learning models process MRI, genetic, and clinical data.
* Two parallel pathways: one for non-imaging data and one for MRI features.
* Features merged and classified through fully connected layers for multi-class (CN, MCI, FTD, AD) and binary (AD vs CN) classification.
![modelWorkflow](https://github.com/user-attachments/assets/7bf0c456-682e-489e-9487-d61fb2e7fc06)
*Figure 3: Workflow of the proposed framework with model architecture.*

### 4.4 Web Integration
An interactive web interface allows users to upload images and receive AD predictions. Text-based data is also analyzed and combined with imaging predictions for comprehensive diagnostics.

### 4.5 Evaluated Techniques
Performance of different models is compared based on accuracy, sensitivity, specificity, and robustness, including:

#### 4.5.1 CNN
* Conv2D and MaxPooling layers extract MRI features.
* Dense layers classify AD stages (AD, CI, CN) with softmax output.
<img width="553" height="486" alt="Confusion_Matrix_AD_CI_CN" src="https://github.com/user-attachments/assets/7aaa3520-861a-43b2-bb66-72ed884b5d62" />

#### 4.5.2 XGBoost
* Gradient-boosted decision tree model for multi-class AD prediction.
* Optimized with L1/L2 regularization and early stopping.

#### 4.5.3 Random Forest
* Ensemble of decision trees with optimized hyperparameters.
* Achieves 93% accuracy for binary AD classification.

#### 4.5.4 LightGBM
* Gradient boosting framework optimized for speed and accuracy.
* Balanced predictions between AD and non-AD cases.

#### 4.5.5 EfficientNetB3
* Pre-trained on ImageNet, adapted with dense layers, dropout, and softmax for multi-class AD classification.

#### 4.5.6 Decision Tree
* Three-class classification (AD, CN, FTD) with max depth control.
* Achieved 86% overall accuracy.

#### 4.5.7 VGG16
* Pre-trained on ImageNet and fine-tuned for AD stages.
* Features frozen for stable training, softmax used for classification.

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/3abda287-dee3-46e0-8150-5133feff9489" width="500"><br>
      <sub>Confusion Matrix XGBoost</sub>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/9c60ddd5-91f3-4ed0-ab97-da2e50e7f3f2" width="500"><br>
      <sub>Confusion Matrix RF</sub>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/09d0a027-c9e3-479f-b015-a9229d241fd9" width="500"><br>
      <sub>Confusion Matrix LGBM</sub>
    </td>
  </tr>
</table>

### Alzheimer's Disease (CNN Demo Version)

```python
for idx in range(16):
    image = dataset_X[idx]
    plt.subplot(4, 4, idx + 1)
    plt.title(my_labels[dataset_y[idx]])
    plt.gray()
    plt.imshow(image)
    plt.tight_layout()
plt.show()

new_X, new_y = [], []
max_limit = 3000
X_cat0, X_cat1, X_cat2 = dataset_X[dataset_y == 0], dataset_X[dataset_y == 1], dataset_X[dataset_y == 2]

for lbl, X_group in zip(my_labels.keys(), [X_cat0, X_cat1, X_cat2]):
    cnt = 0
    for im in X_group:
        if cnt > max_limit - label_counts[lbl]:
            break
        for _ in range(2):
            aug_im = augmentation_pipeline(im)
            new_X.append(aug_im)
            new_y.append(lbl)
            cnt += 1

new_X, new_y = np.array(new_X), np.array(new_y)
print(new_X.shape, new_y.shape)

# Splitting dataset
X_tv, X_tt, y_tv, y_tt = train_test_split(dataset_X, dataset_y, test_size=0.15, stratify=dataset_y)
X_tr, X_val, y_tr, y_val = train_test_split(X_tv, y_tv, test_size=0.15, stratify=y_tv)

print(X_tr.shape, X_val.shape, X_tt.shape)
print(f"Total: {dataset_X.shape[0]}, Train: {X_tr.shape[0]}, Val: {X_val.shape[0]}, Test: {X_tt.shape[0]}")

# Building the model
cnn_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(200, (3,3), activation="relu", input_shape=X_tr.shape[1:]),
    tf.keras.layers.MaxPooling2D((3,3)),
    tf.keras.layers.Conv2D(100, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D((3,3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax")
])

for i, layer in enumerate(cnn_model.layers):
    print(f"Layer {i}:", layer.name, layer.output_shape, layer.count_params())

# Callbacks
cb_checkpoint = tf.keras.callbacks.ModelCheckpoint("my_model.h5", save_best_only=True)
cb_early = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

history = cnn_model.fit(X_tr, y_tr, epochs=100, callbacks=[cb_checkpoint, cb_early], validation_data=(X_val, y_val))

results_df = pd.DataFrame(history.history)
results_df.head()
```

### Alzheimer's Disease with (EfficientNetB3 Demo Version)
```python
data_dir = "/path/to/data"  # placeholder
filepaths = ["dummy_path_1", "dummy_path_2", "..."]
labels = ["Class1", "Class2", "Class3"]
df = pd.DataFrame({"filepaths": filepaths, "labels": labels})

train_df, valid_df, test_df = df, df, df  # dummy split
print("Dataset split complete. Train/Val/Test sizes:", len(train_df), len(valid_df), len(test_df))

def dummy_preprocess(img):
    return img  # placeholder

tr_gen = ImageDataGenerator(preprocessing_function=dummy_preprocess, horizontal_flip=True)
ts_gen = ImageDataGenerator(preprocessing_function=dummy_preprocess)

train_gen = tr_gen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels',
                                       target_size=(224,224), color_mode='rgb', class_mode='categorical', batch_size=32)
valid_gen = ts_gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels',
                                       target_size=(224,224), color_mode='rgb', class_mode='categorical', batch_size=32)
test_gen = ts_gen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels',
                                      target_size=(224,224), color_mode='rgb', class_mode='categorical', batch_size=32, shuffle=False)
img_shape = (224,224,3)
class_count = 3  # dummy number of classes

base_model = tf.keras.applications.EfficientNetB3(include_top=False, weights="imagenet", input_shape=img_shape, pooling='max')

model = Sequential([
    base_model,
    BatchNormalization(),
    Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.016), activity_regularizer=regularizers.l1(0.006)),
    Dropout(0.45, seed=123),
    Dense(class_count, activation='softmax')
])

model.compile(optimizer=Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch+1}: Dummy callback reporting. Accuracy: {logs.get('accuracy', 0):.2f}")

callbacks = [MyCallback()]

history = model.fit(train_gen, epochs=5, validation_data=valid_gen, callbacks=callbacks, verbose=0)
print("Training placeholder complete.")

train_score = [0.0, 0.0]  # placeholder
valid_score = [0.0, 0.0]
test_score = [0.0, 0.0]

print(f"Train Loss: {train_score[0]}, Accuracy: {train_score[1]}")
print(f"Validation Loss: {valid_score[0]}, Accuracy: {valid_score[1]}")
print(f"Test Loss: {test_score[0]}, Accuracy: {test_score[1]}")

classes = ["Class1", "Class2", "Class3"]
y_true = [0,1,2,1,0]  # dummy
y_pred = [0,1,2,0,0]  # dummy

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", classification_report(y_true, y_pred, target_names=classes))
```

## 5. Result Analysis
This section presents the performance evaluation of various classification models for Alzheimer's disease (AD) progression detection. It includes binary and multiclass classification analyses and compares unimodal vs multimodal approaches.

### 5.1 Binary Classification of Alzheimer's Disease
The binary classification identifies Alzheimer’s affected (AP) and non-affected (AN) categories.
| Model          | Class | Precision | Recall | F1-score | Accuracy |
| -------------- | ----- | --------- | ------ | -------- | -------- |
| Random Forest  | AN    | 0.91      | 0.98   | 0.94     | 93%      |
|                | AP    | 0.96      | 0.82   | 0.89     |          |
| XGBoost        | AN    | 0.94      | 0.98   | 0.96     | 95%      |
|                | AP    | 0.96      | 0.90   | 0.93     |          |
| LGBM           | AN    | 0.95      | 0.98   | 0.97     | 96%      |
|                | AP    | 0.97      | 0.92   | 0.94     |          |
| EfficientNetB3 | AD    | 0.81      | 0.92   | 0.86     | 83%      |
|                | MCI   | 0.86      | 0.70   | 0.77     |          |

* **Observations**: LGBM achieves the highest accuracy (96%), followed by XGBoost (95%) and Random Forest (93%). EfficientNetB3 performs moderately with 83% accuracy.

### 5.2 Multiclass Classification of Alzheimer's Disease
The multiclass analysis considers stages like AD, Cognitive Impairment (CI), and Cognitively Normal (CN).
| Model         | Detection Class      | Precision | Recall | F1-score | Accuracy |
| ------------- | -------------------- | --------- | ------ | -------- | -------- |
| CNN           | AD                   | 0.98      | 0.98   | 0.98     | 98%      |
|               | CI                   | 0.97      | 0.98   | 0.98     |          |
|               | CN                   | 0.97      | 0.96   | 0.97     |          |
| Random Forest | AD                   | 1.00      | 1.00   | 1.00     | 64%      |
|               | CN                   | 0.37      | 0.11   | 0.16     |          |
|               | FTD                  | 0.48      | 0.83   | 0.61     |          |
| Decision Tree | AD                   | 1.00      | 0.99   | 1.00     | 86%      |
|               | CN                   | 0.96      | 0.61   | 0.75     |          |
|               | FTD                  | 0.71      | 0.98   | 0.83     |          |
| VGG16         | AD                   | 0.92      | 0.92   | 0.92     | 89%      |
|               | CI                   | 0.87      | 0.86   | 0.86     |          |
|               | CN                   | 0.87      | 0.89   | 0.88     |          |
| Custom CNN    | Mild Impairment      | 1.00      | 0.65   | 0.79     | 89%      |
|               | Moderate Impairment  | 0.86      | 1.00   | 0.92     |          |
|               | No Impairment        | 0.83      | 1.00   | 0.91     |          |
|               | Very Mild Impairment | 0.99      | 0.83   | 0.90     |          |
| Deep CNN      | Mild Impairment      | 0.98      | 0.93   | 0.96     | 95%      |
|               | Moderate Impairment  | 1.00      | 1.00   | 1.00     |          |
|               | No Impairment        | 0.92      | 0.99   | 0.96     |          |
|               | Very Mild Impairment | 0.98      | 0.90   | 0.94     |          |

* **Observations**: CNN and Deep CNN perform best, with 98% and 95% accuracy. Random Forest shows poor performance (64%) due to class imbalance.

### 5.3 Unimodal vs Multimodal Approaches
Comparison of unimodal (single data type) vs multimodal (combined data types) for AD detection.
| Modality   | Data Used           | Alzheimer's Type | Precision | Recall | F1-score | Accuracy |
| ---------- | ------------------- | ---------------- | --------- | ------ | -------- | -------- |
| Unimodal   | EHR                 | CN, AD, MCI      | 0.76      | 0.77   | 0.76     | 76%      |
|            | Imaging             | CN, AD           | 0.83      | 0.83   | 0.83     | 83%      |
|            | SNP                 | CN, MCI/AD       | 0.66      | 0.57   | 0.53     | 56%      |
| Multimodal | EHR + SNP + Imaging | CN, AD, MCI      | 0.77      | 0.78   | 0.78     | 78%      |
|            | EHR + SNP           | CN, MCI, AD      | 0.78      | 0.79   | 0.78     | 78%      |
|            | EHR + Imaging       | CN, MCI, AD      | 0.76      | 0.77   | 0.77     | 77%      |
|            | SNP + Imaging       | CN, MCI/AD       | 0.62      | 0.55   | 0.57     | 63%      |

* **Observations**: Imaging-based unimodal models outperform others (83%), while multimodal combinations improve accuracy but may struggle with integrating heterogeneous data.
<img width="1206" height="595" alt="Screenshot 2025-11-29 213130" src="https://github.com/user-attachments/assets/42087073-a134-48f0-9643-0a3970992dbb" />
Figure 4: Unimodal vs multimodal models comparison.


## 6. Discussions
Our study evaluates multimodal approaches for analyzing the progression from Mild Cognitive Impairment (MCI) to Alzheimer's Disease (AD). Integrating imaging, clinical, and genetic data improves predictive accuracy, with multimodal models outperforming unimodal ones. In binary classification, traditional ML models such as LGBM, XGBoost, and Random Forest achieved high precision and recall, with LGBM performing best (96% accuracy). Deep learning models like EfficientNetB3 were competitive in detecting AD but less effective for early cognitive decline (83% accuracy). For multiclass classification, CNN-based architectures excelled, achieving up to 98% accuracy. Custom CNN and VGG16 reached 89%, while Decision Tree achieved 86% and Random Forest struggled (64%). These results highlight the strength of CNNs in capturing complex patterns in imaging data, while traditional ML models remain robust for binary tasks. Key indicators of cognitive decline included hippocampal atrophy and cortical thinning, supported by clinical tests and CSF biomarkers. Genetic markers contributed modestly but improved overall performance when combined with imaging and clinical data.

Overall, our framework outperforms prior studies in both binary and multiclass AD detection, demonstrating the effectiveness of combining multimodal data with advanced machine learning and deep learning techniques.

### Comparison with Existing Studies
| Reference     | Used Models                  | Binary Classification Accuracy                | Multiclass Classification Accuracy |
| ------------- | ---------------------------- | --------------------------------------------- | ---------------------------------- |
| Song et al. 2021        | CNN                          | 94%                                           | 74.5%                              |
| El-Sappagh et al. 2021        | SVM, RF, KNN, LR, DT         | RF 90.51%                                     | -                                  |
| Wang et al. 2024       | Decision Tree, Fuzzy systems | 87.08%                                        | 93.95%                             |
| Lee et al. 2019       | RNN                          | 75%                                           | 81%                                |
| Tufail et al. 2022       | 3D CNN                       | 86%                                           | 0.6749 (GM), 0.3953 (MCC)          |
| Amrutesh et al. 2022        | RF, SVM, DT, etc.            | 92.14%                                        | -                                  |
| Buvari and Pettersson, 2020       | CNN, Dense NN                | Numerical 73.59%, Image 71.43%, Hybrid 74.89% | -                                  |
| Baglat et al. 2020       | LR, DT, RF, SVM, AdaBoost    | 86.8%                                         | -                                  |
| **Our Study** | **XGBoost, LGBM, CNN, Deep CNN**  | **LGBM 96%, XGBoost 95%**              | **CNN 98%, Deep CNN 95%**          |


## 7. Future Research Directions
Future research in this direction can be performed on improving multimodal data fusion to better integrate imaging, clinical, and genetic information for comprehensive Alzheimer's prediction. Advancements in explainable AI techniques will further clarify decision-making processes, improving physician confidence in AI-driven diagnoses. Additionally, leveraging real-time monitoring and longitudinal data could capture subtle cognitive changes over time, supporting early interventions. These improvements will strengthen the clinical applicability of multimodal models, leading to earlier and more accurate AD diagnosis.

## Acknowledgements
The work of Dr. Debasish Ghose and Dr. Jia Uddin was partially supported by the UTFORSK Programme, funded by the Norwegian Directorate for Higher Education and Skills (HK-dir), under Project “Sustainable AI Literacy in Higher Education through Multilateral Collaborations (SAIL-MC)” (Project No. UTF-2024/10225).
