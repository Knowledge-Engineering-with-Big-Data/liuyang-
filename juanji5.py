import os

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras_tuner import RandomSearch
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error, confusion_matrix
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint


data = pd.read_csv("C:\\Users\\ran\\Desktop\\lunwen\\gongcheng\\shuju\\zuixin\\selected_data.csv", encoding='gbk')

# 接下来是您原始代码的其他部分...

# Loading data
# data = pd.read_csv("C:\\Users\\ran\\Desktop\\论文\\工程\\数据\\最新\\trainingData.csv")
features = data[['NDVI', 'TWI','SPI', '工程岩组','构造密度', '河流距离', '坡度', '坡长','曲率', '土地利用', '斜坡结构', '斜坡形态']]
labels = data['是否为灾害点']

# Splitting and scaling the data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshaping data for 2D-CNN
X_train_reshaped = X_train_scaled.reshape(-1, 4, 3, 1)
X_test_reshaped = X_test_scaled.reshape(-1, 4, 3, 1)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_predictions = np.zeros(X_train_reshaped.shape[0])

def build_model(hp):
    model = keras.Sequential()

    # First Convolutional Layer
    model.add(keras.layers.Conv2D(
        filters=hp.Int('conv_filters_1', min_value=32, max_value=128, step=16),
        kernel_size=(2, 2),
        activation='relu',
        padding='same',  # Use 'same' padding here
        input_shape=(4, 3, 1)
    ))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1)))

    # Second Convolutional Layer (Optional, but can help model to learn more complex patterns)
    model.add(keras.layers.Conv2D(
        filters=hp.Int('conv_filters_2', min_value=32, max_value=64, step=16),
        kernel_size=(2, 2),
        activation='relu',
        padding='same'
    ))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.1)))

    # Flatten & Dense Layers
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(
        units=hp.Int('dense_units_1', min_value=32, max_value=128, step=16),
        activation='relu'
    ))
    model.add(keras.layers.Dropout(hp.Float('dropout_3', min_value=0.0, max_value=0.5, step=0.1)))

    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


best_models = []
model_save_path = "saved_models"  # 确保这个文件夹存在
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

for i, (train_index, val_index) in enumerate(kf.split(X_train_reshaped)):
    X_train_fold, X_val_fold = X_train_reshaped[train_index], X_train_reshaped[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    tuner = RandomSearch(build_model, objective='val_accuracy', max_trials=10, executions_per_trial=3,
                         directory='D:\\workstation\\suanfa_projects',
                         project_name=f'susceptibility_fold_{i+1}')

    tuner.search(X_train_fold, y_train_fold, validation_data=(X_val_fold, y_val_fold), epochs=100, batch_size=32)
    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.save(os.path.join(model_save_path, f'best_model_fold_{i+1}.h5'))
    best_models.append(best_model)
    val_predictions = best_model.predict(X_val_fold).flatten()
    oof_predictions[val_index] = val_predictions

# 单独评估每个保存的模型
for i, model in enumerate(best_models):
    model = keras.models.load_model(os.path.join(model_save_path, f'best_model_fold_{i+1}.h5'))
    scores = model.evaluate(X_test_reshaped, y_test, verbose=0)
    print(f"Model from fold {i+1}:")
    print(f"  - Test Loss: {scores[0]}")
    print(f"  - Test Accuracy: {scores[1]}")
# 获取最佳模型的参数
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best hyperparameters for this fold: ", best_hyperparameters.values)
predictions_proba_cnn = best_model.predict(X_test_reshaped)
fpr_cnn, tpr_cnn, thresholds_cnn = roc_curve(y_test, predictions_proba_cnn)
roc_auc_cnn = auc(fpr_cnn, tpr_cnn)

oof_auc = roc_auc_score(y_train, oof_predictions)
print(f"Overall Out-of-Fold AUC: {oof_auc}")

fpr_oof, tpr_oof, _ = roc_curve(y_train, oof_predictions)
plt.figure()
plt.plot(fpr_oof, tpr_oof, label=f'CNN OOF ROC curve (area = {auc(fpr_oof, tpr_oof):.3f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - CNN OOF Predictions')
plt.legend(loc="lower right")
plt.show()

# Compute evaluation metrics
threshold = 0.5  # You can adjust this threshold based on your requirement
predictions_cnn = (predictions_proba_cnn > threshold).astype(int).flatten()
precision_cnn = precision_score(y_test, predictions_cnn)
recall_cnn = recall_score(y_test, predictions_cnn)
f1_cnn = f1_score(y_test, predictions_cnn)
mse_cnn = mean_squared_error(y_test, predictions_cnn)
rmse_cnn = np.sqrt(mse_cnn)
# For MAPE, we need to handle cases where the true value is 0 to avoid division by zero
mape_cnn = np.mean(np.abs((y_test - predictions_cnn) / (y_test + 1e-10))) * 100  # Added small value to avoid div by zero
conf_matrix_cnn = confusion_matrix(y_test, predictions_cnn)

# Print the evaluation metrics
print(f"Precision: {precision_cnn}")
print(f"Recall: {recall_cnn}")
print(f"F1 Score: {f1_cnn}")
print(f"MSE: {mse_cnn}")
print(f"RMSE: {rmse_cnn}")
print(f"MAPE: {mape_cnn}%")
print("Confusion Matrix:")
print(conf_matrix_cnn)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_cnn, annot=True, fmt="d", cmap="Blues", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - 2D-CNN Model')
plt.show()
np.save("cnn_proba.npy", predictions_proba_cnn)
np.save("cnn_labels.npy", y_test)
np.save("cnn_oof_predictions.npy", oof_predictions)
