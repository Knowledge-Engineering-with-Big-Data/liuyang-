import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, mean_squared_error, \
    confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

# 加载数据
data_path = "C:\\Users\\ran\\Desktop\\lunwen\\gongcheng\\shuju\\zuixin\\selected_data.csv"
data = pd.read_csv(data_path, encoding='gbk')
features = data[
    ['NDVI', 'TWI', 'SPI', '工程岩组', '构造密度', '河流距离', '坡度', '坡长', '曲率', '土地利用', '斜坡结构',
     '斜坡形态']]
labels = data['是否为灾害点']

# 分割数据为训练集和测试集
X, X_test, y, y_test = train_test_split(features, labels, test_size=0.3, random_state=42,stratify=labels)

# 5折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_predictions = np.zeros(X.shape[0])  # 存储训练集上的预测
test_predictions = np.zeros(X_test.shape[0])  # 存储测试集上的预测

# 定义超参数网格
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9]
}

for train_index, val_index in kf.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # 初始化XGBoost模型
    xgb_model = xgb.XGBClassifier(random_state=42)

    # 网格搜索超参数调优
    grid_search = GridSearchCV(xgb_model, param_grid, cv=3, n_jobs=-1, verbose=2, scoring='roc_auc')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # 在验证集和测试集上进行预测
    val_predictions = best_model.predict_proba(X_val)[:, 1]
    test_predictions += best_model.predict_proba(X_test)[:, 1] / kf.n_splits

    # 保存验证集上的预测结果
    oof_predictions[val_index] = val_predictions

full_data_path = "C:\\Users\\ran\\Desktop\\lunwen\\gongcheng\\shuju\\zuixin\\predictData.csv"
full_data = pd.read_csv(full_data_path, encoding='gbk')
full_features = full_data[['NDVI', 'TWI', 'SPI', '工程岩组', '构造密度', '河流距离', '坡度', '坡长', '曲率', '土地利用', '斜坡结构', '斜坡形态']]


# 使用最佳模型进行预测
full_predictions = best_model.predict_proba(full_features)[:, 1]

# 创建新的 DataFrame 包含经纬度和预测结果
output_data = pd.DataFrame()
output_data['经度'] = full_data.iloc[:, 0]  # 假设第一列是经度
output_data['纬度'] = full_data.iloc[:, 1]  # 假设第二列是纬度
output_data['易发性概率'] = full_predictions

# 保存为 CSV 文件
output_file_path = "C:\\Users\\ran\\Desktop\\lunwen\\gongcheng\\shuju\\zuixin\\XGBoost_predictions.csv"
output_data.to_csv(output_file_path, index=False, encoding='gbk')

# Out-of-Fold ROC-AUC
oof_auc = roc_auc_score(y, oof_predictions)
print(f"Out-of-Fold AUC: {oof_auc}")
fpr_oof, tpr_oof, _ = roc_curve(y, oof_predictions)
roc_auc_oof = auc(fpr_oof, tpr_oof)

test_probabilities = best_model.predict_proba(X_test)[:, 1]

# 计算测试集的AUC
test_auc = roc_auc_score(y_test, test_probabilities)

print(f"Test Set AUC: {test_auc}")

plt.figure()
plt.plot(fpr_oof, tpr_oof, label=f'XGBoost OOF ROC curve (area = {roc_auc_oof:.3f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - XGBoost OOF Predictions')
plt.legend(loc="lower right")
plt.show()

# 计算评估指标
threshold = 0.5
predictions_best_xgb = (test_predictions > threshold).astype(int)
precision_best_xgb = precision_score(y_test, predictions_best_xgb)
recall_best_xgb = recall_score(y_test, predictions_best_xgb)
f1_best_xgb = f1_score(y_test, predictions_best_xgb)
mse_best_xgb = mean_squared_error(y_test, predictions_best_xgb)
rmse_best_xgb = np.sqrt(mse_best_xgb)
mape_best_xgb = np.mean(np.abs((y_test - predictions_best_xgb) / (y_test + 1e-10))) * 100

# 打印评估指标
print(f"\nModel Evaluation Metrics on Test Set:")
print(f"Precision: {precision_best_xgb}")
print(f"Recall: {recall_best_xgb}")
print(f"F1 Score: {f1_best_xgb}")
print(f"MSE: {mse_best_xgb}")
print(f"RMSE: {rmse_best_xgb}")
print(f"MAPE: {mape_best_xgb}%")

# 混淆矩阵
conf_matrix_best_xgb = confusion_matrix(y_test, predictions_best_xgb)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_best_xgb, annot=True, fmt="d", cmap="Blues", xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Optimized XGBoost Model')
plt.show()

# 保存测试集的预测概率结果
np.save("xgb_proba.npy", test_predictions)
# 保存测试集的实际标签
np.save("xgb_labels.npy", y_test.to_numpy())
np.save("xgb_oof_predictions.npy", oof_predictions)
