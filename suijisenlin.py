# 加载数据
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, mean_squared_error, \
    confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib


# 加载数据
data_path = "C:\\Users\\ran\\Desktop\\lunwen\\gongcheng\\shuju\\zuixin\\selected_data.csv"
data = pd.read_csv(data_path, encoding='gbk')
features = data[['NDVI', 'TWI', 'SPI', '工程岩组', '构造密度', '河流距离', '坡度', '坡长', '曲率', '土地利用', '斜坡结构', '斜坡形态']]
labels = data['是否为灾害点']


# 分割数据为训练集和测试集
X, X_test, y, y_test = train_test_split(features, labels, test_size=0.3, random_state=42, stratify=labels)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)


# 5折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_predictions = np.zeros(X.shape[0])  # 存储训练集上的预测
test_predictions = np.zeros(X_test.shape[0])  # 存储测试集上的预测


for train_index, val_index in kf.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    # 定义参数网格
    param_grid = {
        'n_estimators': [10, 50, 100, 150, 200],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    # 创建模型
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=2, scoring='roc_auc')
    grid_search.fit(X_train, y_train)

    # 获取最佳模型
    best_model = grid_search.best_estimator_

    # 在验证集和测试集上进行预测
    val_predictions = best_model.predict_proba(X_val)[:, 1]
    test_predictions += best_model.predict_proba(X_test)[:, 1] / kf.n_splits

    # 保存验证集上的预测结果
    oof_predictions[val_index] = val_predictions

    oof_auc = roc_auc_score(y, oof_predictions)
    print(f"Out-of-Fold AUC: {oof_auc}")
    fpr_oof, tpr_oof, _ = roc_curve(y, oof_predictions)
    roc_auc_oof = auc(fpr_oof, tpr_oof)

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
output_file_path = "C:\\Users\\ran\\Desktop\\lunwen\\gongcheng\\shuju\\zuixin\\RF_predictions-.csv"
output_data.to_csv(output_file_path, index=False, encoding='gbk')

plt.figure()
plt.plot(fpr_oof, tpr_oof, label=f'Random Forest OOF ROC curve (area = {roc_auc_oof:.3f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Random Forest OOF Predictions')
plt.legend(loc="lower right")
plt.show()

# 计算评估指标
threshold = 0.5
predictions_best_grid = (test_predictions > threshold).astype(int)
precision_best_grid = precision_score(y_test, predictions_best_grid)
recall_best_grid = recall_score(y_test, predictions_best_grid)
f1_best_grid = f1_score(y_test, predictions_best_grid)
mse_best_grid = mean_squared_error(y_test, predictions_best_grid)
rmse_best_grid = np.sqrt(mse_best_grid)
mape_best_grid = np.mean(np.abs((y_test - predictions_best_grid) / (y_test + 1e-10))) * 100



# 打印评估指标
print(f"\nModel Evaluation Metrics on Test Set:")
print(f"Precision: {precision_best_grid}")
print(f"Recall: {recall_best_grid}")
print(f"F1 Score: {f1_best_grid}")
print(f"MSE: {mse_best_grid}")
print(f"RMSE: {rmse_best_grid}")
print(f"MAPE: {mape_best_grid}%")

# 混淆矩阵
conf_matrix_best_grid = confusion_matrix(y_test, predictions_best_grid)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_best_grid, annot=True, fmt="d", cmap="Blues", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Optimized Random Forest Model')
plt.show()


# 保存测试集的预测概率结果
np.save("rf_proba.npy", test_predictions)

# 保存测试集的实际标签
np.save("rf_labels.npy", y_test.to_numpy())
np.save("rf_oof_predictions.npy", oof_predictions)
joblib.dump(best_model, 'best_rf_model.joblib')  # 将模型保存到文件
# # 特征重要性分析
# feature_importances = best_grid.feature_importances_
# feature_names = features.columns
#
# # 创建一个特征重要性的DataFrame
# importance_df = pd.DataFrame({
#     'feature': feature_names,
#     'importance': feature_importances
# }).sort_values(by='importance', ascending=False)
#
# # 绘制条形图
# plt.figure(figsize=(10, 6))
# sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
# plt.title('Feature Importance - Optimized Random Forest Model')
# plt.xlabel('Importance')
# plt.ylabel('Feature')
# plt.show()