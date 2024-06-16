import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from skopt.space import Real, Categorical
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, mean_squared_error, \
    confusion_matrix, roc_auc_score

# 加载数据
data_path = "C:\\Users\\ran\\Desktop\\lunwen\\gongcheng\\shuju\\zuixin\\trainingData-2.csv"
data = pd.read_csv(data_path, encoding='gbk')
features = data[['NDVI', 'TWI', 'SPI', '工程岩组', '构造密度', '河流距离', '粗糙度', '坡长', '曲率', '土地利用', '斜坡结构', '斜坡形态']]
labels = data['是否为灾害点']

# 分割数据
X, X_test, y, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 定义搜索空间
search_space = {
    'C': Real(0.1, 100, prior='log-uniform'),
    'gamma': Real(0.01, 1, prior='log-uniform'),
    'kernel': Categorical(['linear', 'rbf'])
}

# 5折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_predictions = np.zeros(X.shape[0])  # 存储训练集上的预测
test_predictions = np.zeros(X_test.shape[0])  # 存储测试集上的预测

for train_index, val_index in kf.split(X):
    X_train_fold, X_val_fold = X_scaled[train_index], X_scaled[val_index]
    y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

    # 初始化 SVM 模型
    svm_model = SVC(probability=True, random_state=42)

    # 贝叶斯搜索超参数调优
    bayes_search = BayesSearchCV(svm_model, search_space, n_iter=50, cv=3, n_jobs=-1, verbose=2, random_state=42)
    bayes_search.fit(X_train_fold, y_train_fold)

    # 获取最佳模型
    best_model = bayes_search.best_estimator_

    # 在验证集上进行预测
    val_pred = best_model.predict_proba(X_val_fold)[:, 1]
    oof_predictions[val_index] = val_pred

    # 在测试集上进行预测
    test_pred = best_model.predict_proba(X_test_scaled)[:, 1]
    test_predictions += test_pred / kf.n_splits

# 计算OOF AUC
oof_auc = roc_auc_score(y, oof_predictions)
print(f"Out-of-Fold AUC: {oof_auc}")

# ROC曲线
fpr_oof, tpr_oof, _ = roc_curve(y, oof_predictions)
plt.figure()
plt.plot(fpr_oof, tpr_oof, label=f'SVM OOF ROC curve (area = {auc(fpr_oof, tpr_oof):.3f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - SVM OOF Predictions')
plt.legend(loc="lower right")
plt.show()

# 使用阈值 0.5 计算预测类别
# 使用阈值 0.5 计算预测类别
predictions_svm = (test_predictions > 0.5).astype(int)
precision_svm = precision_score(y_test, predictions_svm)
recall_svm = recall_score(y_test, predictions_svm)
f1_svm = f1_score(y_test, predictions_svm)
mse_svm = mean_squared_error(y_test, predictions_svm)
rmse_svm = np.sqrt(mse_svm)
mape_svm = np.mean(np.abs((y_test - predictions_svm) / (y_test + 1e-10))) * 100

# 打印评估指标
print(f"\nModel Evaluation Metrics on Test Set:")
print(f"Precision: {precision_svm}")
print(f"Recall: {recall_svm}")
print(f"F1 Score: {f1_svm}")
print(f"MSE: {mse_svm}")
print(f"RMSE: {rmse_svm}")
print(f"MAPE: {mape_svm}%")

# 混淆矩阵
conf_matrix_svm = confusion_matrix(y_test, predictions_svm)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_svm, annot=True, fmt="d", cmap="Blues", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - SVM Model')
plt.show()

# 保存测试集的预测概率结果
np.save("svm_proba.npy", test_predictions)

# 保存测试集的实际标签
np.save("svm_labels.npy", y_test.to_numpy())
np.save("svm_oof_predictions.npy", oof_predictions)