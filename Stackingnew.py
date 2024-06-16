import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, mean_squared_error

from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import itertools


# 加载数据
data = pd.read_csv('C:\\Users\\ran\\Desktop\\lunwen\\gongcheng\\shuju\\zuixin\\trainingData-2.csv', encoding='gbk')
features = data[['NDVI', 'TWI', 'SPI', '工程岩组', '构造密度', '河流距离', '粗糙度', '坡长', '曲率', '土地利用', '斜坡结构', '斜坡形态']]
labels = data['是否为灾害点']

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
# 加载基模型的交叉验证预测和测试集预测
rf_oof = np.load("rf_oof_predictions.npy")
svm_oof = np.load("svm_oof_predictions.npy")
xgb_oof = np.load("xgb_oof_predictions.npy")
cnn_oof = np.load("cnn_oof_predictions.npy")
residual_oof = np.load("residual_oof_predictions.npy")

rf_test = np.load("rf_proba.npy")
svm_test = np.load("svm_proba.npy")
xgb_test = np.load("xgb_proba.npy")
cnn_test = np.load("cnn_proba.npy")
residual_test = np.load("residual_proba.npy")

# 合并为新的训练集和测试集特征
stacked_train_features = np.column_stack((rf_oof, svm_oof, xgb_oof, cnn_oof, residual_oof))
stacked_test_features = np.column_stack((rf_test, svm_test, xgb_test, cnn_test, residual_test))

# 逻辑回归元模型
meta_model = LogisticRegression()
meta_model.fit(stacked_train_features, y_train)

# 逻辑回归元模型超参数调优
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid_search.fit(stacked_train_features, y_train)

# 使用最佳模型进行预测
best_meta_model = grid_search.best_estimator_
meta_model_predictions = best_meta_model.predict_proba(stacked_test_features)[:, 1]

# 评估元模型性能
auc_score = roc_auc_score(y_test, meta_model_predictions)
print(f"Optimized Stacked Model AUC: {auc_score}")

full_data_path = "C:\\Users\\ran\\Desktop\\lunwen\\gongcheng\\shuju\\zuixin\\predictData.csv"
full_data = pd.read_csv(full_data_path, encoding='gbk')
full_features = full_data[['NDVI', 'TWI', 'SPI', '工程岩组', '构造密度', '河流距离', '坡度', '坡长', '曲率', '土地利用', '斜坡结构', '斜坡形态']]
rf_full_pred = rf_model.predict_proba(full_features)[:, 1]  # 替换rf_model为您的随机森林模型
svm_full_pred = svm_model.predict_proba(full_features)[:, 1] # 替换svm_model为您的SVM模型
xgb_full_pred = xgb_model.predict_proba(full_features)[:, 1] # 替换xgb_model为您的XGBoost模型
cnn_full_pred = cnn_model.predict_proba(full_features)[:, 1] # 替换cnn_model为您的CNN模型
residual_full_pred = residual_model.predict_proba(full_features)[:, 1] # 替换residual_model为您的残差网络模型

# 合并预测结果
stacked_full_features = np.column_stack((rf_full_pred, svm_full_pred, xgb_full_pred, cnn_full_pred, residual_full_pred))

# 使用堆叠模型进行预测
stacked_full_predictions = best_meta_model.predict_proba(stacked_full_features)[:, 1]

# 创建新的 DataFrame 包含经纬度和预测结果
output_data = pd.DataFrame()

# 假设 meta_model_predictions 是您的测试集预测概率
# 测试集的真实标签是 y_test
# 对于二分类问题，您可能需要将概率转换为标签
threshold = 0.5  # 设置阈值
predictions_label = (meta_model_predictions > threshold).astype(int)

# 计算精度、召回率和F1分数
precision = precision_score(y_test, predictions_label)
recall = recall_score(y_test, predictions_label)
f1 = f1_score(y_test, predictions_label)

# 计算MSE和RMSE
mse = mean_squared_error(y_test, meta_model_predictions)
rmse = np.sqrt(mse)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")

fpr, tpr, _ = roc_curve(y_test, meta_model_predictions)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

cm = confusion_matrix(y_test, predictions_label)

plt.figure()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, [0, 1])
plt.yticks(tick_marks, [0, 1])

fmt = 'd'
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


