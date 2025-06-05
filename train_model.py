import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# 假设 data_split 函数用于划分数据集
def data_split(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
    return train_x, test_x, train_y, test_y

# 假设 data_scaling 函数用于数据缩放
def data_scaling(train_x, test_x):
    scaler = StandardScaler()
    train_x_scaled = scaler.fit_transform(train_x)
    test_x_scaled = scaler.transform(test_x)
    return train_x_scaled, test_x_scaled

# 读取数据集
try:
    data = pd.read_csv('aftprocessdata/processed_brfss.csv')
except FileNotFoundError:
    print("未找到 processed_brfss.csv 文件，请确保数据集文件存在。")
    exit(1)

# 数据划分
train_x, test_x, train_y, test_y = data_split(data)

# 数据缩放
train_x_scaled, test_x_scaled = data_scaling(train_x, test_x)

# 定义要搜索的参数网格
# param_grid = {
#     'learning_rate': [0.01, 0.1, 0.2],
#     'n_estimators': [50, 100, 200],
#     'max_depth': [3, 5, 7],
#     'subsample': [0.8, 0.9, 1.0],
#     'colsample_bytree': [0.8, 0.9, 1.0]
# }
param_grid = {
    'learning_rate': [0.32],
    'n_estimators': [200],
    'max_depth': [1, 2],
    'subsample': [0.75],
    'colsample_bytree': [0.8]
}


# 创建 XGBoost 分类器
model = xgb.XGBClassifier()

# 创建 GridSearchCV 对象
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='f1_macro')

# 在训练集上进行网格搜索
grid_search.fit(train_x_scaled, train_y)

# 输出最佳参数和最佳得分
print("最佳参数:", grid_search.best_params_)
print("最佳得分:", grid_search.best_score_)

# 使用最佳模型进行预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(test_x_scaled)

# 输出分类报告
print(classification_report(test_y, y_pred))