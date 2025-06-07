训练使用xgboost，可以不用上云平台，本地就能跑
项目目前已经完成，下载之后数据data需先解压，随后运行model，即可看到训练结果
数据处理使用了：
1.数据清洗
过滤掉不合理的 BMI 值（10 到 60 以外的删掉）
2.特征构造
新增了二分类特征 IsObese（BMI≥30判为肥胖）
新增组合健康风险评分 HealthRiskScore = MentHlth + PhysHlth
BMI 分箱（分成“偏瘦”“正常”“超重”“肥胖”四类）
年龄分组（分箱年龄段）
3.类别变量编码
对 Sex、Education、Income、BMI_Category、AgeGroup 使用One-Hot编码，并去掉第一个类别避免多重共线性
4.数值特征标准化
用 StandardScaler 标准化了 BMI、健康风险评分、年龄，使它们均值为0，标准差为1
5.处理类别不平衡
用 SMOTE（Synthetic Minority Over-sampling Technique） 进行少数类过采样，生成合成样本，缓解糖尿病（Diabetes_binary）正负样本不平衡的问题


模型训练使用了基模型 + 集成模型的训练方法
1.基模型（Base Learners）
1)XGBoost（xgb.XGBClassifier）
梯度提升树（GBDT）的一种高效实现，适合二分类任务，参数中设置了较多正则化项和样本采样策略，控制过拟合。
2)LightGBM（LGBMClassifier）
另一种基于梯度提升的决策树框架，训练速度快，支持类别不平衡参数 scale_pos_weight。
3)随机森林（RandomForestClassifier）
基于多棵决策树的随机森林，利用集成思想减少过拟合，且使用了 class_weight='balanced' 来缓解类别不平衡。
2.集成方法（Ensemble Method）
1)堆叠（StackingClassifier）
把上述三个基模型作为第一层学习器，输出的预测结果作为新的特征输入给一个次级学习器（final_estimator，这里是一个LightGBM分类器）进行最终预测。
通过交叉验证 (cv=5) 来训练堆叠模型，减少过拟合。
passthrough=True 意味着除了第一层模型的输出，还保留了原始特征一起输入给次级学习器。