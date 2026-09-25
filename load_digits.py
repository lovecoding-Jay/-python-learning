from sklearn.base import clone

from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# 1. 加载手写数字数据集
X, y = load_digits(return_X_y=True)

# 原来 y 是 0 到 9 的数字类别
# 这里改成二分类：是数字 1 为 1，不是数字 1 为 0
y = (y == 1).astype(int)

print("总样本数：", len(y))
print("正例，也就是数字 1 的数量：", y.sum())
print("负例，也就是不是数字 1 的数量：", len(y) - y.sum())
print("每个样本的特征数量：", X.shape[1])

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

# 3. 建立模型流水线
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000),
)

# 4. 在训练集上做 5 折交叉验证
cv_scores = cross_val_score(
    clone(model),
    X_train,
    y_train,
    cv=5,
)

# 5. 使用全部训练数据训练最终模型
model.fit(X_train, y_train)

# 6. 在测试集上预测
pred = model.predict(X_test)

# 7. 输出评价结果
print("\n测试集准确率：", accuracy_score(y_test, pred))

cm = confusion_matrix(y_test, pred)

print("\n混淆矩阵：")
print(cm)

print("\n混淆矩阵中的四个数字：")
print("TN，实际不是 1，预测也不是 1：", cm[0, 0])
print("FP，实际不是 1，却预测成 1：", cm[0, 1])
print("FN，实际是 1，却预测成不是 1：", cm[1, 0])
print("TP，实际是 1，也预测成 1：", cm[1, 1])

print("\n分类报告：")
print(
    classification_report(
        y_test,
        pred,
        target_names=["不是数字1", "数字1"],
    )
)

print("\n5 折交叉验证每次的准确率：", cv_scores)
print("5 折交叉验证平均准确率：", cv_scores.mean())