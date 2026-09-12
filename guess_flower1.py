import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 1. 加载数据
iris = load_iris()

X = iris.data
y = iris.target

print("X 的形状：", X.shape)
print("y 的形状：", y.shape)


# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

print("训练集样本数：", len(X_train))
print("测试集样本数：", len(X_test))


# 3. 计算每个特征的中位数
thresholds = np.median(X_train, axis=0)

print("每个特征的中位数：", thresholds)


# 把连续数值转换成 0 和 1
def to_low_high(data):
    compared = data >= thresholds
    result = compared.astype(int)
    return result


Xtr = to_low_high(X_train)
Xte = to_low_high(X_test)

print("转换后的训练数据前 5 行：")
print(Xtr[:5])


# 4. 统计每个类别在训练集中有多少个样本
classes = [0, 1, 2]

counts = []

for c in classes:
    count = 0

    for answer in y_train:
        if answer == c:
            count = count + 1

    counts.append(count)

print("每个类别的样本数量：", counts)


# 5. 计算每个类别的先验概率
total_train = len(y_train)
priors = []

for class_index in range(len(classes)):
    p_class = counts[class_index] / total_train
    priors.append(p_class)

print("每个类别的先验概率：", priors)


# 6. 计算每个类别的条件概率
# likelihood[类别下标][特征下标] = 该特征为高的概率
likelihood = []

feature_count = Xtr.shape[1]
#找出属于当前类别的训练数据
for class_index in range(len(classes)):
    c = classes[class_index]

    # 找出训练集中所有属于当前类别的数据
    X_class = []

    for row_index in range(len(y_train)):
        if y_train[row_index] == c:
            X_class.append(Xtr[row_index])

    class_probabilities = []

    # 对当前类别的 4 个特征分别统计
    for feature_index in range(feature_count):
        high_count = 0

        for row in X_class:
            if row[feature_index] == 1:
                high_count = high_count + 1

        p_high = high_count / len(X_class)
        class_probabilities.append(p_high)

    likelihood.append(class_probabilities)

print("每个类别的条件概率：")
for class_index in range(len(classes)):
    print("类别", classes[class_index], "：", likelihood[class_index])


# 7. 预测一个样本
def predict_one(x):
    best_class = None
    best_score = None

    for class_index in range(len(classes)):
        c = classes[class_index]

        # 一开始只使用该类别的先验概率
        score = np.log(priors[class_index])

        # 再把每个特征的概率加入得分
        for feature_index in range(len(x)):
            p_high = likelihood[class_index][feature_index]

            if x[feature_index] == 1:
                p = p_high
            else:
                p = 1.0 - p_high

            score = score + np.log(p)

        # 选择得分最高的类别
        if best_score is None:
            best_score = score
            best_class = c
        elif score > best_score:
            best_score = score
            best_class = c

    return best_class


# 8. 预测所有测试样本
preds = []

for x in Xte:
    result = predict_one(x)
    preds.append(result)

print("前 10 个预测结果：", preds[:10])
print("前 10 个真实结果：", y_test[:10])


# 9. 计算准确率
accuracy = accuracy_score(y_test, preds)

print("手写朴素贝叶斯准确率：", round(accuracy, 4))