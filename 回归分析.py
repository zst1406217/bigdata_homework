# 回归分析
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.model_selection import KFold

kf = KFold(n_splits=5)

train_use = train.copy()
train_Y = train_use['life']
del train_use['life']
del train_use['train_file_name']
train_X = train_use
test_use = test.copy()
test_Y = test_use['life']
del test_use['life']
del test_use['train_file_name']
test_X = test_use

train_X.to_csv("D:/Data_Ana/HM/data/train_X.csv")
train_Y.to_csv("D:/Data_Ana/HM/data/train_Y.csv")
test_X.to_csv("D:/Data_Ana/HM/data/test_X.csv")
test_Y.to_csv("D:/Data_Ana/HM/data/test_Y.csv")


def split_data(data):
    """
    split data into train and test
    Output:
        x_train: the features of train data
        y_train: the labels of train data
        x_test: the features of test data
        y_test: the labels of test data
    """
    y = data['life']  # get the label
    x = data.loc[:, data.columns != 'life']  # get the feature
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=1)  # split the data
    assert x_train.shape[0] + \
        x_test.shape[0] == data.shape[0]  # check the data
    return x_train, x_test, y_train, y_test


def predict(x_train, x_test, y_train):
    # knn
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)
    knn_pred = knn.predict(x_test)

    # svm
    svm = SVC()
    svm.fit(x_train, y_train)
    svm_pred = svm.predict(x_test)

    # logistic regression
    log_lm = LogisticRegression()
    log_lm.fit(x_train, y_train)
    logy_pred = log_lm.predict(x_test)

    # mlp
    mlp = MLPClassifier()
    mlp.fit(x_train, y_train)
    mlp_pred = mlp.predict(x_test)

    # random forest
    rf = RandomForestClassifier(random_state=42)
    rf.fit(x_train, y_train)
    rfy_pred = rf.predict(x_test)

    # adaboost
    ada = AdaBoostClassifier(random_state=42)
    ada.fit(x_train, y_train)
    aday_pred = ada.predict(x_test)

    # gradient boosting
    gbrt = GradientBoostingClassifier(random_state=42)
    gbrt.fit(x_train, y_train)
    gbrt_pred = gbrt.predict(x_test)

    # bagging
    bag = BaggingClassifier(random_state=42)
    bag.fit(x_train, y_train)
    bag_pred = bag.predict(x_test)

    predict = {'knn': knn_pred, 'svm': svm_pred, 'logistic': logy_pred, 'mlp': mlp_pred,
               'random forest': rfy_pred, 'adaboost': aday_pred, 'gradient boosting': gbrt_pred, 'bagging': bag_pred}

    return predict

# 评价函数


def get_score(predict, target):
    N = len(predict)
    sum = 0
    for i in range(N):
        sum += (np.log10(target.values[i]+1)-np.log10(predict[i]+1))**2
    score = np.sqrt(sum/N)
    return score


regressor = tree.DecisionTreeRegressor(criterion='squared_error',
                                       random_state=0,
                                       min_samples_leaf=2,
                                       min_samples_split=10)
predict_test = pd.DataFrame({'life': np.zeros(len(test_X))})
for fold_id, (trn_idx, val_idx) in enumerate(kf.split(train_X)):
    print(f'\nFold_{fold_id} Training ================================\n')
    data_train = train_X.iloc[trn_idx]
    target_train = train_Y.iloc[trn_idx]
    data_valid = train_X.iloc[val_idx]
    target_valid = train_Y.iloc[val_idx]
    regressor = regressor.fit(data_train, target_train)
    predict_train = regressor.predict(data_train)
    predict_valid = regressor.predict(data_valid)
    predict_test['life'] += regressor.predict(test_X)/5

    score_train = get_score(predict_train, target_train)
    score_valid = get_score(predict_valid, target_valid)
    print('score_train:', score_train, 'score_valid:', score_valid)
score_test = get_score(predict_test['life'], test_Y)
print('score_test:', score_test)
