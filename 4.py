# -------------------------------------------------------------- #
# from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
# -------------------------------------------------------------- #
import os
from IPython.core.display import display
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# 4.2 구간 분할, 이산화 그리고 선형 모델, 트리모델
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
# 원-핫-인코딩
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
# 4.5.1 일변량 통계
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectPercentile
# 4.5.3 반복적 특성 선택
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
# -------------------------------------------------------------- #

# 이 파일은 열 이름을 나타내는 헤더가 없으므로 header=None으로 지정하고 'names' 매개변수로 열 이름을 제공합니다.
# data = pd.read_csv(
#     os.path.join(mglearn.datasets.DATA_PATH, 'adult.data'),
#     header=None, index_col=False,
#     names =['age', 'workclass', 'fnlwgt', 'education', 'eduacation-num',
#             'marital-status', 'occupation', 'relationship', 'race', 'gender',
#             'capital-gain', 'capital-loass', 'hours-per-week', 'native-country',
#             'income'])
# 예제를 위해 몇 개의 열만 선택합니다.
# data = data[['age', 'workclass', 'education', 'gender', 'hours-per-week', 'occupation', 'income']]
# IPython.display 함수는 주피터 노트북을 위해 포맷팅된 출력을 만듭니다.
# display(data.head())
# print("원본 특성 : \n", list(data.columns), "\n")
# data_dummies = pd.get_dummies(data)
# print("get_dummies 후의 특성 : \n", list(data_dummies.columns))
# display(data_dummies.head())

# 타깃을 뺀 모든 특성이 포함 p263
# features = data_dummies.loc[:, 'age':'occupation_ Transport-moving']
# Numpy 배열 추출
# X = features.values
# y = data_dummies['income_ >50K'].values
# print("X.shape : {}   y.shape : {}".format(X.shape, y.shape))

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# logreg = LogisticRegression()
# logreg.fit(X_train, y_train)
# print("테스트 점수 : {:.2f}".format(logreg.score(X_test, y_test)))

# 4.1.2 숫자로 표현된 범주형 특성 p265
# 숫자 특성과 범주형 문자열 특성을 가진 DataFrame을 만듭니다.
# demo_df = pd.DataFrame({'number' : [0, 1, 2, 1], 'att' : ['양말', '여우', '양말', '상자']})
# display(demo_df)
# display(pd.get_dummies(demo_df))
# demo_df['number'] = demo_df['number'].astype(str)
# display(pd.get_dummies(demo_df, columns=['number', 'att']))

# 4.2 구간 분할, 이산화 그리고 선형 모델, 트리모델
# X, y = mglearn.datasets.make_wave(n_samples=100)
# line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
# reg = DecisionTreeRegressor(min_samples_split=3).fit(X, y)
# plt.plot(line, reg.predict(line), label="Decision Tree")
# reg = LinearRegression().fit(X, y)
# plt.plot(line, reg.predict(line), '--', label="Linear Regression")
# plt.plot(X[:, 0], y, 'o', c='k')
# plt.ylabel("Regression")
# plt.xlabel("att")
# plt.legend()
# plt.show()

# 구간 분할 p268
# bins = np.linspace(-3, 3, 11) # -3 ~ 3 까지 10구간 분할
# print("구간 : {}".format(bins))
# which_bin = np.digitize(X, bins=bins)
# print("데이터 포인트 :\n", X[:5])
# print("데이터 포인트의 소속 구간 :\n", which_bin[:5])

# 변환을 위해 OneHotEncoder를 사용합니다.
# encoder = OneHotEncoder(sparse=False)
# encoder.fit은 which_bin에 나타난 유일한 값을 찾습니다.
# encoder.fit(which_bin)
# 원-핫-인코딩으로 변환합니다.
# X_binned = encoder.transform(which_bin)
# print(X_binned[:5])
# print("X_binned.shape : {}".format(X_binned.shape))
# line_binned = encoder.transform(np.digitize(line, bins=bins))
# reg = LinearRegression().fit(X_binned, y)
# plt.plot(line, reg.predict(line_binned), label='linear Regression')

# reg = DecisionTreeRegressor(min_samples_split=3).fit(X_binned, y)
# plt.plot(line, reg.predict(line_binned), '--', label='Descion Tree')
# plt.vlines(bins, -3, 3, linewidth=1, alpha=.2)
# plt.legend()
# plt.xlabel("Regression")
# plt.ylabel("att")
# plt.show()

# X_combined = np.hstack([X, X_binned])
# print(X_combined.shape)

# reg = LinearRegression().fit(X_combined, y)
# line_combined = np.hstack([line, line_binned])
# plt.plot(line, reg.predict(line_combined), label="linear regression")
# for bin in bins:
#     plt.plot([bin, bin], [-3, 3], ':', c='k', linewidth=1)
# plt.legend()
# plt.plot(X[:, 0], y, 'o', c='k')
# plt.show()

# X_product = np.hstack([X_binned, X * X_binned])
# print(X_product.shape)
# reg = LinearRegression().fit(X_product, y)
# line_product = np.hstack([line_binned, line * line_binned])
# plt.plot(line, reg.predict(line_product), label='Linear Regression')
# for bin in bins:
#     plt.plot([bin, bin], [-3, 3], ':', c='k', linewidth=1)
# plt.plot(X[:, 0], y, 'o', c='k')
# plt.legend()
# plt.show()

# x ** 10 까지 고차항을 추가합니다.
# 기본값인 "include_bias=True"는 절편을 위해 값이 1인 특성을 추가합니다.
# poly = PolynomialFeatures(degree=10, include_bias=False)
# poly.fit(X)
# X_poly = poly.transform(X)
# print("X_poly.shape : {}".format(X_poly.shape))
# print("X 원소 : \n  : {}".format(X[:5]))
# print("X_poly 원소 :\n{}".format(X_poly[:5]))
# print("항 이름 :\n    {}".format(poly.get_feature_names()))

# reg = LinearRegression().fit(X_poly, y)
# line_poly = poly.transform(line)
# plt.plot(line, reg.predict(line_poly), label="linear Regression")
# plt.plot(X[:, 0], y, 'o', c='k')
# plt.legend()
# plt.show()

# for gamma in [1, 10]:
#     svr = SVR(gamma=gamma).fit(X, y)
#     plt.plot(line, svr.predict(line), label='SVR gamma = {}'.format(gamma))
# plt.plot(X[:, 0], y, 'o', c='k')
# plt.legend()
# plt.show()

# boston = load_boston()
# X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=0)
# 데이터 스케일 조정
# scaler = MinMaxScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled  = scaler.transform(X_test)

# poly = PolynomialFeatures(degree=2).fit(X_train_scaled)
# X_train_poly = poly.transform(X_train_scaled)
# X_test_poly  = poly.transform(X_test_scaled)
# print("X_train.shape      : {}".format(X_train.shape))
# print("X_train_poly.shape : {}".format(X_train_poly.shape))
# print("다항 특성 이름 :\n {}".format(poly.get_feature_names()))

# ridge = Ridge().fit(X_train_scaled, y_train)
# print("상호 작용 특성이 없을 대 점수 : {:.3f}".format(ridge.score(X_test_scaled, y_test)))
# ridge = Ridge().fit(X_train_poly, y_train)
# print("상호 작용 특성이 있을 때 점수 : {:.3f}".format(ridge.score(X_test_poly, y_test)))

# rf = RandomForestRegressor(n_estimators=100, random_state=0).fit(X_train_scaled, y_train)
# print("상호작용 특성이 없을 때 점수 : {:.3f}".format(rf.score(X_test_scaled, y_test)))
# rf = RandomForestRegressor(n_estimators=100, random_state=0).fit(X_train_poly, y_train)
# print("상호작용 특성이 있을 때 점수 : {:.3f}".format(rf.score(X_test_poly, y_test)))

# 4.4 일변량 비선형 변환 p281
# rnd = np.random.RandomState(0)
# X_org = rnd.normal(size=(1000, 3))
# w = rnd.normal(size=3)
# X = rnd.poisson(10 * np.exp(X_org))
# y = np.dot(X_org, w)
# print(X[:10, 0])
# print("특성 출현 횟수 :\n {}".format(np.bincount(X[:, 0])))
# bins = np.bincount(X[:, 0])
# plt.bar(range(len(bins)), bins, color='grey')
# plt.show()
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# score = Ridge().fit(X_train, y_train).score(X_test, y_test)
# print("테스트 점수 : {:.3f}".format(score))
# X_train_log = np.log(X_train + 1)
# X_test_log  = np.log(X_test  + 1)
# plt.hist(X_train_log[:, 0], bins=25, color='grey')
# plt.show()
# score = Ridge().fit(X_train_log, y_train).score(X_test_log, y_test)
# print("테스트 점수 : {:.3f}".format(score))

# 4.5
# 4.5.1 일변량 통계
# cancer = load_breast_cancer()
# 고정된 난수를 발생합니다.
# rng = np.random.RandomState(42)
# print(len(cancer.data))
# noise = rng.normal(size=(len(cancer.data), 50))
# 데이터에 노이즈 특성을 추가합니다.
# 처음 30개는 원본 특성이고 다음 50개는 노이즈입니다.
# X_w_noise = np.hstack([cancer.data, noise])
# X_train, X_test, y_train, y_test = train_test_split(X_w_noise, cancer.target, random_state=0, test_size=.5)
# f_classif(기본값)와 SelectPercentile을 사용하여 특성의 50%를 선택합니다.
# select = SelectPercentile(percentile=50)
# select.fit(X_train, y_train)
# 훈련 세트에 적용합니다.
# X_train_selected = select.transform(X_train)

# print("X_train.shape : {}".format(X_train.shape))
# print("X_train_selected.shape : {}".format(X_train_selected.shape))

# mask = select.get_support()
# print(mask)
# plt.matshow(mask.reshape(1, -1), cmap = 'gray_r')
# plt.xlabel("att number")
# plt.show()

# 테스트 데이터 변환
# X_test_selected = select.transform(X_test)
# lr = LogisticRegression()
# lr.fit(X_train, y_train)
# print("전체 특성을 사용한 점수 : {:.3f}".format(lr.score(X_test, y_test)))
# lr.fit(X_train_selected, y_train)
# print("선택된 일부 특성을 사용한 점수 : {:.3f}".format(lr.score(X_test_selected, y_test)))

# 4.5.2 모델 기반 특성 선택
# select = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold="median")
# select.fit(X_train, y_train)
# X_train_l1 = select.transform(X_train)
# print("X_train.shape: {}".format(X_train.shape))
# print("X_train_l1.shape: {}".format(X_train_l1.shape))
# mask = select.get_support()
# True는 검은색, False는 흰색으로 마스킹합니다.
# plt.matshow(mask.reshape(1, -1), cmap='gray_r')
# plt.xlabel("att number")
# plt.show()

# X_test_l1 = select.transform(X_test)
# score = LogisticRegression().fit(X_train_l1, y_train).score(X_test_l1, y_test)
# print("테스트 점수 : {:.3f}".format(score))

# 4.5.3 반복적 특성 선택
# select = RFE(RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=40)
# select.fit(X_train, y_train)
# 선택한 특성을 표시합니다.
# mask = select.get_support()
# plt.matshow(mask.reshape(1, -1), cmap='gray_r')
# plt.xlabel("att number")
# plt.show()

# X_train_rfe = select.transform(X_train)
# X_test_rfe  = select.transform(X_test)
# score = LogisticRegression().fit(X_train_rfe, y_train).score(X_test_rfe, y_test)
# print("테스트 점수 : {:.3f}".format(score))
# print("테스트 점수 : {:.3f}".format(select.score(X_test, y_test)))

citibike = mglearn.datasets.load_citibike()
print("시티바이크 데이터 : \n {}".format(citibike.head()))

plt.figure(figsize=(10, 5))
xticks = pd.date_range(start=citibike.index.min(), end=citibike.index.max(), freq='D')
week = ["Sun", "Mon", "Tues","wed", "Turs", "Fri", "Sat"]
xticks_name = [week[int(w)] + d for w, d in zip(xticks.strftime("%w"), xticks.strftime("%m-%d"))]
# plt.xticks(xticks, xticks_name, rotation=90, ha="left")
# plt.plot(citibike, linewidth=1)
# plt.show()

# 타깃값 추출 (대여 횟수)
y = citibike.values
# POSIX 시간을 10**9로 나누어 변환
X = citibike.index.astype("int64").values.reshape(-1, 1) // 10**9

# 처음 184개 데이터 포인트를 훈련 세트로 사용하고 나머지는 테스트 세트로 사용합니다.
n_train = 184
# 주어진 특성을 사용하여 평가하고 그래프를 그립니다.
def eval_on_features(features, target, regressor):
    # 훈련 세트와 테스트 세트로 나눕니다.
    X_train, X_test = features[:n_train], features[n_train:]
    # 타깃값도 나눕니다.
    y_train, y_test = target[:n_train], target[n_train:]
    regressor.fit(X_train, y_train)
    print("테스트 세트 R^2 : {:.2f}".format(regressor.score(X_test, y_test)))
    y_pred = regressor.predict(X_test)
    y_pred_train = regressor.predict(X_train)
    plt.figure(figsize=(10, 3))
    plt.xticks(range(0, len(X), 8), xticks_name, rotation=90, ha="left")

    plt.plot(range(n_train), y_train, label="train")
    plt.plot(range(n_train, len(y_test) + n_train), y_test, '-', label="test")
    plt.plot(range(n_train), y_pred_train, '--', label="train predict")
    plt.plot(range(n_train, len(y_test) + n_train), y_pred, '--', label="test predict")

    plt.legend(loc=(1.01, 0))
    plt.xlabel("days")
    plt.ylabel("count")
    plt.show()

regressor = RandomForestRegressor(n_estimators=100, random_state=0)
# eval_on_features(X, y, regressor)

# 전문가 지식 (시간) 추가
# X_hour = citibike.index.hour.values.reshape(-1, 1)
# eval_on_features(X_hour, y, regressor)

# 전문가 지식 (요일, 시간) 추가
X_hour_week = np.hstack([citibike.index.dayofweek.values.reshape(-1, 1), citibike.index.hour.values.reshape(-1, 1)])
# eval_on_features(X_hour_week, y, regressor)

# eval_on_features(X_hour_week, y, LinearRegression())
enc = OneHotEncoder()
X_hour_week_onehot = enc.fit_transform(X_hour_week).toarray()
# eval_on_features(X_hour_week_onehot, y, Ridge())

# 상호작용 특성 적용
poly_transformer = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_hour_week_onehot_poly = poly_transformer.fit_transform(X_hour_week_onehot)
lr = Ridge()
# eval_on_features(X_hour_week_onehot_poly, y, lr)

hour = ["%02d:00" % i for i in range(0, 24, 3)]
day  = ["Mon", "Tues", "Wed", "Tur", "Fri", "Sat", "Sun"]
features = day + hour
features_poly = poly_transformer.get_feature_names(features)
features_nonzero = np.array(features_poly)[lr.coef_ != 0]
coef_nonzero = lr.coef_[lr.coef_ != 0]
plt.figure(figsize=(15, 2))
plt.plot(coef_nonzero, 'o')
plt.xticks(np.arange(len(coef_nonzero)), features_nonzero, rotation=90)
plt.xlabel("att name")
plt.ylabel("coef size")
plt.show()
