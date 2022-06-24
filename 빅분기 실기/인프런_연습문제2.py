## ** 연습문제2_회귀예측문제**

##### 유의사항 #####
# 성능이 우수한 예측모형을 구축하기 위해서는 적절한 데이터 전처리,
# 피쳐엔지니어링, 분류 알고리즘 사용, 초매개변수 최적화, 모형 앙상블 등이 수반되어야 한다.
# 수험번호.csv 파일이 만들어지도록 코드를 제출한다.
# 제출한 모형의 성능은 RMSE, MAE 평기지표에 따라 채점한다.
# 종송변수 mpg

##### 데이터 파일 읽기 예제 #####
# 데이터 파일 읽기 예제
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
df = sns.load_dataset('mpg')
X_train, X_test, y_train, y_test = train_test_split(df, df['mpg'], test_size=0.2, random_state=42)
X_train = X_train.drop(['mpg'], axis=1)
X_test = X_test.drop(['mpg'], axis=1)
print(X_train.head())

##### 답안제출 #####
# 아래 코드 예측변수와 수험번호를 개인별로 변경하여 활용
# pd.DataFrame({'id':y_test.index, 'pred':pred}).to_csv('000.csv', index=False)

##### 사용자코드 #####
## 결측치 처리
print('X_train', X_train.isna().sum())
print('X_test', X_test.isna().sum())

# 연속형 변수의 결측치를 중앙값으로 처리
X_train.horsepower = X_train.horsepower.fillna(X_train.horsepower.median())
X_test.horsepower = X_test.horsepower.fillna(X_test.horsepower.median())

## 라벨인코딩
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

cat_col = ['origin', 'name']
X_train[cat_col] = X_train[cat_col].apply(le.fit_transform)
X_test[cat_col] = X_test[cat_col].apply(le.fit_transform)

## 더미 변수화
print(X_train.dtypes)
dum_col = ['origin']
for i in dum_col:
    X_train[i] = X_train[i].astype('category')
    X_test[i] = X_test[i].astype('category')
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)
print(X_train.head())

## 변수 생성
X_train['weight_qcut'] = pd.qcut(X_train['weight'], 5, labels=False)
X_test['weight_qcut'] = pd.qcut(X_test['weight'], 5, labels=False)

## 스케일링
from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()

scale_col = ['displacement', 'horsepower', 'weight']
scale.fit(X_train[scale_col])
scale.fit(X_test[scale_col])
X_train[scale_col] = scale.transform(X_train[scale_col])
X_test[scale_col] = scale.transform(X_test[scale_col])

## train, valid split
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

## 모델 학습
from sklearn.ensemble import RandomForestRegressor
model1 = RandomForestRegressor()
model1.fit(X_train, y_train)
pred1 = model1.predict(X_valid)
print(pred1)

from sklearn.ensemble import AdaBoostRegressor
model2 = AdaBoostRegressor()
model2.fit(X_train, y_train)
pred2 = model2.predict(X_valid)
print(pred2)

## 앙상블(stacking)
from sklearn.ensemble import StackingRegressor
estimators = [('rf', model1), ('ada', model2)]
model3 = StackingRegressor(estimators=estimators, final_estimator=RandomForestRegressor())
model3.fit(X_train, y_train)
pred3 = model3.predict(X_valid)
print(pred3)

## 성능확인
from sklearn.metrics import mean_squared_error, mean_absolute_error
print('랜덤포레스트 RMSE', mean_squared_error(pred1, y_valid, squared=False))
print('랜덤포레스트 MAE', mean_absolute_error(pred1, y_valid))
print('AdaBoost RMSE', mean_squared_error(pred2, y_valid, squared=False))
print('AdaBoost MAE', mean_absolute_error(pred2, y_valid))
print('Stacking RMSE', mean_squared_error(pred3, y_valid, squared=False))
print('Stacking MAE', mean_absolute_error(pred3, y_valid))

## 하이퍼 파라미터 튜닝
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':[50, 100], 'max_depth':[4, 6]}
model4 = RandomForestRegressor()
clf = GridSearchCV(estimator=model4, param_grid=parameters,cv=3)
clf.fit(X_train, y_train)
print('최적의 파라미터', clf.best_params_)

## 저장
pred = model1.predict(X_test)
pd.DataFrame({'id':y_test.index, 'pred':pred}).to_csv('123.csv', index=False)

print(pd.read_csv('123.csv'))
