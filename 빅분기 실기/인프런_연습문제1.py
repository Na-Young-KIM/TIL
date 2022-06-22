## **연습문제1_분류예측문제**

##### 유의사항 #####
# 성능이 우수한 예측모형을 구축하기 위해서는 적절한 데이터 전처리,
# 수험번호.csv 파일이 만들어지도록 코드를 제출한다.
# 제출한 모형의 성능은 ROC-AUC 평기지표에 따라 채점한다.
# predict_proba로 예측, 종속변수 survived열의 범주1 확률을 예측
# 데이터 파일 읽기 예제

##### 답안 제출 #####
# 아래 코드 예측변수와 수험번호를 개인별로 변경하여 제출
# pd.DataFrame({'id':y_test.index, 'pred':pred}).to_csv('0003000000.csv', index=False)

##### 데이터 파일 읽기 #####
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
df = sns.load_dataset('titanic')
# X_train, X_test, y_train, y_test = train_test_split(df, df['survived'], test_size=0.2, random_state=42, stratify=df['survived'])
# X_train = X_train.drop(['alive', 'survived'], axis=1)
# X_test = X_test.drop(['alive', 'survived'], axis=1)
print('data', df.head())


### 결측치 처리 
print('df 결측치', df.isna().sum())

# 연속형 변수는 중앙값으로 처리
df['age'] = df['age'].fillna(df.age.median())

# 이산형 변수는 가장 많은 값으로 처리
print(df.deck.value_counts())
df['deck'] = df['deck'].fillna('C')
print(df.embarked.value_counts())
df['embarked'] = df['embarked'].fillna('S')
print(df.embark_town.value_counts())
df['embark_town'] = df['embark_town'].fillna('Southampton')

print('결측치 처리 후 결측치', df.isna().sum())


### 라벨 인코딩
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

cat_col = ['sex', 'embarked', 'class', 'who', 'deck', 'adult_male', 'embark_town', 'alone']
df[cat_col] = df[cat_col].apply(le.fit_transform)
print(df.head())

### 더미 변수화
dum_col = ['class', 'pclass', 'sex']
for i in dum_col:
    df[i] = df[i].astype('category')
df = df.drop(['alive'], axis=1)
df = pd.get_dummies(df)
print(df.head())

### 스케일링
from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
con_col = ['age', 'fare']
scale.fit(df[con_col])
df[con_col] = scale.transform(df[con_col])

### train, test split
X_train, X_test, y_train, y_test = train_test_split(df, df['survived'], test_size=0.2, random_state=42, stratify=df['survived'])
X_train = X_train.drop(['survived'], axis=1)
X_test = X_test.drop(['survived'], axis=1)
print(X_train.head())

### 모형학습
# 랜덤포레스트
from sklearn.ensemble import RandomForestClassifier
model1 = RandomForestClassifier()
model1.fit(X_train, y_train)
pred1 = model1.predict_proba(X_test)
print(pred1[:,1])

# xgboost
from sklearn.ensemble import AdaBoostClassifier
model2 = AdaBoostClassifier()
model2.fit(X_train, y_train)
pred2 = model2.predict_proba(X_test)
print(pred2[:,1])

### 정확도
from sklearn.metrics import roc_auc_score
print('randomforest roc_auc', roc_auc_score(y_test, pred1[:,1]))
print('adaboost roc_auc', roc_auc_score(y_test, pred2[:,1]))

### 답안제출
pd.DataFrame({'id':y_test.index, 'pred':pred1[:,1]}).to_csv('123.csv', index=False)

## 확인
print(pd.read_csv('123.csv'))