import pandas as pd
from keras.layers import Dense, Embedding, Flatten
from keras.models import Sequential
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

data_train = pd.read_csv('./input/train.csv')
data_test = pd.read_csv('./input/test.csv')


def simplify_ages(df):
    df.Age = df.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df


def simplify_cabins(df):
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df


def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df


def format_name(df):
    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])
    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])
    return df


def drop_features(df):
    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)


def transform_features(df):
    df = simplify_ages(df)
    df = simplify_cabins(df)
    df = simplify_fares(df)
    df = format_name(df)
    df = drop_features(df)
    return df


def extract_result(array):
    return array[0]


data_train = transform_features(data_train)
data_test = transform_features(data_test)


def encode_features(df_train, df_test):
    features = ['Fare', 'Cabin', 'Age', 'Sex', 'Lname', 'NamePrefix']
    df_combined = pd.concat([df_train[features], df_test[features]])

    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test


data_train, data_test = encode_features(data_train, data_test)

data_train.head()
X_all = data_train.drop(['Survived', 'PassengerId'], axis=1)

y_all = data_train['Survived']
num_test = 0.20

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)

max_length = 9
model = Sequential()
# 96*7
model.add(Embedding(868, 32, input_length=max_length))
model.add(Flatten())
model.add(Dense(32, input_dim=max_length, activation='relu'))
# model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=60)

score = model.evaluate(X_test, y_test)

# Predict Result
ids = data_test['PassengerId']

predictions = model.predict_classes(data_test.drop('PassengerId', axis=1))

pred = []
[pred.append(pair) for pair in map(extract_result, predictions)]

output = pd.DataFrame({'PassengerId': ids, 'Survived': pred})

output.to_csv('submission.csv', index=False)
