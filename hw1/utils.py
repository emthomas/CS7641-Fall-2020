import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


class Adult(object):
    def __init__(self):
        self.fields = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                       'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                       'native-country', 'income']
        self.categorical_fields = ['income']
        self.nominal_fields = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race',
                               'sex', 'native-country']
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.df = None
        self.y_test = None
        self.target_names = ['<$50k', '>=$50k']
        self.label_encoder = LabelEncoder()
        self.top_10_features = ['fnlwgt', 'age', 'capital-gain', 'hours-per-week', 'is_married-civ-spouse', 'education-num', 'is_husband', 'capital-loss', 'is_never-married', 'is_exec-managerial']
        self.top_10_features_idx = [1, 0, 3, 5, 35, 2, 57, 4, 37, 46]

    def get_data(self, model=None):
        """
        age: continuous.
        workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
        fnlwgt: continuous.
        education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
        education-num: continuous.
        marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
        occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
        relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
        race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
        sex: Female, Male.
        capital-gain: continuous.
        capital-loss: continuous.
        hours-per-week: continuous.
        native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

        :return: X_train, X_test, y_train, y_test
        """

        if self.X_train is not None:
            if model == 'KMeans':
                return self.X_train, self.X_test, self.y_train, self.y_test, self.target_names
            return self.X_train, self.X_test, self.y_train, self.y_test

        df1 = pd.read_csv('../data/adult/adult.data')
        df1.columns = self.fields
        df2 = pd.read_csv('../data/adult/adult.test')
        df2.columns = self.fields
        df = pd.concat([df1, df2])
        self.df = df

        print(df.shape)

        for category in self.nominal_fields:
            dum_df = pd.get_dummies(df, columns=[category], prefix=["is"])
            df = dum_df

        for category in self.categorical_fields:
            df[category] = self.label_encoder.fit_transform(df[category])

        df = df[[c for c in df if c not in ['income']] + ['income']]
        clean_headers = [x.replace(" ","").lower() for x in list(df.columns)]
        df.columns = clean_headers
        # print(clean_headers)
        self.fields = list(df.columns)

        if model == 'KMeans':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()

        X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values

        test_size = 0.2
        if model:
            if 'SVC' in model:
                test_size = 0.8
            elif 'MLP' in model:
                test_size = 0.8
            elif model == 'KMeans':
                test_size = 0.2
            elif 'NN' in model:
                test_size = 0.95

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=7, stratify=y)

        # scaler = StandardScaler()
        self.X_train = scaler.fit_transform(X_train)
        self.X_test = scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test

        if model == 'KMeans':
            return self.X_train, self.X_test, self.y_train, self.y_test, self.target_names

        return self.X_train, self.X_test, self.y_train, self.y_test


class Wine(object):
    def __init__(self):
        self.fields = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
                       "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol",
                       "quality"]
        self.categorical_fields = ['quality']
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def get_data(self, model=None):
        """
        age: continuous.
        workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
        fnlwgt: continuous.
        education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
        education-num: continuous.
        marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
        occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
        relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
        race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
        sex: Female, Male.
        capital-gain: continuous.
        capital-loss: continuous.
        hours-per-week: continuous.
        native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

        :return: X_train, X_test, y_train, y_test
        """

        if self.X_train is not None:
            return self.X_train, self.X_test, self.y_train, self.y_test

        df1 = pd.read_csv('../data/wine/winequality-red.csv')
        df1.columns = self.fields
        df2 = pd.read_csv('../data/wine/winequality-white.csv')
        df2.columns = self.fields
        df = pd.concat([df1])

        bins = (2, 6, 8)
        group_names = ['bad', 'good']
        df['quality'] = pd.cut(df['quality'], bins=bins, labels=group_names)

        for category in self.categorical_fields:
            le = LabelEncoder()
            df[category] = le.fit_transform(df[category])

        X = df.drop('quality', axis=1)
        y = df['quality']

        test_size = 0.8 if 'SVC' in model else 0.2
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=7, stratify=y)
        sc = StandardScaler()

        self.X_train = sc.fit_transform(X_train)
        self.X_test = sc.fit_transform(X_test)
        self.y_train = y_train
        self.y_test = y_test

        return self.X_train, self.X_test, self.y_train, self.y_test


class Diabetes(object):
    def __init__(self):
        self.fields = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
                       "DiabetesPedigreeFunction", "Age", "Outcome"]
        self.categorical_fields = ["outcome"]
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.df = None
        self.target_names = ['negative', 'positive']
        self.label_encoder = LabelEncoder()
        self.top_10_features = ['glucose', 'bmi', 'age', 'diabetespedigreefunction', 'bloodpressure', 'pregnancies', 'skinthickness', 'insulin']
        self.top_10_features_idx = [1, 5, 7, 6, 2, 0, 3, 4]

    def get_data(self, model=None):
        """
        age: continuous.
        workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
        fnlwgt: continuous.
        education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
        education-num: continuous.
        marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
        occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
        relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
        race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
        sex: Female, Male.
        capital-gain: continuous.
        capital-loss: continuous.
        hours-per-week: continuous.
        native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

        :return: X_train, X_test, y_train, y_test
        """

        if self.X_train is not None:
            if model == 'KMeans':
                return self.X_train, self.X_test, self.y_train, self.y_test, self.target_names
            return self.X_train, self.X_test, self.y_train, self.y_test

        df1 = pd.read_csv('../data/diabetes/diabetes.csv')
        df1.columns = self.fields
        df = pd.concat([df1])

        clean_headers = [x.replace(" ", "").lower() for x in list(df.columns)]
        df.columns = clean_headers
        # print(clean_headers)
        self.fields = list(df.columns)

        df["outcome"].replace({1: "positive", 0: "negative"}, inplace=True)
        self.df = df
        print(df.shape)

        for category in self.categorical_fields:
            df[category] = self.label_encoder.fit_transform(df[category])

        if model == 'KMeans':
            scaler = MinMaxScaler(feature_range=(0, 1))
        else:
            scaler = StandardScaler()

        X_un, y_un = df.iloc[:, :-1].values, df.iloc[:, -1].values

        if model == 'KMeans' or model == 'lda':
            X_up, y_up = resample(X_un[y_un == 1],
                                  y_un[y_un == 1],
                                  replace=True,
                                  n_samples=X_un[y_un == 0].shape[0],
                                  random_state=123)

        try:
            X = np.vstack((X_un[y_un == 0], X_up))
            y = np.hstack((y_un[y_un == 0], y_up))
        except Exception as e:
            X = X_un
            y = y_un
            print(e)

        test_size = 0.1
        if model:
            if 'SVC' in model:
                test_size = 0.2
            elif 'MLP' in model:
                test_size = 0.5
            elif 'NN_' in model:
                test_size = 0.2
            elif 'NN' in model:
                test_size = 0.1

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=7, stratify=y)

        # scaler = StandardScaler()
        self.X_train = scaler.fit_transform(X_train)
        self.X_test = scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test

        if model == 'KMeans':
            return self.X_train, self.X_test, self.y_train, self.y_test, self.target_names

        return self.X_train, self.X_test, self.y_train, self.y_test


class Credit(object):
    def __init__(self):
        self.fields = ['id', 'limit_bal', 'sex', 'education', 'marriage', 'age', 'pay_0', 'pay_2', 'pay_3', 'pay_4',
                       'pay_5', 'pay_6', 'bill_amt1', 'bill_amt2', 'bill_amt3', 'bill_amt4', 'bill_amt5', 'bill_amt6',
                       'pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6',
                       'default payment next month']
        self.categorical_fields = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def get_data(self, model=None):
        """
        age: continuous.
        workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
        fnlwgt: continuous.
        education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
        education-num: continuous.
        marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
        occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
        relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
        race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
        sex: Female, Male.
        capital-gain: continuous.
        capital-loss: continuous.
        hours-per-week: continuous.
        native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

        :return: X_train, X_test, y_train, y_test
        """

        if self.X_train is not None:
            return self.X_train, self.X_test, self.y_train, self.y_test

        df1 = pd.read_csv('../data/credit/credit.csv')
        df1.columns = self.fields
        df = pd.concat([df1])

        for category in self.categorical_fields:
            le = LabelEncoder()
            df[category] = le.fit_transform(df[category])

        X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
        test_size = 0.8 if 'SVC' in model else 0.2
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=7, stratify=y)

        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(X_train)
        self.X_test = scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test

        return self.X_train, self.X_test, self.y_train, self.y_test


class Config(object):
    def __init__(self, name, estimator, cv, params=None):
        self.name = name
        self.estimator = estimator
        self.cv = cv
        self.params = params
