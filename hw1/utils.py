import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


class Adult(object):
    def __init__(self):
        self.fields = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                       'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                       'native-country', 'income']
        self.categorical_fields = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race',
                                   'sex', 'native-country', 'income']
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

        df1 = pd.read_csv('../data/adult/adult.data')
        df1.columns = self.fields
        df2 = pd.read_csv('../data/adult/adult.test')
        df2.columns = self.fields
        df = pd.concat([df1, df2])

        print(df.shape)

        for category in self.categorical_fields:
            le = LabelEncoder()
            df[category] = le.fit_transform(df[category])

        X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
        test_size = 0.2
        if model:
            if 'SVC' in model:
                test_size = 0.8
            elif 'MLP' in model:
                test_size = 0.8
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=7, stratify=y)

        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(X_train)
        self.X_test = scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test

        return self.X_train, self.X_test, self.y_train, self.y_test


class Wine(object):
    def __init__(self):
        self.fields = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"]
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
        self.fields = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
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

        df1 = pd.read_csv('../data/diabetes/diabetes.csv')
        df1.columns = self.fields
        df = pd.concat([df1])

        print(df.shape)

        for category in self.categorical_fields:
            le = LabelEncoder()
            df[category] = le.fit_transform(df[category])

        X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
        test_size = 0.2
        if model:
            if 'SVC' in model:
                test_size = 0.2
            elif 'MLP' in model:
                test_size = 0.5

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=7, stratify=y)

        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(X_train)
        self.X_test = scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test

        return self.X_train, self.X_test, self.y_train, self.y_test


class Credit(object):
    def __init__(self):
        self.fields = ['id', 'limit_bal', 'sex', 'education', 'marriage', 'age', 'pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6', 'bill_amt1', 'bill_amt2', 'bill_amt3', 'bill_amt4', 'bill_amt5', 'bill_amt6', 'pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6', 'default payment next month']
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
