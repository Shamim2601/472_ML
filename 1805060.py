from collections import defaultdict
import copy
import sys
import pandas as pd
import numpy as np
import math
import numbers
import random
from sklearn import metrics
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings("ignore")
from scipy.special import expit  # Sigmoid function



# ------- helper functions and class definitions ---------

def cast_to_type(x):
    try:
        return x.astype('float').astype('Int64')
    except:
        try:
            return x.astype('float')
        except:
            return x

def measure_performance(y_test, y_pred):
    labels = list(set(y_test))
    print('\t\t\t\t [', labels[0],"\t\t",labels[1],"]")
    cm = metrics.confusion_matrix(y_test, y_pred, labels=labels)
    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN) * 100
    print('True positive rate \t\t', TPR)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) * 100
    print('True negative rate \t\t', TNR)

    # Precision or positive predictive value
    PPV = TP/(TP+FP) * 100
    print('Positive predictive value \t', PPV)

    # Negative predictive value
    NPV = TN/(TN+FN) * 100
    # print('True positive rate ', TPR)

    # Fall out or false positive rate
    FPR = FP/(FP+TN) * 100
    # print('True positive rate ', TPR)

    # False negative rate
    FNR = FN/(TP+FN) * 100


    # False discovery rate
    FDR = FP/(TP+FP) * 100
    print('False discovery rate    \t', FDR)

    F1_score = 2 * PPV * TPR / (PPV + TPR) 
    print('F1 score \t\t\t', F1_score,"\n")





# ------- Logistic Regression Implementation --------

class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000, threshold=0.5, feature_subset=10):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.threshold = threshold
        self.feature_subset = feature_subset
        self.weights = None

    def _sigmoid(self, z):
        return expit(z)

    def _select_features(self, X, y):
        # Select features based on information gain
        information_gains = self._calculate_information_gain(X, y)
        selected_features = np.argsort(information_gains)[-self.feature_subset:]
        return selected_features

    def _calculate_information_gain(self, X, y):
        # Calculate information gain for each feature
        gains = []
        for i in range(X.shape[1]):
            feature_values = X[:, i]
            unique_values = np.unique(feature_values)
            gain = 0
            for val in unique_values:
                subset_indices = (feature_values == val)
                entropy = self._calculate_entropy(y[subset_indices])
                weight = np.sum(subset_indices) / len(y)
                gain += weight * entropy
            gains.append(gain)
        return np.array(gains)

    def _calculate_entropy(self, y):
        # Calculate binary entropy
        if len(y) == 0:
            return 0
        p_positive = np.sum(y == 1) / len(y)
        p_negative = 1 - p_positive
        entropy = -p_positive * np.log2(p_positive) - p_negative * np.log2(p_negative)
        return entropy

    def _early_terminate(self, error):
        # Early termination if error is below the threshold
        return self.threshold > 0 and error < self.threshold

    def _gradient_descent(self, X, y):
        # Initialize weights
        self.weights = np.zeros(X.shape[1])

        for _ in range(self.max_iter):
            # Calculate predicted probabilities
            y_pred = self._sigmoid(np.dot(X, self.weights))

            # Calculate error (cross-entropy loss)
            error = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

            # Check for early termination
            if self._early_terminate(error):
                break

            # Calculate gradient
            gradient = np.dot(X.T, y_pred - y) / len(y)

            # Update weights
            self.weights -= self.learning_rate * gradient

    def fit(self, X, y):
        # Select features based on information gain if feature_subset is specified
        if self.feature_subset:
            selected_features = self._select_features(X, y)
            self.selected_features = selected_features
            X = X[:, selected_features]

        # Add bias term
        X = np.c_[np.ones(X.shape[0]), X]

        # Run gradient descent
        self._gradient_descent(X, y)

    def predict(self, X):
        # Select relevant features if feature_subset is specified
        if self.feature_subset:
            X = X[:, self.selected_features]

        # Add bias term
        X = np.c_[np.ones(X.shape[0]), X]

        # Make predictions
        # print(X[0])
        # print(self.weights)
        y_pred = self._sigmoid(np.dot(X, self.weights))
        return (y_pred >= 0.5).astype(int)






# ----------  AdaBoost implelmentation --------------

class AdaBoost:
    def __init__(self, K):
        self.K = K
        self.alphas = []
        self.models = []

    def fit(self, X, y):
        # Initialize weights
        n_samples, n_features = X.shape
        w = np.ones(n_samples) / n_samples

        for k in range(self.K):
            # Create a weak learner (Logistic Regression in this case
            weak_learner = LogisticRegression()

            # Fit the weak learner on the resampled data
            weak_learner.fit(X, y)

            # Predict on the training data
            y_pred = weak_learner.predict(X)

            # Compute weighted error
            error = np.sum(w * (y_pred != y)) / np.sum(w)

            # Compute the classifier weight
            alpha = 0.5 * np.log((1 - error) / error)

            # Update sample weights
            w = w * np.exp(-alpha * y * y_pred)
            w /= np.sum(w)

            # Store the weak learner and its weight
            self.models.append(weak_learner)
            self.alphas.append(alpha)

    def predict(self, X):
        # Make predictions using all weak learners
        pred = np.zeros(X.shape[0])
        for alpha, model in zip(self.alphas, self.models):
            pred += alpha * model.predict(X)

        # Convert predictions to binary
        return np.sign(pred)


    
missing_val = ["n/a", "na", "--","NA","N/A","?"]


## telco dataset
def read_telco():
    telecom_cust = pd.read_csv('./datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv', na_values = missing_val)
    # print(telecom_cust.head())
    # print(telecom_cust.dtypes)

    # Converting Total Charges to a numerical data type.
    telecom_cust.TotalCharges = pd.to_numeric(telecom_cust.TotalCharges, errors='coerce')
    # print(telecom_cust.isnull().sum())
    #Removing missing values 
    telecom_cust.dropna(inplace = True)
    #Remove customer IDs from the data set
    df2 = telecom_cust.iloc[:,1:]
    #Converting the predictor variable in a binary numeric variable
    df2['Churn'].replace(to_replace='Yes', value=1, inplace=True)
    df2['Churn'].replace(to_replace='No',  value=0, inplace=True)

    num_col = ['tenure', 'MonthlyCharges', 'TotalCharges']
    cat_col = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

    df3 = pd.get_dummies(df2, columns = cat_col)

    for column in num_col:
        bins = np.linspace(min(df3[column]), max(df3[column]), 4)
        df3[column] = pd.cut(df3[column], bins,labels=False, include_lowest = True)
    df = pd.get_dummies(df3, columns = num_col)

    column_to_move = 'Churn'

    # Ensure the column exists in the DataFrame
    if column_to_move in df.columns:
        # Get the list of column names excluding the one to be moved
        columns = [col for col in df.columns if col != column_to_move]
        
        # Reorder the columns putting the specified column at the end
        df = df[columns + [column_to_move]]

    return df


## Adult Dataset
def read_adult():
    df = pd.read_csv('./datasets/adult.csv', na_values = missing_val, skip_blank_lines=True)
    df.dropna(inplace = True)

    df.drop(columns=['native-country'], inplace=True) # Usa mainly

    df['income'].replace(to_replace='>50K', value=1, inplace=True)
    df['income'].replace(to_replace='<=50K',  value=0, inplace=True)

    num_col = ['age','fnlwgt','educational-num','capital-gain','capital-loss','hours-per-week']
    cat_col = ['workclass','education','marital-status','occupation','relationship','race','gender']

    df2 = pd.get_dummies(df, columns = cat_col)

    for column in num_col:
        bins = np.linspace(min(df2[column]), max(df2[column]), 4)
        df2[column] = pd.cut(df2[column], bins,labels=False, include_lowest = True)
    df = pd.get_dummies(df2, columns = num_col)

    column_to_move = 'income'

    # Ensure the column exists in the DataFrame
    if column_to_move in df.columns:
        # Get the list of column names excluding the one to be moved
        columns = [col for col in df.columns if col != column_to_move]
        
        # Reorder the columns putting the specified column at the end
        df = df[columns + [column_to_move]]

    # print(df.head())
    # sys.exit()

    return df



## creditcard dataset
def read_creditcard():
    df = pd.read_csv('./datasets/creditcard.csv', na_values = missing_val, skip_blank_lines=True)
    df = df.drop(columns=['Time'])
    pos_sample = df[df.iloc[:,-1] == 1]
    neg_sample = df[df.iloc[:,-1] == 0].sample(n=20000, random_state=0)
    df = shuffle(pd.concat( (pos_sample,neg_sample), axis=0), random_state=0) 
    df.reset_index(drop=True)

    return df




if __name__ == '__main__':

    # ------ ---- reading dataset  --------------
    # comment out any two line to read the other dataaset

    df = read_telco()
    # df = read_adult()
    # df = read_creditcard()


    # ------------  pre-processing  ---------------

    # remove/replace spaces
    df = df.applymap(lambda x: x.strip().lower() if isinstance(x, str) else x)
    df.replace(r'^[\s?]*$', np.nan, regex=True, inplace = True)

    # remove redundant columns
    for col in df.columns:
        if len(df[col].unique()) == 1:  # only 1 value
            print(col, " has only 1 value, so removed\n")
            df.drop(col,inplace=True,axis=1)

    for col in df.columns:
        if len(df[col].unique()) == len(df[col]):  # all values are unique
            print(col, " has entirely unique value, so removed\n")
            df.drop(col,inplace=True,axis=1)
    # print(df.head())

    df.apply(cast_to_type) # casting to numeric
    for col in df.columns:
        if df[col].isnull().any() :
            if np.issubdtype(df[col].dtype, np.number):
                df[col] = df[col].fillna(df[col].mean()) # replace missing vals with mean

    # drop null columns and rows
    nullThreshold = 0.7
    df.dropna(thresh=df.shape[0]*nullThreshold,axis=1, inplace= True) # columns having non-null values less than threshold
    df.dropna(axis = 0, inplace = True) # rows having any null values



    # -------- Splitting for training -----------

    # train-test splitting
    train, test = train_test_split(df,test_size=0.2, random_state=0)
    # print(test.info())
    x_train, y_train = train.drop(train.columns[-1], axis=1).values,  train[train.columns[-1]].values
    x_test, y_test = test.drop(test.columns[-1], axis=1).values,  test[test.columns[-1]].values
    # After this, we can use x_train, y_train, x_test, and y_test as input for training and evaluating machine learning models. 
    # Each element in x_train and x_test represents a row of features, and the corresponding element in y_train and y_test represents the label for that row.




    # ---------- Apply Logistic Regression : Performance  ------------

    # training
    LRC = LogisticRegression()
    LRC.fit(x_train, y_train)

    # testing
    # print(x_train.shape)
    y_train_pred = LRC.predict(x_train)
    y_pred = LRC.predict(x_test)

    print("------- Logistic Regression Performance ------------")
    print('\nTraining set performance')
    print("Accuracy: \t\t\t",metrics.accuracy_score(y_train,y_train_pred) * 100, "\n")
    measure_performance(y_train, y_train_pred)

    print('\nTesting set performance')
    print("Accuracy: \t\t\t", metrics.accuracy_score(y_test,y_pred) * 100, "\n")
    measure_performance(y_test, y_pred)




    # ------- Apply AdaBoost : Performance -------------
    
    print("------- AdaBoost Performance ------------")

    for K in [5, 10, 15, 20]:
      boost = AdaBoost(K)

      boost.fit(x_train, y_train)

      y_pred = boost.predict(x_test)
      y_train_pred = boost.predict(x_train)

      print('\nNo of rounds:  \t', K)
      print('Training set performance')
      print("Accuracy: \t", metrics.accuracy_score(y_train,y_train_pred) * 100)

      print('Testing set performance')
      print("Accuracy: \t", metrics.accuracy_score(y_test,y_pred) * 100)
      