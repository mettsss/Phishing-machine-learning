import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import codecs

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from collections import Counter
from imblearn.combine import SMOTETomek
import warnings

warnings.filterwarnings("ignore")


def load_raw_dataset(filename=None):
    f = codecs.open(filename, 'r', 'utf-8', errors='replace')
    df = pd.read_csv(f, index_col=False)
    df = df.iloc[:, 1:].drop(columns=['url'])
    df_X = df.iloc[:, 1:]
    df_y = df.iloc[:, 0]
    X = pd.DataFrame.to_numpy(df_X)
    y = pd.DataFrame.to_numpy(df_y)
    f.close()
    return df_X, df_y, X, y


def plot_confusion_matrix(cm_input, title='Normalized Confusion Matrix'):
    labels = ['phishing', 'legitimate']
    cm = cm_input.astype('float') / cm_input.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.text(-0.15, 0.05, '%.3f' % cm[0][0], fontsize=14, color='w')
    plt.text(0.8, 0.05, '%.3f' % cm[0][1], fontsize=14, color='k')
    plt.text(-0.15, 1.05, '%.3f' % cm[1][0], fontsize=14, color='k')
    plt.text(0.8, 1.05, '%.3f' % cm[1][1], fontsize=14, color='w')
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    np.set_printoptions(precision=2)
    plt.show()


if __name__ == "__main__":
    file = '../final_data.csv'

    df_X, df_y, X, y = load_raw_dataset(filename=file)

    for i in range(len(y)):
        if y[i] == 'legitimate':
            y[i] = '1'
        else:
            y[i] = '0'

    print('Original dataset shape %s\n' % Counter(y))

    smt = SMOTETomek(random_state=42)

    X, y = smt.fit_resample(X, y)

    print('Resampled dataset shape %s\n' % Counter(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    # print('\n------Decision Tree------\n')
    #
    # clf1 = DecisionTreeClassifier()
    # clf1.fit(X_train, y_train)
    #
    # y_pred = clf1.predict(X_test)
    #
    # print('\nDecision Tree accuracy score %s%%\n' % accuracy_score(y_test, y_pred))
    #
    # plot_confusion_matrix(cm_input=confusion_matrix(y_test, y_pred), title="Decision Tree Confusion Matrix")
    #
    # print(classification_report(y_test, y_pred))
    #
    # feat_dict = {}
    # for col, val in sorted(zip(df_X.columns, clf1.feature_importances_), key=lambda x: x[1], reverse=True):
    #     feat_dict[col] = val
    #
    # feat_df = pd.DataFrame({'Feature': feat_dict.keys(), 'Importance': feat_dict.values()})
    #
    # values = feat_df.Importance
    # idx = feat_df.Feature
    # plt.figure(figsize=(10, 8))
    # clrs = ['green' if (x < max(values)) else 'red' for x in values]
    # sns.barplot(y=idx, x=values, palette=clrs).set(title='Feature Importance')
    # plt.show()

    print('\n------SVM (linear)------\n')

    clf2 = LinearSVC()
    clf2.fit(X_train, y_train)

    y_pred = clf2.predict(X_test)

    print('\nSVM (linear) accuracy score %s%%\n' % accuracy_score(y_test, y_pred))

    plot_confusion_matrix(cm_input=confusion_matrix(y_test, y_pred), title="SVM (linear) Confusion Matrix")

    print(classification_report(y_test, y_pred))

    feat_dict = {}
    coef = clf2.coef_.reshape(22,)
    for col, val in sorted(zip(df_X.columns, coef), key=lambda x: x[1], reverse=True):
        feat_dict[col] = val

    feat_df = pd.DataFrame({'Feature': feat_dict.keys(), 'Coefficient': feat_dict.values()})

    values = feat_df.Coefficient
    idx = feat_df.Feature
    plt.figure(figsize=(10, 8))
    clrs = ['green' if (x < max(values)) else 'red' for x in values]
    sns.barplot(y=idx, x=values, palette=clrs).set(title='Feature Importance')
    plt.show()

    # print('\n------SVM (rbf)------\n')
    #
    # clf3 = SVC()
    # clf3.fit(X_train, y_train)
    #
    # y_pred = clf3.predict(X_test)
    #
    # print('\nSVM (rbf) accuracy score %s%%\n' % accuracy_score(y_test, y_pred))
    #
    # plot_confusion_matrix(cm_input=confusion_matrix(y_test, y_pred), title="SVM (rbf) Confusion Matrix")
    #
    # print(classification_report(y_test, y_pred))