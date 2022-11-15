import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
from math import sqrt
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance


def normalize(data, selected_features):
    # check if list of features or single feature
    if isinstance(selected_features, list):
        for fcol in selected_features:
            # normalization to range 0-1
            data[fcol] = (data[fcol] - data[fcol].min()) / (data[fcol].max() - data[fcol].min())
        return data
    else:
        data[selected_features] = (data[selected_features] - data[selected_features].min()) / (
                    data[selected_features].max() - data[selected_features].min())
        return data


def log_transform(data, selected_features):
    # check if list of features or single feature
    if isinstance(selected_features, list):
        for fcol in selected_features:
            data[fcol] = (data[fcol] + 1).transform(np.log)
        return data

    else:
        data[selected_features] = (data[selected_features] + 1).transform(np.log)
        return data


def drop_outlier(data, selected_features):
    if isinstance(selected_features, list):
        for fcol in selected_features:
            lower_lim = data[fcol].quantile(0.02)
            upper_lim = data[fcol].quantile(0.98)

            # replace outliers with lower/upper limit
            data.loc[(data[fcol] > upper_lim), fcol] = upper_lim
            data.loc[(data[fcol] < lower_lim), fcol] = lower_lim
        return data
    else:
        lower_lim = data[selected_features].quantile(0.02)
        upper_lim = data[selected_features].quantile(0.98)

        # replace outliers with lower/upper limit
        data.loc[(data[selected_features] > upper_lim), selected_features] = upper_lim
        data.loc[(data[selected_features] < lower_lim), selected_features] = lower_lim
        return data


def drop_replace(data, selected_features):
    if isinstance(selected_features, list):
        for fcol in selected_features:
            lower_lim = data[fcol].quantile(0.02)
            upper_lim = data[fcol].quantile(0.98)

            # replace outliers with NaN
            data.loc[(data[fcol] > upper_lim), fcol] = np.nan
            data.loc[(data[fcol] < lower_lim), fcol] = np.nan
        return data
    else:
        lower_lim = data[selected_features].quantile(0.02)
        upper_lim = data[selected_features].quantile(0.98)

        # replace outliers with NaN
        data.loc[(data[selected_features] > upper_lim), selected_features] = np.nan
        data.loc[(data[selected_features] < lower_lim), selected_features] = np.nan
        return data


def plot_data_dist(data_before, data_mod, selected_features, figures2save, ttv):
    if isinstance(selected_features, list):
        for fcol in selected_features:
            sns.set()
            fig, axes = plt.subplots(1, 2)
            fig.suptitle('Before and After Normalization of related data:{}'.format(ttv))
            sns.distplot(data_before[fcol], ax=axes[0])
            sns.distplot(data_mod[fcol], ax=axes[1])
            # plt.show()
            plt.savefig(figures2save + ttv + '_' + str(fcol) + '_Normalized.png')
            plt.close(fig)
    else:
        sns.set()
        fig, axes = plt.subplots(1, 2)
        fig.suptitle('Before and After Normalization of related data:{}'.format(ttv))
        sns.distplot(data_before[selected_features], ax=axes[0])
        sns.distplot(data_mod[selected_features], ax=axes[1])
        # plt.show()
        plt.savefig(figures2save + ttv + '_' + str(selected_features) + '_Normalized.png')
        plt.close(fig)


'-------------------------------------------------------------------------------------------'


def corr_plot(X_train, data_version, path):
    # Visualise Correlation of Train Set Features
    # use the pands .corr() function to compute pairwise correlations for the dataframe
    corr_train = X_train.corr()
    # visualise the data with seaborn
    plt.figure(figsize=(22, 12))
    sns.set_style(style='white')
    sns.heatmap(corr_train)  # , mask=mask, cmap=cmap, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    plt.title('Correlation plot of data: {}'.format(data_version))
    plt.savefig(path + data_version + '_' + 'Trainset_CorrelationPlot.png')
    plt.close()
    return corr_train


def plot_high_corr(corr_train, data_version, path):
    high = corr_train[corr_train >= .85]
    low = corr_train[corr_train < -.85]
    fig, axes = plt.subplots(1, 2, figsize=(22, 12))
    plt.suptitle('High positive and negative correlation to drop', size=24)
    if high.isnull().values.all():
        pass
    else:
        sns.heatmap(high, cmap="Reds", ax=axes[0])

    if low.isnull().values.all():
        pass
    else:
        sns.heatmap(low, cmap="Greens", ax=axes[1])
    plt.savefig(path + data_version + '_' + 'Trainset_high_pos&neg_corr.png')
    plt.close(fig)


def drop_high_corr(corr_train, data_test, data_train, data_validation):
    upper_tri = corr_train.where(np.triu(np.ones(corr_train.shape), k=1).astype(np.bool))
    to_drop_high = [column for column in upper_tri.columns if any(upper_tri[column] > 0.85)]
    to_drop_low = [column for column in upper_tri.columns if any(upper_tri[column] < -0.85)]

    data_test = data_test.drop(to_drop_high, axis=1)
    data_test = data_test.drop(to_drop_low, axis=1)

    data_train = data_train.drop(to_drop_high, axis=1)
    data_train = data_train.drop(to_drop_low, axis=1)

    data_validation = data_validation.drop(to_drop_high, axis=1)
    data_validation = data_validation.drop(to_drop_low, axis=1)
    return data_test, data_train, data_validation


def r_squared(X_train, y_train):
    cv = KFold(n_splits=10, shuffle=False)
    classifier_pipeline = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10))
    y_pred = cross_val_predict(classifier_pipeline, X_train, y_train, cv=cv)
    rmse = str(round(sqrt(mean_squared_error(y_train, y_pred)), 2))
    rsquared = str(round(r2_score(y_train, y_pred), 2))
    # print("RMSE: " + str(round(sqrt(mean_squared_error(y_train, y_pred)), 2)))
    # print("R_squared: " + str(round(r2_score(y_train, y_pred), 2)))

    return rmse, rsquared


def get_variance(X_train, data_version, path):
    variance = (X_train.var())  # .sort_values()
    # print(variance, '------------------- Variance')
    plt.figure(figsize=(22, 12))
    plt.bar(variance.index, variance)
    plt.xticks(fontsize=8, rotation=30, ha='right')
    plt.title('Variance Distribution')
    plt.xlabel('Features')
    plt.ylabel('Variance Scores')
    plt.savefig(path + data_version + '_' + 'Trainset_variance.png')
    plt.show()
    return variance


def sort_select(features, scores):
    features_fscores = list(zip(features, list(scores)))
    filtered_list = [tup for tup in features_fscores if tup[1] > np.mean(scores)]
    selectec_features = [i[0] for i in filtered_list]
    # print(selectec_features)

    return selectec_features


def anova(X_train, y_train, X_test, data_version, path):
    fs = SelectKBest(score_func=f_classif, k='all')
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    features = fs.get_feature_names_out()
    # transform test input data
    X_test_fs = fs.transform(X_test)

    #for i in range(len(fs.scores_)):
    #    print('Feature %s: %f' % (features[i], fs.scores_[i]))

    selectec_features = sort_select(features, fs.scores_)
    plt.figure(figsize=(22, 12))
    plt.bar(features, fs.scores_)
    plt.xticks(fontsize=8, rotation=30, ha='right')
    plt.title('ANOVA Feature Selection')
    plt.xlabel('Features')
    plt.ylabel('ANOVA f-test Feature Importance')
    plt.savefig(path + data_version + '_' + 'Trainset_ANOVA.png')
    plt.show()
    plt.close()

    fs = SelectKBest(score_func=mutual_info_classif, k='all')
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    features = fs.get_feature_names_out()
    # transform test input data
    X_test_fs = fs.transform(X_test)

    #for i in range(len(fs.scores_)):
    #    print('Feature %s: %f' % (features[i], fs.scores_[i]))

    selectec_features_mif = sort_select(features, fs.scores_)
    plt.figure(figsize=(22, 12))
    plt.bar(features, fs.scores_)
    plt.xticks(fontsize=8, rotation=30, ha='right')
    plt.title('ANOVA Feature Selection - MIF')
    plt.xlabel('Features')
    plt.ylabel('ANOVA Mutual Information')
    plt.savefig(path + data_version + '_' + 'Trainset_ANOVA_MIF.png')
    plt.show()
    plt.close()

    return selectec_features, selectec_features_mif


def mergeAU_1_2(data_test, data_train, data_validation):
    data_test['AU01+2_c_mean'] = data_test['AU01_c_mean'] + data_test['AU02_c_mean']
    data_test['AU01+2_c_mean'] = data_test['AU01+2_c_mean'] / 2
    data_test = data_test.drop(columns=['AU01_c_mean', 'AU02_c_mean'])
    # reorder
    cols = list(data_test.columns)
    cols = [cols[0]] + [cols[-1]] + cols[1:-1]
    # cols= [cols[1:]] + cols[:-1]
    data_test = data_test[cols]

    data_train['AU01+2_c_mean'] = data_train['AU01_c_mean'] + data_train['AU02_c_mean']
    data_train['AU01+2_c_mean'] = data_train['AU01+2_c_mean'] / 2
    data_train = data_train.drop(columns=['AU01_c_mean', 'AU02_c_mean'])
    # reorder
    cols = list(data_train.columns)
    cols = [cols[0]] + [cols[-1]] + cols[1:-1]
    data_train = data_train[cols]

    data_validation['AU01+2_c_mean'] = data_validation['AU01_c_mean'] + data_validation['AU02_c_mean']
    data_validation['AU01+2_c_mean'] = data_validation['AU01+2_c_mean'] / 2
    data_validation = data_validation.drop(columns=['AU01_c_mean', 'AU02_c_mean'])
    # reorder
    cols = list(data_validation.columns)
    cols = [cols[0]] + [cols[-1]] + cols[1:-1]
    data_validation = data_validation[cols]

    return data_test, data_train, data_validation


def feature_importance(X_train, y_train, data_version, path):
    rf = RandomForestRegressor(n_estimators=150)
    rf.fit(X_train, y_train)
    sort = rf.feature_importances_.argsort()
    plt.figure(figsize=(22, 12))
    plt.barh(X_train.columns[sort], rf.feature_importances_[sort])
    plt.xticks(fontsize=5, rotation=30, ha='right')
    plt.title('Feature Importance')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.savefig(path + data_version + '_' + 'Trainset_FeatureImportance.png')
    plt.show()

    return rf.feature_importances_


def rfecv_classic(X_train, y_train):
    from sklearn.feature_selection import RFECV
    rfecv = RFECV(estimator=LogisticRegression(max_iter=5000, solver='liblinear'), step=1, min_features_to_select=5, cv=10, scoring='accuracy')
    rfecv = rfecv.fit(X_train, y_train.values.ravel())
    print('Optimal number of features :', rfecv.n_features_)
    print('Best features :', X_train.columns[rfecv.support_])
    print(rfecv.grid_scores_)

    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (accuracy)")
    plt.title('Model: Logistic Regression')
    min_features_to_select = 5
    #plt.plot(range(min_features_to_select, len(rfe_cv.grid_scores_) + min_features_to_select),
     #   rfe_cv.grid_scores_,)
   #plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), np.mean(rfecv.grid_scores_, axis=1))
    plt.show()


from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
def rfecv(X_train, y_train):
    rfe_cv = RFECV(estimator=LogisticRegression(max_iter=5000, solver='liblinear'), step=1, min_features_to_select=5, cv=3, scoring='accuracy')
    rfe_cv = rfe_cv.fit(X_train, y_train.values.ravel())
    print('Optimal number of features :', rfe_cv.n_features_)
    print('Best features :', X_train.columns[rfe_cv.support_])
    print(rfe_cv.grid_scores_)

    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score of number of selected features")
    plt.plot(range(1, len(rfe_cv.grid_scores_) + 1), rfe_cv.grid_scores_)
    plt.title('Model: Logistic Regression')
    # plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_,
    # #         label='label: %s // # Features = %d' % (y_train.name, rfecv.n_features_))
    # plt.title('OrigData// Label = %s // Model: Logistic Regression // iter=40' % (y_train.name))
    #plt.legend()
    plt.show()

    #
    # x_train_rfecv = rfecv.transform(X_train)
    # x_validation_rfecv = rfecv.transform(X_validation)
    # rfecv_model = model.fit(x_train_rfecv, y_train.values.ravel())
    # print('Logistic Regression Accuracy with Recursive feature elimination with cross validation.')
    # print()
    # print('size = ', len(y_validation), '|  0 = ', len(y_validation) - y_validation.sum(), '|  1 = ',
    #       y_validation.sum())
    # print()
    # generate_accuracy_and_heatmap(rfecv_model, x_validation_rfecv, y_validation.values.ravel())

    # return rfecv.grid_scores_, X_train.columns[rfecv.support_]


def probatus_rfecv(X, y):
    from probatus.feature_elimination import ShapRFECV
    from sklearn.model_selection import RandomizedSearchCV
    import numpy as np
    import pandas as pd
    import lightgbm

    clf = lightgbm.LGBMClassifier(max_depth=5, class_weight='balanced')
    param_grid = {'n_estimators': [5, 7, 10], 'num_leaves': [3, 5, 7, 10]}
    search = RandomizedSearchCV(clf, param_grid, cv=5, scoring='roc_auc', refit=False)

    shap_elimination = ShapRFECV(search, step=0.2, cv=10, scoring='roc_auc', n_jobs=3)
    report = shap_elimination.fit_compute(X, y, check_additivity=False)

    performance_plot = shap_elimination.plot()
