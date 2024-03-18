
from sklearn.svm import SVC
from settings.loaders import load_mixeddata
from settings.utils_features import *

matplotlib.use('TkAgg')

sys.path.insert(1,  r'C:\Users\ao4518\Desktop\PHD\distance')

from settings.utils_plots import *

if __name__ == '__main__':

    set = ['curve_mean', 'curve_sampen', 'deltaheading_iqr', 'deltaheading_mean', 'heading_std', 'quality_sampen']

    path = MOTHERPATH + '/results_featureselection/'
    selection_sweden, selection_oxford = load_mixeddata()
    selection = np.concatenate([selection_sweden, selection_oxford])

    filenames = []
    target = []
    for s in selection:
        filenames.append(s.filename)
        if s.dataset == 'oxford':
            if 'unconventional' in s.filename:
                target.append(1) # bad
            else:
                target.append(0) # good
        elif s.dataset == 'sweden':
            if 'good'in s.filename or 'stop' in s.filename:
                target.append(0)
            else:
                target.append(1)

    features = obtain_onlyfeatures(selection, set)

    # drop filename column
    filenames = features['filename']
    features.drop(columns=['filename'], inplace=True)
    scaler = StandardScaler()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    features_scaled = scaler.fit_transform(features)
    features_scaled = pd.DataFrame(features_scaled, columns=features.columns)

    if True:
        # Fixing multicollinearity with VIF
        features_scaled['curve_mean_sampen'] = features_scaled['curve_mean'] + features_scaled['curve_sampen']
        features_scaled.drop(columns=['curve_mean'], inplace=True)
        features_scaled.drop(columns=['curve_sampen'], inplace=True)
        features_scaled['deltaheading_mean_iqr'] = features_scaled['deltaheading_mean'] + features_scaled['deltaheading_iqr']
        features_scaled.drop(columns=['deltaheading_mean'], inplace=True)
        features_scaled.drop(columns=['deltaheading_iqr'], inplace=True)

    all_ytrue=[]
    all_yprob=[]
    all_yprob_train=[]
    all_ytrue_train=[]
    parameters_list=[]
    impo_features=[]

    fig_train = plt.figure()
    fig_dimrow = 2
    fig_dimcol = 2

    ### Modelling
    #1.
    title='Logistic Regression'
    model = LogisticRegression(max_iter=2500, class_weight='balanced', random_state=1)
    space = dict()
    space['penalty'] = ['None', 'l1', 'l2', 'elasticnet']
    space['C'] = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
    param = utils_plots.gridsearch_model(model, features_scaled, target, space, cv)
    print(title +': '+ str(param))
    model = LogisticRegression(penalty=param['penalty'], C=param['C'], random_state=1, class_weight='balanced', max_iter=2500)
    lst_ytrue, lst_ypred, y_prob, model = modelling_cross_return(features=features_scaled, y=target, title=title, mdl=model, cv=cv, scaler=None)
    all_yprob_train.append(np.concatenate(y_prob))
    all_ytrue_train.append(np.concatenate(lst_ytrue))
    parameters_list.append(param)

    ax2 = fig_train.add_subplot(fig_dimrow,fig_dimcol, 1)
    report_performance_ax(np.concatenate(lst_ytrue), np.concatenate(lst_ypred), title=title + '', ax=ax2)

    if False:
        print("Logistic regression coefficients: ")
        coefs = model.coef_[0]
        intercept = model.intercept_
        print(model.coef_)
        print(model.intercept_)
        exponentials = []
        percentages = []
        for g in range(len(coefs)):
            exp_val = np.exp(coefs[g])
            exponentials.append(exp_val)
            val_percentage = (exp_val-1)*(100)
            percentages.append(val_percentage)
            print(f"{features_scaled.columns[g]}: percentage value {np.round(val_percentage, 2)}% , original coefficient: {np.round(coefs[g], 2)}, exponential value: {np.round(exp_val, 2)}")

        # plot of the exponential values and percentages (in the same plot)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.bar(features_scaled.columns, exponentials, color='#658EA9')
        ax.set_xticklabels(features_scaled.columns, rotation=20)
        ax.axhline(y=0, color='black', linestyle='--')
        ax.set_ylabel('Odds ratio')
        ax.set_title('Logistic regression coefficients\' importance')
        plt.show()

    #2.
    model = SVC()
    title='SVM'
    space = {'C': [ 1e-2, 1e-1, 1, 10, 100],
             'gamma': [10, 1, 0.1, 0.001, 0.0001],
             'kernel': ['linear', 'rbf']}
    param = utils_plots.gridsearch_model(model, features_scaled, target, space, cv)
    print(title +': '+ str(param))
    model = SVC(class_weight='balanced', random_state=1, probability=True, C=float(param['C']), gamma=param['gamma'], kernel=param['kernel'])
    lst_ytrue, lst_ypred, y_prob, model = modelling_cross_return(features=features_scaled, y=target, title=title, mdl=model, cv=cv, scaler=None)
    ax = fig_train.add_subplot(fig_dimrow,fig_dimcol, 2)
    report_performance_ax(np.concatenate(lst_ytrue), np.concatenate(lst_ypred), title=title + '', ax=ax)
    all_yprob_train.append(np.concatenate(y_prob))
    all_ytrue_train.append(np.concatenate(lst_ytrue))
    parameters_list.append(param)

    #3.
    title='Random Forest'
    model = RandomForestClassifier(max_depth=2, class_weight='balanced', random_state=1)
    space = {
        'max_features': ['sqrt', 'log2'],
        'criterion': ['gini', 'entropy'],
    }
    param = utils_plots.gridsearch_model(model, features, target, space, cv)
    print(title +': '+ str(param))
    model = RandomForestClassifier(criterion=param['criterion'], max_features=param['max_features'], max_depth=3,
                                   class_weight='balanced', random_state=1)
    lst_ytrue, lst_ypred, y_prob, model = modelling_cross_return(features=features_scaled, y=target, title=title, mdl=model, cv=cv, scaler=None)
    all_yprob_train.append(np.concatenate(y_prob))
    all_ytrue_train.append(np.concatenate(lst_ytrue))
    ax = fig_train.add_subplot(fig_dimrow, fig_dimcol, 3)
    report_performance_ax(np.concatenate(lst_ytrue), np.concatenate(lst_ypred), title=title + '', ax=ax)
    parameters_list.append(param)

    algo = ['Logistic Regr', 'SVM', 'RandomForest']
    ax = fig_train.add_subplot(fig_dimrow,fig_dimcol, 4)
    utils_plots.roccurves_plot_ax(all_ytrue_train, all_yprob_train, algo, ax)
