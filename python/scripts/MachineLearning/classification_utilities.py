from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import pandas as pd
import numpy as np   
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.feature_selection import RFECV
from sklearn.svm import NuSVC
from scipy.stats import uniform, truncnorm, randint
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import get_scorer, make_scorer, accuracy_score, precision_score

def evaluate_metrics(estimator,X,y,kpi):
    return get_scorer(kpi).__call__(estimator=estimator,X=X,y_true=y)

def custom_score_function(y_true, y_pred,sample_weight):
    return precision_score(y_true, y_pred,average='weighted',
                      sample_weight=sample_weight.loc[y_true.index],zero_division=0)
    
def custom_scorer(sample_weights):
    return make_scorer(custom_score_function,greater_is_better=True,
                       needs_proba=False,needs_threshold=False,sample_weight = sample_weights)
    
def imprimir_resultados(nombre_modelo,classifier,estimador,x_train,ytrain,x_test,ytest,kpi_train,kpi_weighted):
    score_cv = classifier.best_score_
    
    train_metrics_original = evaluate_metrics(estimator = estimador ,X=x_train,y=ytrain,kpi=kpi_train)
    train_metrics_weighted = evaluate_metrics(estimator = estimador ,X=x_train,y=ytrain,kpi=kpi_weighted)
    
    test_metrics_original = evaluate_metrics(estimator = estimador ,X=x_test,y=ytest,kpi=kpi_train)
    test_metrics_weighted = evaluate_metrics(estimator = estimador ,X=x_test,y=ytest,kpi=kpi_weighted)
    
    print(nombre_modelo+" F.O. cv " + " = %.3f" % (score_cv))
    print (nombre_modelo + " F.O. train weighted = %.3f unweighted  = %.3f " % (train_metrics_weighted,train_metrics_original))
    print (nombre_modelo + " F.O. test weighted = %.3f unweighted  = %.3f " % (test_metrics_weighted,test_metrics_original))
    
    
    
def calcular_rf (x_train, ytrain, num_iters,num_folds_rs,randomState,kpi_train,x_test,ytest,sample_weights=pd.Series([])):
    model = RandomForestClassifier(n_jobs=1,random_state=0,class_weight="balanced", criterion="gini")

    # parameters
    n_estimators = randint(10,2000) # A más árboles, menos overfitting
    max_depth = randint(1,32)
    min_samples_split = truncnorm(a=0, b=1, loc=0.25, scale=0.1)
    min_samples_leaf = truncnorm(a=0, b=1, loc=0.15, scale=0.1)
    max_features = truncnorm(a=0, b=1, loc=0.25, scale=0.1)
    bootstrap = [True, False]
    oob_score = [True,False]
    ccp_alpha = truncnorm(a=0, b=1, loc=0.020, scale=0.010)
    # Create the random grid
    parameters = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap,
                'oob_score':oob_score,
                'ccp_alpha':ccp_alpha}

    
    if sample_weights.empty:
        my_scorer = kpi_train
    else:
        my_scorer = custom_scorer(sample_weights=sample_weights)
    
    classifier = RandomizedSearchCV(model,parameters, n_iter = num_iters ,
                                    scoring = my_scorer,n_jobs=-1, cv=num_folds_rs,random_state=randomState)
    classifier.fit(x_train,ytrain)

    df_importancia_variables = pd.DataFrame(classifier.best_estimator_.feature_importances_,x_train.columns,columns=["valor"])
    df_importancia_variables.sort_values("valor",ascending=False,inplace=True)
    
    estimador = classifier.best_estimator_

    imprimir_resultados(nombre_modelo= "RF",classifier=classifier,estimador=estimador,x_train=x_train,
                        ytrain=ytrain,x_test=x_test,ytest=ytest,kpi_train=kpi_train,kpi_weighted=my_scorer)
    
    return classifier, df_importancia_variables

def calcular_xgb (x_train, ytrain, num_iters,num_folds_rs,randomState,kpi_train,x_test,ytest,sample_weights=pd.Series([])):
    #Definir modelo de clasificación y eliminación de variables
    model = XGBClassifier(random_state=0,n_jobs=1,objective="binary:logistic")

    # parameters
    n_estimators = randint(100,2000)# A más árboles, menos overfitting
    max_depth = randint(1,32)
    learning_rate = truncnorm(a=0, b=1, loc=0.15, scale=0.1)
    gamma = truncnorm(a=0, b=5, loc=0.5, scale=0.3)
    min_child_weight = randint(1,12)
    max_delta_step  = randint(1,7)
    reg_alpha = uniform(0.01, 0.99)
    colsample_bytree= uniform(0.01, 0.99)
    max_delta_step = randint(1,9)
    subsample = uniform(0.01, 0.99)

    # Create the random grid
    parameters = {'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'gamma': gamma,
                'min_child_weight':min_child_weight,
                "max_delta_step":max_delta_step,
                "reg_alpha":reg_alpha,
                "colsample_bytree":colsample_bytree,
                'max_delta_step':max_delta_step,
                'subsample':subsample
                }

    if sample_weights.empty:
        my_scorer = kpi_train
    else:
        my_scorer = custom_scorer(sample_weights=sample_weights)
    
    classifier = RandomizedSearchCV(model,parameters, n_iter = num_iters ,scoring = my_scorer,n_jobs=-1, cv=num_folds_rs,random_state=randomState)
    classifier.fit(x_train,ytrain )
    
    df_importancia_variables = pd.DataFrame(classifier.best_estimator_.feature_importances_,x_train.columns,columns=["valor"])

    estimador = classifier.best_estimator_

    imprimir_resultados(nombre_modelo= "XGB",classifier=classifier,estimador=estimador,x_train=x_train,
                        ytrain=ytrain,x_test=x_test,ytest=ytest,kpi_train=kpi_train,kpi_weighted=my_scorer)
    
    return classifier, df_importancia_variables

def calcular_catboost (x_train, ytrain, x_test,ytest, num_iters,randomState,kpi_train):
    
    model_cat = CatBoostClassifier(iterations=num_iters, 
                            task_type="GPU",
                            devices='0:1',eval_metric='BalancedAccuracy')

    train_dataset = Pool(data=x_train,
                        label=ytrain)

    eval_dataset = Pool(data=x_test,
                        label=ytest)   
                        
    model_cat.fit(train_dataset,use_best_model=True,eval_set=eval_dataset,verbose=False)  

    print(model_cat.best_score_)

    
    df_importancia_variables = pd.DataFrame(model_cat.feature_importances_,x_train.columns,columns=["valor"])
    df_importancia_variables.sort_values("valor",ascending=False,inplace=True)

    estimador = model_cat
    print("Catboosting F.O. train " + " = %.3f" % (evaluate_metrics(estimator = estimador ,X=x_train,y=ytrain,kpi=kpi_train.lower())))
    print("Catboosting F.O. test " + " = %.3f" % (evaluate_metrics(estimator = estimador,X=x_test,y=ytest,kpi=kpi_train.lower())))
    
    return model_cat, df_importancia_variables

def calcular_svc (x_train, ytrain, num_iters, num_folds_rfe,num_folds_rs,randomState,kpi_train,x_test,ytest,sample_weights=pd.Series([])):
    #Definir modelo de clasificación y eliminación de variables
    model = NuSVC(kernel='linear',class_weight = "balanced",random_state=randomState,probability=True)
    model_rfe = RFECV(estimator = model,step=4,min_features_to_select=6,cv=num_folds_rfe,scoring=kpi_train,n_jobs=1)

    #parameters
    nu = uniform(loc=0.0,scale=0.95)
    shrinking = [True,False]

    # Create the random grid
    parameters = {'estimator__nu': nu,
                'estimator__shrinking': shrinking
                }

    if sample_weights.empty:
        my_scorer = kpi_train
    else:
        my_scorer = custom_scorer(sample_weights=sample_weights)
    
    classifier = RandomizedSearchCV(model_rfe,parameters, n_iter = num_iters ,scoring = my_scorer,n_jobs=-1, cv=num_folds_rs,random_state=randomState)
    classifier.fit(x_train,ytrain)

    selected = pd.DataFrame({"variable":x_train.columns[classifier.best_estimator_.support_==True],
    "Seleccionada":classifier.best_estimator_.support_[classifier.best_estimator_.support_==True],
    "importancia":classifier.best_estimator_.estimator_.coef_[0]})
    
    estimador = classifier.best_estimator_.estimator_
 
    imprimir_resultados(nombre_modelo= "SVC",classifier=classifier,estimador=estimador,x_train=x_train[selected.variable],
                        ytrain=ytrain,x_test=x_test[selected.variable],ytest=ytest,kpi_train=kpi_train,kpi_weighted=my_scorer)
    
    return classifier, selected

def calcular_SGD(x_train, ytrain, num_iters, num_folds_rfe,num_folds_rs,randomState,kpi_train,x_test,ytest,sample_weights=pd.Series([])):
    #Definir modelo de clasificación y eliminación de variables
    model = SGDClassifier(class_weight = 'balanced',random_state = randomState,penalty='elasticnet')
    model_rfe = RFECV(estimator = model,step=4,min_features_to_select=6,cv=num_folds_rfe,scoring=kpi_train,n_jobs=1)

    #Parameters
    loss = ['modified_huber','log']
    alpha = learning_rate = truncnorm(a=0, b=0.2, loc=0.0025, scale=0.001)
    l1_ratio = uniform(0.01, 0.99)
    shuffle = [True,False]
    learning_rate = ['optimal','adaptive','constant']
    eta0 = truncnorm(a=0, b=0.2, loc=0.001, scale=0.001)

    # Create the random grid
    parameters = {'estimator__loss': loss,
                'estimator__alpha': alpha,
                'estimator__l1_ratio': l1_ratio,
                'estimator__shuffle': shuffle,
                "estimator__learning_rate":learning_rate,
                "estimator__eta0":eta0}

    if sample_weights.empty:
        my_scorer = kpi_train
    else:
        my_scorer = custom_scorer(sample_weights=sample_weights)
        
    classifier = RandomizedSearchCV(model_rfe,parameters, n_iter = num_iters ,scoring = my_scorer,n_jobs=-1, cv=num_folds_rs,random_state=randomState)
    classifier.fit(x_train,ytrain)
    
    
    
    selected = pd.DataFrame({"variable":x_train.columns[classifier.best_estimator_.support_==True],
    "Seleccionada":classifier.best_estimator_.support_[classifier.best_estimator_.support_==True],
    "importancia":classifier.best_estimator_.estimator_.coef_[0]})
    
    estimador = classifier.best_estimator_.estimator_
 
    imprimir_resultados(nombre_modelo= "SGD",classifier=classifier,estimador=estimador,x_train=x_train[selected.variable],
                        ytrain=ytrain,x_test=x_test[selected.variable],ytest=ytest,kpi_train=kpi_train,kpi_weighted=my_scorer)
    
    
    return classifier, selected

def calcular_logit (x_train, ytrain, num_iters, num_folds_rfe,num_folds_rs,randomState,kpi_train,x_test,ytest,sample_weights=pd.Series([])):
    
    #Definir modelo de clasificación y eliminación de variables
    model = LogisticRegression(random_state=0,class_weight='balanced',solver='saga',n_jobs=1)
    model_rfe = RFECV(estimator = model,step=4,min_features_to_select=6,cv=num_folds_rfe,scoring=kpi_train,n_jobs=1)

    #parameters
    penalty = ['l1','elasticnet']
    C = uniform(0.01, 1.99)
    max_iter=randint(50,600)
    l1_ratio = uniform(0.01, 0.99)

    # Create the random grid
    parameters = {'estimator__penalty': penalty,
                'estimator__C': C,
                'estimator__max_iter': max_iter,
                'estimator__l1_ratio': l1_ratio
                }
    
    if sample_weights.empty:
        my_scorer = kpi_train
    else:
        my_scorer = custom_scorer(sample_weights=sample_weights)
        
    classifier = RandomizedSearchCV(model_rfe,parameters, n_iter = num_iters ,scoring = my_scorer,n_jobs=-1, cv=num_folds_rs,random_state=randomState)
    classifier.fit(x_train,ytrain)

    selected = pd.DataFrame({"variable":x_train.columns[classifier.best_estimator_.support_==True],
    "Seleccionada":classifier.best_estimator_.support_[classifier.best_estimator_.support_==True],
    "importancia":classifier.best_estimator_.estimator_.coef_[0]})
    
    estimador = classifier.best_estimator_.estimator_
        
    imprimir_resultados(nombre_modelo= "Logit",classifier=classifier,estimador=estimador,x_train=x_train[selected.variable],
                        ytrain=ytrain,x_test=x_test[selected.variable],ytest=ytest,kpi_train=kpi_train,kpi_weighted=my_scorer)
    
    return classifier, selected

def calcular_lda (x_train, ytrain, num_iters, num_folds_rfe,num_folds_rs,randomState,kpi_train,x_test,ytest,sample_weights=pd.Series([])):
    
    #Definir modelo de clasificación y eliminación de variables
    model = LinearDiscriminantAnalysis()
    model_rfe = RFECV(estimator = model,step=4,min_features_to_select=6,cv=num_folds_rfe,scoring=kpi_train,n_jobs=1)

    #regularization
    solver = ['lsqr','eigen','svd']
    shrinkage = uniform(0.01, 0.3)
    # Maximum number of literations
    # Create the random grid
    parameters = {'estimator__solver': solver,
                'estimator__shrinkage': shrinkage
                }

    if sample_weights.empty:
        my_scorer = kpi_train
    else:
        my_scorer = custom_scorer(sample_weights=sample_weights)
        
    classifier = RandomizedSearchCV(model_rfe,parameters, n_iter = num_iters ,scoring = my_scorer,n_jobs=-1, cv=num_folds_rs,random_state=randomState)
    classifier.fit(x_train,ytrain)
    
    selected = pd.DataFrame({"variable":x_train.columns[classifier.best_estimator_.support_==True],
    "Seleccionada":classifier.best_estimator_.support_[classifier.best_estimator_.support_==True],
    "importancia":classifier.best_estimator_.estimator_.coef_[0]})
    
    estimador = classifier.best_estimator_.estimator_
    
    imprimir_resultados(nombre_modelo= "LDA",classifier=classifier,estimador=estimador,x_train=x_train[selected.variable],
                        ytrain=ytrain,x_test=x_test[selected.variable],ytest=ytest,kpi_train=kpi_train,kpi_weighted=my_scorer)
    
    return classifier, selected

def calcular_qda (x_train,ytrain, lista_variables, num_iters, num_folds,randomState,kpi_train,x_test,ytest,sample_weights=pd.Series([])):
    
    #Mejores parámetros por selección de variables
    mejores_variables = None
    mejor_score = 0
    mejor_estimador = None
    ###########################################
    for lista in lista_variables:
        #Modelo 
        model = QuadraticDiscriminantAnalysis()
        #regularization
        reg_param = uniform(0.01, 0.3)
        # Create the random grid
        parameters = {'reg_param': reg_param
                    }
        
        if sample_weights.empty:
            my_scorer = kpi_train
        else:
            my_scorer = custom_scorer(sample_weights=sample_weights)
        
        classifier = RandomizedSearchCV(model,parameters, n_iter = num_iters ,scoring = my_scorer,n_jobs=-1, cv=num_folds,random_state=randomState)
        #Estimar por selección de variables
        classifier.fit(x_train[lista],ytrain)

        estimador = classifier.best_estimator_
        #Evaluación del modelo por selección de variables
        score_test = evaluate_metrics(estimator = estimador,X=x_test[lista],y=ytest,kpi=kpi_train) * 0.5 +classifier.best_score_*0.5
        if score_test >mejor_score:
            mejor_score = score_test
            mejores_variables = lista
            mejor_estimador = classifier
        #################################################
    estimador = mejor_estimador.best_estimator_
    imprimir_resultados(nombre_modelo= "QDA",classifier=mejor_estimador,estimador=estimador,x_train=x_train[mejores_variables],
                        ytrain=ytrain,x_test=x_test[mejores_variables],ytest=ytest,kpi_train=kpi_train,kpi_weighted=my_scorer)
    
    
    return mejor_estimador, pd.Series(mejores_variables)

def calcular_KNN (x_train,ytrain, lista_variables, num_iters, num_folds,randomState,kpi_train,x_test,ytest,sample_weights=pd.Series([])):
    #Mejores parámetros por selección de variables
    mejores_variables = None
    mejor_score = 0
    mejor_estimador = None
    ###########################################
    for lista in lista_variables:
        model = KNeighborsClassifier(n_jobs=1)
        
        n_neighbors = randint(1,22)
        weights = ['uniform','distance']
        algorithm = ["kd_tree","ball_tree","brute"]
        leaf_size = randint(1,80)
        
        # Create the random grid
        parameters = {'n_neighbors': n_neighbors,
                    'weights': weights,
                    'algorithm': algorithm,
                    'leaf_size': leaf_size}
        
        if sample_weights.empty:
            my_scorer = kpi_train
        else:
            my_scorer = custom_scorer(sample_weights=sample_weights)

        classifier = RandomizedSearchCV(model,parameters, n_iter = num_iters ,scoring = my_scorer,n_jobs=-1, cv=num_folds,random_state=randomState)
        #Estimar por selección de variables
        classifier.fit(x_train[lista],ytrain)
        
        estimador = classifier.best_estimator_
        
        #Evaluación del modelo por selección de variables
        score_test = evaluate_metrics(estimator = estimador,X=x_test[lista],y=ytest,kpi=kpi_train)* 0.5 +classifier.best_score_*0.5
        if score_test >mejor_score:
            mejor_score = score_test
            mejores_variables = lista
            mejor_estimador = classifier
        #################################################
    estimador = mejor_estimador.best_estimator_
    imprimir_resultados(nombre_modelo= "KNN",classifier=mejor_estimador,estimador=estimador,x_train=x_train[mejores_variables],
                        ytrain=ytrain,x_test=x_test[mejores_variables],ytest=ytest,kpi_train=kpi_train,kpi_weighted=my_scorer)
    
    return mejor_estimador, pd.Series(mejores_variables)

def calcular_MLP (x_train,ytrain,lista_variables, num_iters, num_folds,randomState,kpi_train,x_test,ytest,sample_weights=pd.Series([])):
    #Mejores parámetros por selección de variables
    mejores_variables = None
    mejor_score = 0
    mejor_estimador = None
    ###########################################
    for lista in lista_variables:
        #Definir modelo con parámetros fijos
        model = MLPClassifier(random_state=randomState,validation_fraction=0.15,n_iter_no_change = 30)
        #definir parámetros variables
        
        #Pendiente hidden layers
        activation = ['identity','logistic','tanh','relu'] 
        solver = ['lbfgs','sgd','adam']
        alpha = uniform(0.01, 0.99)
        learning_rate = ['constant','invscaling','adaptive']
        learning_rate_init = uniform(0.01, 0.4)
        max_iter = randint(50,600)
        shuffle = [True,False]
        momentum = uniform(0.01, 0.99)
        early_stopping = [True,False]
        beta_1= uniform(0.01, 0.99)
        beta_2= uniform(0.01, 0.99)
        

        # Create the random grid
        parameters = {'activation': activation,
                    'solver': solver,
                    'alpha': alpha,
                    'learning_rate': learning_rate,
                    'learning_rate_init':learning_rate_init,
                    'max_iter':max_iter,
                    'shuffle':shuffle,
                    'momentum':momentum,
                    'early_stopping':early_stopping,
                    'beta_1':beta_1,
                    'beta_2':beta_2}
        
        if sample_weights.empty:
            my_scorer = kpi_train
        else:
            my_scorer = custom_scorer(sample_weights=sample_weights)
        
        classifier = RandomizedSearchCV(model,parameters, n_iter = num_iters ,scoring = my_scorer,n_jobs=-1, cv=num_folds,random_state=randomState)
        #Estimar por selección de variables
        classifier.fit(x_train[lista],ytrain)
        
        estimador = classifier.best_estimator_
        
        #Evaluación del modelo por selección de variables
        score_test = evaluate_metrics(estimator = estimador,X=x_test[lista],y=ytest,kpi=kpi_train)* 0.5 +classifier.best_score_*0.5
        if score_test >mejor_score:
            mejor_score = score_test
            mejores_variables = lista
            mejor_estimador = classifier
        #################################################
    estimador = mejor_estimador.best_estimator_
    imprimir_resultados(nombre_modelo= "MLP",classifier=mejor_estimador,estimador=estimador,x_train=x_train[mejores_variables],
                        ytrain=ytrain,x_test=x_test[mejores_variables],ytest=ytest,kpi_train=kpi_train,kpi_weighted=my_scorer)
    
    return mejor_estimador, pd.Series(mejores_variables)


# from sklearn.ensemble import VotingClassifier
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from scipy.stats import uniform, truncnorm, randint
# #models
# model_logit = LogisticRegression(random_state=0,class_weight='balanced',solver='saga',n_jobs=1)
# model_rf = RandomForestClassifier(n_jobs=1,random_state=0,class_weight="balanced")
# ensemble_clf = VotingClassifier(estimators=[('lr', model_logit), ('rf', model_rf)],voting='soft')

# #parameters logit
# penalty = ['l1','elasticnet']
# C = uniform(0.01, 1.99)
# max_iter=[20,50,100,200,250,500]
# l1_ratio = [0.2,0.4,0.6,0.8]

# #parameters rf
# n_estimators = randint(10,1000)
# criterion = ["gini","entropy"]
# max_depth = [int(x) for x in np.linspace(5, 40, num = 10)]
# min_samples_split = [2, 5, 7,10,12]
# min_samples_leaf = [1, 2, 4,7]
# max_features = ["sqrt","log2"]
# bootstrap = [True, False]

# # Create the random grid
# parameters_rf = {'rf__n_estimators': n_estimators,
#             'rf__criterion':criterion,
#             'rf__max_features': max_features,
#             'rf__max_depth': max_depth,
#             'rf__min_samples_split': min_samples_split,
#             'rf__min_samples_leaf': min_samples_leaf,
#             'rf__bootstrap': bootstrap}

# # Create the random grid
# parameters_logit = {'lr__penalty': penalty,
# 'lr__C': C,
# 'lr__max_iter': max_iter,
# 'lr__l1_ratio': l1_ratio
# }

# parameters_ensemble = {**parameters_logit, **parameters_rf}

# classifier = RandomizedSearchCV(ensemble_clf,parameters_ensemble, n_iter = 100 ,scoring = f05score,n_jobs=-1, 
# cv=10,random_state=0)

# classifier.fit(X_train,y_alto_train)


# print("AUC train")
# print(classifier.best_score_)


# df_importancia_variables = pd.DataFrame(rf_u2.feature_importances_,X_train_u1.columns,columns=["valor"])
# df_importancia_variables.sort_values("valor",ascending=False,inplace=True)
# df_importancia_variables= df_importancia_variables[df_importancia_variables["valor"]>0.02]
# df_importancia_variables=df_importancia_variables.reset_index(level=0)

# import matplotlib.pyplot as plt
# import seaborn as sns
# %matplotlib inline
# # Creating a bar plot
# sns.barplot(x=df_importancia_variables["valor"],y=df_importancia_variables["index"])
# # Add labels to your graph
# plt.xlabel('Importancia')
# plt.ylabel('Variables')
# plt.title("Importancia de variables")
# plt.show()





