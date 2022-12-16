## opensourceSW_Final
------------
+ Configuration instructions
  + I use SVM(support vector machine)(final)
``` Python
SVC(C=25,kernel='rbf',gamma=0.8,random_state=0,probability=True)
```
  +I used knn
  ```
  KNeighborsClassifier(n_neighbors=1)
  ```
  +I used histgradientboosting
  ```
  histgradintboosting(random_state=0)
  ```
  
  +I used randomforest
  ```
  RandomForestClassifier()
  ```
  +I used optuna for hyperparameter tuning
  ```
  pip install optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler

from sklearn.metrics import mean_absolute_error

def objective(trial: Trial, X, y, test,y_test2):
    param = {
        'C' : trial.suggest_float('', 1.0, 200.0),
        'max_depth' : trial.suggest_int('max_depth', 8, 16),
        'min_child_weight' : trial.suggest_int('min_child_weight', 1, 300),
        'gamma' : trial.suggest_int('gamma', 1, 3),
        'learning_rate' : 0.01,
        'colsample_bytree' : trial.suggest_discrete_uniform('colsample_bytree', 0.5, 1, 0.1),
        'nthread' : -1,
        'subsample' : trial.suggest_categorical('subsample', [0.6,0.7,0.8,1.0]),
        'random_state' : 1127
    }
    model = sklearn.svm.SVC(**param)
    xgb_model = model.fit(X, y, verbose=True) 
    

    score = mean_absolute_error(xgb_model.predict(X), y)
    
    return score


study = optuna.create_study(direction='minimize', sampler=TPESampler())

study.optimize(lambda trial : objective(trial, X_train, y_train, X_test2,y_test2), n_trials = 50)

print('Best trial : score {}, \nparams {}'.format(study.best_trial.value, study.best_trial.params))
```
  
  +I used voting classifier
```
VotingClassifier(estimators=[
        ('lr', clf1), ('rf', clf2), ('gnb', clf3),('df',clf4)], voting='hard') 
```
  +I changed test size(0.3->0.001)
  ```
  X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.001, random_state=0)
  ```

+ Operating instructions
  + download `tumor_datset` and open in `jupyter notebook`
+ Copyright and licensing information
  + `MIT licence` 
+ Contact information for the distributor or author
  + `https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC`

#### My Info
> 20220742_정의연
>> antt6942@gmail.com
>>> chung_ung_univ AI Department
