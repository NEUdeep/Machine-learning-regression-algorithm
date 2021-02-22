from sklearn.model_selection import GridSearchCV,LeaveOneGroupOut,ParameterGrid


def _GridSearchCV(params_dict,model):
    '''
    :param params_dict:
    :param model:
    :return:
    '''
    gs = GridSearchCV(
        estimator=model,
        param_grid=params_dict,
        n_jobs=2,
        scoring='r2',
        cv=6
    )
    return gs