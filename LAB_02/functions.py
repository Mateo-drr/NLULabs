# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

from sklearn import svm
from sklearn.metrics import classification_report

def fitsvm(max_iter, data, labels,test, ng_test):
    model = svm.LinearSVC(max_iter=max_iter)
    model.fit(data,labels)

    #Predict labels from test samples
    hyps = model.predict(test)
    refs = ng_test.target
    cross_valid(model, test, refs)
    #Evaluate the model
    report = classification_report(refs, hyps, target_names=ng_test.target_names)
    print('Classification Report: \n',report)
    return report[-200:-1]

def cross_valid(clf, test, target): #clf is the model, test are the testing samples
    import math
    from sklearn.model_selection import cross_validate
    import warnings
    warnings.filterwarnings("ignore")

    #some are not for this so they give nan, hence the if below
    print('Evaluation Scores:')
    scores = ['accuracy', 'balanced_accuracy', 'top_k_accuracy', 'average_precision', 'neg_brier_score', 'f1', 'f1_micro', 'f1_macro', 'f1_weighted', 'f1_samples', 'neg_log_loss', 'precision', 'recall', 'jaccard', 'roc_auc', 'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted']
    for scr in scores: #Calculate each metric and print it
        scores = cross_validate(clf, test, target, cv=5, scoring=[scr])
        if not math.isnan(sum(scores['test_'+scr])/len(scores['test_'+scr])):
            print('\t',scr , sum(scores['test_'+scr])/len(scores['test_'+scr]))