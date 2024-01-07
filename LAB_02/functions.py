# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

from sklearn import svm
from sklearn.metrics import classification_report

def fitsvm(max_iter, data, labels):
    """
    Fit SVM model with linear kernel using scikit-learn's LinearSVC.

    Parameters:
    max_iter (int): Maximum number of iterations for model training.
    data (array-like): Training data features.
    labels (array-like): Training data labels.

    Returns:
    model: Trained SVM model.
    """
    model = svm.LinearSVC(max_iter=max_iter)
    model.fit(data,labels)
    return model

def testsvm(model,test, ng_test):
    """
    Tests a trained SVM model on the test set and prints the classification report.

    Parameters:
    model: Trained SVM model.
    test (array-like): Test data features.
    ng_test: Test data labels and other information from the 20 newsgroups dataset.

    Returns:
    str: The last 200 characters of the classification report.
    """
    #Predict labels from test samples
    hyps = model.predict(test)
    refs = ng_test.target
    cross_valid(model, test, refs)
    #Evaluate the model
    report = classification_report(refs, hyps, target_names=ng_test.target_names)
    print('Classification Report: \n',report)
    return report[-200:-1]

def cross_valid(clf, test, target): 
    """
    Performs cross-validation with a given classifier and prints evaluation scores.

    Parameters:
    clf: Classifier model.
    test (array-like): Test data features.
    target (array-like): Target labels.

    Prints:
    Evaluation scores for various metrics.
    """
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
            
def vectorizerGetData(vectorizer,sample_data,ng_test):
    """
    Fits a vectorizer on the sample data and transforms the test data.

    Parameters:
    vectorizer: Text vectorizer (e.g., CountVectorizer or TfidfVectorizer).
    sample_data (array-like): Sample data used for fitting the vectorizer.
    ng_test: Test data information from the 20 newsgroups dataset.

    Returns:
    train: Transformed training data.
    test: Transformed test data.
    """
    train = vectorizer.fit_transform(sample_data)
    test = vectorizer.transform(ng_test.data)
    return train,test

def printRes(comp_res,variations):
    """
    Prints the detailed and simplified results of runs.
    
    Parameters:
    comp_res (list): List of classification report strings for each variation.
    variations (list): List of model variation names.
    
    Prints:
    Avg results including precision, recall, and f1-score for each variation.
    Just accuracy for each variation.
    """
    simpres=[]
    print('\nResults comparison:\n')
    for i,result in enumerate(comp_res):
        print(variations[i]+':')
        print('\t\t\t\t\t\t  precision    recall  f1-score',result,'\n')
    
        res = result.split(' ')
        simpres.append([item for item in res if item != ""][2])
        
    print('-------------------------------------------------------------------')
    print('Comparison simplified\n')
    for i in range(0,len(variations)):
        print(variations[i]+':','acc:',simpres[i])