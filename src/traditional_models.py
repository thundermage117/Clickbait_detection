def flatten_and_concatenate(head_vec,body_vec):
    """
    Flattens a list of head and body tensors and concatenates them along the first dimension
    :param tensors: list of tensors
    :return: flattened and concatenated tensor
    """
    import torch
    n_samples,max_word,word2vecLength=head_vec.shape
    head_vec_copy=torch.reshape(head_vec,(n_samples,max_word*word2vecLength))
    body_vec_copy=torch.reshape(body_vec,(n_samples,max_word*word2vecLength))
    merged_vec=torch.cat((head_vec_copy,body_vec_copy),1)
    return merged_vec

def SVM_accuracy(merged_vec,labels_val):
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import train_test_split
    from warnings import simplefilter
    from sklearn.exceptions import ConvergenceWarning
    simplefilter("ignore", category=ConvergenceWarning)
    X_train, X_test, y_train, y_test = train_test_split(merged_vec, labels_val, test_size=0.2, random_state=42)
    clf = LinearSVC(random_state=0, tol=1e-6)
    clf.fit(X_train, y_train)
    return clf.score(X_test,y_test)
  
def LogRegr_accuracy(merged_vec,labels_val):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from warnings import simplefilter
    from sklearn.exceptions import ConvergenceWarning
    simplefilter("ignore", category=ConvergenceWarning)
    X_train, X_test, y_train, y_test = train_test_split(merged_vec, labels_val, test_size=0.2, random_state=42)
    clf = LogisticRegression(random_state=0, tol=1e-6)
    clf.fit(X_train, y_train)
    return clf.score(X_test,y_test)

def RandFrst_accuracy(merged_vec_new,labels_val_copy):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    X_train, X_test, y_train, y_test = train_test_split(merged_vec_new, labels_val_copy, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf.score(X_test,y_test)

