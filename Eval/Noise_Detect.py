import os, glob
import numpy as np
from sklearn import svm
from sklearn.metrics import f1_score, accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import pickle

def main():

    data_files = glob.glob(os.path.normpath('./Dataset/*Train_rastaFeat.npy'))
    
    trained_file = os.path.normpath('./Checkpoints/RastaSVM_Norm.sav')

    classifier = 'SVM' # SVM or GMM

    file = data_files[0]
    label_file = file[:-8] + 'Label.npy'
    features = np.load(file, allow_pickle = True)
    label_new = np.load(label_file, allow_pickle = True).reshape((-1,1))

    n_samples = features.shape[0]

    n_train = int(n_samples * 0.8)

    indices_skip = np.arange(0, n_samples, 10)
    features = np.take(features, indices_skip, axis = 0)
    label_new = np.take(label_new, indices_skip, axis = 0)

    skip_samples = features.shape[0]
    num_features = features.shape[1]
    n_train = int(skip_samples * 0.8)

    data_train = features[:n_train,:]
    labels_train = label_new[:n_train,:]

    data_val = features[n_train:,:]
    labels_val = label_new[n_train:,:]

    data_10db = np.zeros((1,num_features))
    data_5db = np.zeros((1,num_features))
    data_0db = np.zeros((1,num_features))

    label_10db = np.zeros((1,1))
    label_5db = np.zeros((1,1))
    label_0db = np.zeros((1,1))

    if '10_dB' in file:
        data_10db = np.concatenate((data_10db,data_val), axis = 0)
        label_10db = np.concatenate((label_10db,labels_val), axis = 0)
    elif '5_dB' in file:
        data_5db = np.concatenate((data_5db,data_val), axis = 0)
        label_5db = np.concatenate((label_5db,labels_val), axis = 0)
    elif '0_dB' in file:
        data_0db = np.concatenate((data_0db,data_val), axis = 0)
        label_0db = np.concatenate((label_0db,labels_val), axis = 0)
    else:
        print('Incorrect SNR value')
        exit()

    for file in data_files[1:]:
        features = np.load(file, allow_pickle = True)
        label_file = file[:-8] + 'Label.npy'
        label_new = np.load(label_file, allow_pickle = True).reshape((-1,1))

        n_samples = features.shape[0]

        n_train = int(n_samples * 0.8)

        indices_skip = np.arange(0, n_samples, 10)
        features = np.take(features, indices_skip, axis = 0)
        label_new = np.take(label_new, indices_skip, axis = 0)

        skip_samples = features.shape[0]
        n_train = int(skip_samples * 0.8)

        data_train = np.concatenate((data_train,features[:n_train,:]),axis = 0)
        labels_train = np.concatenate((labels_train,label_new[:n_train,:]),axis = 0)
        #data_val = np.concatenate((data_val,features[n_train:,:]),axis = 0)
        #labels_val = np.concatenate((labels_val,label_new[n_train:,:]),axis = 0)

        if '10_dB' in file:
            data_10db = np.concatenate((data_10db,features[n_train:,:]), axis = 0)
            label_10db = np.concatenate((label_10db,label_new[n_train:,:]), axis = 0)
        elif '5_dB' in file:
            data_5db = np.concatenate((data_5db,features[n_train:,:]), axis = 0)
            label_5db = np.concatenate((label_5db,label_new[n_train:,:]), axis = 0)
        elif '0_dB' in file:
            data_0db = np.concatenate((data_0db,features[n_train:,:]), axis = 0)
            label_0db = np.concatenate((label_0db,label_new[n_train:,:]), axis = 0)
        else:
            print('Incorrect SNR value')
            exit()

    data_10db = data_10db[1:,:]
    label_10db = label_10db[1:,:]
    data_5db = data_5db[1:,:]
    label_5db = label_5db[1:,:]
    data_0db = data_0db[1:,:]
    label_0db = label_0db[1:,:]

    # Normalize data

    scaler = StandardScaler()

    scaler.fit(data_train)

    data_train = scaler.transform(data_train)
    data_10db = scaler.transform(data_10db)
    data_5db = scaler.transform(data_5db)
    data_0db = scaler.transform(data_0db)

    if os.path.exists(trained_file):
        print('Loading pre-trained model')
        clf = pickle.load(open(trained_file, 'rb'))
    else:
        
        print('Training classifier')

        if classifier == 'SVM':

            clf = svm.SVC(class_weight='balanced') 

            clf.fit(data_train,labels_train)

            pickle.dump(clf, open(trained_file, 'wb'))

        elif classifier == 'GMM':

            clf = GaussianMixture(n_components=2, covariance_type='diag', max_iter = 1000)

            clf.fit(data_train,labels_train)

            pickle.dump(clf, open(trained_file, 'wb'))

        else:
            print('Classifier not supported')
            exit()

        


    # Predict validation data for different snr

    y_predict_10db = clf.predict(data_10db)
    y_predict_5db = clf.predict(data_5db)
    y_predict_0db = clf.predict(data_0db)

    true_conc = np.concatenate((label_10db, label_5db, label_0db), axis = 0)
    predict_conc = np.concatenate((y_predict_10db, y_predict_5db, y_predict_0db), axis = 0)

    f1score_10db = f1_score(label_10db, y_predict_10db)
    acc_10db = accuracy_score(label_10db, y_predict_10db)
    f1score_5db = f1_score(label_5db, y_predict_5db)
    acc_5db = accuracy_score(label_5db, y_predict_5db)
    f1score_0db = f1_score(label_0db, y_predict_0db)
    acc_0db = accuracy_score(label_0db, y_predict_0db)

    f1score = f1_score(true_conc, predict_conc)
    acc = accuracy_score(true_conc, predict_conc)

    print(f1score_10db)
    print(acc_10db)
    print(f1score_5db)
    print(acc_5db)
    print(f1score_0db)
    print(acc_0db)

    print('Total: ')
    print('F1 Score ' + str(f1score))
    print('Accuracy ' + str(acc))

    # Analyse test data
    print('Begin Test analysis')
    test_files = glob.glob(os.path.normpath('./Dataset/*Test_rastaFeat.npy'))

    file = test_files[0]
    label_file = file[:-8] + 'Label.npy'
    data_test = np.load(file, allow_pickle = True)
    label_test = np.load(label_file, allow_pickle = True).reshape((-1,1))

    test_10db = np.zeros((1,num_features))
    test_lab_10db = np.zeros((1,1))
    test_5db = np.zeros((1,num_features))
    test_lab_5db = np.zeros((1,1))
    test_0db = np.zeros((1,num_features))
    test_lab_0db = np.zeros((1,1))

    if '10_dB' in file:
        test_10db = np.concatenate((test_10db,data_test), axis = 0)
        test_lab_10db = np.concatenate((test_lab_10db,label_test), axis = 0)
    elif '5_dB' in file:
        test_5db = np.concatenate((test_5db,data_test), axis = 0)
        test_lab_5db = np.concatenate((test_lab_5db,label_test), axis = 0)
    elif '0_dB' in file:
        test_0db = np.concatenate((test_0db,data_test), axis = 0)
        test_lab_0db = np.concatenate((test_lab_0db,label_test), axis = 0)
    else:
        print('Incorrect SNR value')
        exit()

    test_segments_len = []

    test_segments_len.append(data_test.shape[0])

    for file in test_files[1:]:
        features = np.load(file, allow_pickle = True)
        label_file = file[:-8] + 'Label.npy'
        label_new = np.load(label_file, allow_pickle = True).reshape((-1,1))

        test_segments_len.append(features.shape[0])

        data_test = np.concatenate((data_test,features),axis = 0)
        label_test = np.concatenate((label_test,label_new),axis = 0)

        if '10_dB' in file:
            test_10db = np.concatenate((test_10db,features), axis = 0)
            test_lab_10db = np.concatenate((test_lab_10db,label_new), axis = 0)
        elif '5_dB' in file:
            test_5db = np.concatenate((test_5db,features), axis = 0)
            test_lab_5db = np.concatenate((test_lab_5db,label_new), axis = 0)
        elif '0_dB' in file:
            test_0db = np.concatenate((test_0db,features), axis = 0)
            test_lab_0db = np.concatenate((test_lab_0db,label_new), axis = 0)
        else:
            print('Incorrect SNR value')
            exit()

    test_10db = test_10db[1:,:]
    test_lab_10db = test_lab_10db[1:,:]
    test_5db = test_5db[1:,:]
    test_lab_5db = test_lab_5db[1:,:]
    test_0db = test_0db[1:,:]
    test_lab_0db = test_lab_0db[1:,:]

    data_test = scaler.transform(data_test)
    test_10db = scaler.transform(test_10db)
    test_5db = scaler.transform(test_5db)
    test_0db = scaler.transform(test_0db)

    y_test_10db = clf.predict(test_10db)
    y_test_5db = clf.predict(test_5db)
    y_test_0db = clf.predict(test_0db)

    f1score_10db = f1_score(test_lab_10db, y_test_10db)
    acc_10db = accuracy_score(test_lab_10db, y_test_10db)
    f1score_5db = f1_score(test_lab_5db, y_test_5db)
    acc_5db = accuracy_score(test_lab_5db, y_test_5db)
    f1score_0db = f1_score(test_lab_0db, y_test_0db)
    acc_0db = accuracy_score(test_lab_0db, y_test_0db)

    print('Test:')
    print(f1score_10db)
    print(acc_10db)
    print(f1score_5db)
    print(acc_5db)
    print(f1score_0db)
    print(acc_0db)

    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(label_10db)
    plt.plot(y_predict_10db, 'r')
    plt.subplot(3,1,2)
    plt.plot(label_5db)
    plt.plot(y_predict_5db, 'r')
    plt.subplot(3,1,3)
    plt.plot(label_0db)
    plt.plot(y_predict_0db, 'r')
    plt.show()

if __name__ == "__main__":
    main()