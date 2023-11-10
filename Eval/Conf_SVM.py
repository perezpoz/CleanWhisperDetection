import os, glob
import numpy as np
from sklearn import svm
from sklearn.metrics import f1_score, accuracy_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import pickle

def main():

    """
    Calculate and plot confusion matrix using the trained SVM model.
    """

    # Select augmented or non-augmented
    #data_files = glob.glob(os.path.normpath('./Dataset/*Train_rastaFeat.npy'))
    data_files = glob.glob(os.path.normpath('./Data_aug/*Train_rastaFeat.npy'))
    
    trained_file = os.path.normpath('./Checkpoints/RastaSVM_Norm.sav')

    # Read data
    _, _, data_val, labels_val, _, _ = read_data_files(data_files)

    labels_val = labels_val.reshape((-1,))

    svm_model = pickle.load(open(trained_file, 'rb'))

    print('Scaling data')
    scaler = StandardScaler()
    #norm_train = scaler.fit_transform(data_train)

    norm_val = scaler.transform(data_val)

    print('Training CWAD SVM')
    #svm_model.fit(norm_train, labels_train.reshape((-1,)))
    print('Predict validation samples')
    pred_val = svm_model.predict(norm_val)

    pred_val = pred_val.reshape((-1,)).astype(np.int16)

    cf_mat = confusion_matrix(labels_val, pred_val, normalize = 'all').astype(np.float32)
    disp = ConfusionMatrixDisplay(cf_mat)
    disp.plot()
    plt.show()


def read_data_files(data_files):
    file = data_files[0]
    label_file = file[:-8] + 'CleanLabel.npy'
    features = np.load(file, allow_pickle = True)
    label_new = np.load(label_file, allow_pickle = True).reshape((-1,1))

    n_samples = features.shape[0]

    n_train = int(n_samples * 0.8)

    skip_samples = features.shape[0]
    #num_features = features.shape[1]
    n_train = int(skip_samples * 0.8)

    data_train = features[:n_train,:]
    labels_train = label_new[:n_train,:]

    data_val = features[n_train:,:]
    labels_val = label_new[n_train:,:]

    train_segments_len = []
    val_segments_len = []
    
    train_segments_len.append(data_train.shape[0])
    val_segments_len.append(data_val.shape[0])

    for file in data_files[1:]:
        features = np.load(file, allow_pickle = True)
        label_file = file[:-8] + 'Label.npy'
        label_new = np.load(label_file, allow_pickle = True).reshape((-1,1))

        skip_samples = features.shape[0]
        n_train = int(skip_samples * 0.8)

        data_train_new = features[:n_train,:]
        data_val_new = features[n_train:,:]

        train_segments_len.append(data_train_new.shape[0])
        val_segments_len.append(data_val_new.shape[0])

        data_train = np.concatenate((data_train,data_train_new),axis = 0)
        labels_train = np.concatenate((labels_train,label_new[:n_train,:]),axis = 0)
        data_val = np.concatenate((data_val,data_val_new),axis = 0)
        labels_val = np.concatenate((labels_val,label_new[n_train:,:]),axis = 0)

    return data_train, labels_train, data_val, labels_val, train_segments_len, val_segments_len

if __name__ == "__main__":
    main()