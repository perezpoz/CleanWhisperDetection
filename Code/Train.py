import os, glob
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_curve
from sklearn.preprocessing import StandardScaler
import pickle

import torch
from torch.utils.data import DataLoader

from CWAD_models import LSTM_WAD, Whisper_sequence, early_stopper

import matplotlib.pyplot as plt

from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

def main():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    writer = SummaryWriter()
    
    data_files = glob.glob(os.path.normpath('./Dataset/*Train_rastaFeat.npy'))

    file = data_files[0]
    label_file = file[:-8] + 'Label.npy'
    features = np.load(file, allow_pickle = True)
    label_new = np.load(label_file, allow_pickle = True).reshape((-1,1))

    n_samples = features.shape[0]

    n_train = int(n_samples * 0.8)

    skip_samples = features.shape[0]
    num_features = features.shape[1]
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

        n_samples = features.shape[0]

        n_train = int(n_samples * 0.8)

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

    # Normalize data

    scaler_path = os.path.normpath('./Models/LSTM_Scaler.sav')
    scaler = StandardScaler()

    scaler.fit(data_train)

    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    data_train = scaler.transform(data_train)
    data_val = scaler.transform(data_val)

    epochs = 200
    sequence_length = 30

    load_pretrained = False

    model = LSTM_WAD(input_size = num_features, layers_size = 64, output_size = 1, num_lstm_layers = 2, sequence_length = sequence_length, bidirectional = False)
    model.to(device)
    model = torch.compile(model)

    
    dataset = Whisper_sequence(data = data_train, labels = labels_train, sequence_length = sequence_length, segment_lengths = train_segments_len)
    dl = DataLoader(dataset, batch_size = 256, shuffle = True)

    dataset_val = Whisper_sequence(data = data_val, labels = labels_val, sequence_length = sequence_length, segment_lengths = val_segments_len)
    dl_val = DataLoader(dataset_val, batch_size = 256, shuffle = True)

    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum = 0.9)

    stopper = early_stopper(max_iter_higher_loss=10)

    model_path = os.path.normpath('./Models/LSTM_128_LongDS_3020_mps.pt')

    if (load_pretrained and os.path.exists(model_path)):
        print('Loading pre-trained model.')
        model.load_state_dict(torch.load(model_path))
    else:
        print('Training model using ' + str(device))

        for epoch in range(epochs):

            running_loss = 0
            running_loss_val = 0

            model.train()
            for x,y in tqdm(dl,total = len(dl)):

                optimizer.zero_grad()
                y_hat = model(x.type(torch.FloatTensor).to(device))
                loss = loss_fn(y_hat,y.type(torch.FloatTensor).to(device))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            running_loss /= data_train.shape[0]
            print('Train loss in epoch ' + str(epoch) + ': ' + str(running_loss))

            model.eval()
            for x,y in dl_val:
                y_hat = model(x.type(torch.FloatTensor).to(device))
                loss = loss_fn(y_hat, y.type(torch.FloatTensor).to(device))
                running_loss_val += loss.item()
            
            running_loss_val /= data_val.shape[0]
            print('Validation loss in epoch ' + str(epoch) + ': ' + str(running_loss_val))

            writer.add_scalar('Loss/train', running_loss, epoch)
            writer.add_scalar('Loss/val', running_loss_val, epoch)

            if stopper.update_checkpoint(running_loss_val):
                torch.save(model.state_dict(), model_path)
            
            if stopper.stop_training():
                print('Early stopping in epoch ' + str(epoch))
                break

        model_path = os.path.normpath('./Models/BiLSTM_32_LongDS.pt')
        torch.save(model.state_dict(), model_path)

    dl_th = DataLoader(dataset, batch_size = 256, shuffle = False)


    test_dl = DataLoader(dataset_val, batch_size = 128, shuffle = False)

    final_y_train = np.zeros((len(dataset),1))
    true_train_labels = np.zeros((len(dataset),1))
    tr_idx = 0
    model.eval()
    for x,y in dl_th:
        final_y_train[tr_idx:tr_idx + x.size(0)] = model(x.type(torch.FloatTensor).to(device)).cpu().detach().numpy()
        true_train_labels[tr_idx:tr_idx + x.size(0)] = y.detach().numpy()
        tr_idx += x.size(0)

    fpr_val, tpr_val,th_val = roc_curve(true_train_labels, final_y_train)
    opt_th = roc_optimum_th(tpr_val, fpr_val, th_val)

    print('Optimum threshold: ' + str(opt_th))

    final_y_val = np.zeros((len(dataset_val),1))
    true_test_labels = np.zeros((len(dataset_val),1))
    val_idx = 0
    model.eval()
    for x,y in test_dl:
        final_y_val[val_idx:val_idx + x.size(0)] = model(x.type(torch.FloatTensor).to(device)).cpu().detach().numpy()
        true_test_labels[val_idx:val_idx + x.size(0)] = y.detach().numpy()
        val_idx += x.size(0)

    final_val = [1 if y_val > opt_th else 0 for y_val in final_y_val]

    f1_val = f1_score(true_test_labels, final_val)
    acc_val = accuracy_score(true_test_labels, final_val)
    
    print(f1_val)
    print(acc_val)

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

    test_segments_len = []
    test_segments_10db = []
    test_segments_5db = []
    test_segments_0db = []

    if '10_dB' in file:
        test_segments_10db.append(data_test.shape[0])
        test_10db = np.concatenate((test_10db,data_test), axis = 0)
        test_lab_10db = np.concatenate((test_lab_10db,label_test), axis = 0)
    elif '5_dB' in file:
        test_segments_5db.append(data_test.shape[0])
        test_5db = np.concatenate((test_5db,data_test), axis = 0)
        test_lab_5db = np.concatenate((test_lab_5db,label_test), axis = 0)
    elif '0_dB' in file:
        test_segments_0db.append(data_test.shape[0])
        test_0db = np.concatenate((test_0db,data_test), axis = 0)
        test_lab_0db = np.concatenate((test_lab_0db,label_test), axis = 0)
    else:
        print('Incorrect SNR value')
        exit()

    test_segments_len.append(data_test.shape[0])

    for file in test_files[1:]:
        features = np.load(file, allow_pickle = True)
        label_file = file[:-8] + 'Label.npy'
        label_new = np.load(label_file, allow_pickle = True).reshape((-1,1))

        test_segments_len.append(features.shape[0])

        data_test = np.concatenate((data_test,features),axis = 0)
        label_test = np.concatenate((label_test,label_new),axis = 0)

        if '10_dB' in file:
            test_segments_10db.append(features.shape[0])
            test_10db = np.concatenate((test_10db,features), axis = 0)
            test_lab_10db = np.concatenate((test_lab_10db,label_new), axis = 0)
        elif '5_dB' in file:
            test_segments_5db.append(features.shape[0])
            test_5db = np.concatenate((test_5db,features), axis = 0)
            test_lab_5db = np.concatenate((test_lab_5db,label_new), axis = 0)
        elif '0_dB' in file:
            test_segments_0db.append(features.shape[0])
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
    
    test_dataset = Whisper_sequence(data_test, label_test, sequence_length = sequence_length, segment_lengths = test_segments_len)
    test_dl = DataLoader(test_dataset, batch_size = 128, shuffle = False)
    dataset_10db = Whisper_sequence(test_10db, test_lab_10db, sequence_length = sequence_length, segment_lengths = test_segments_10db)
    test_dl_10db = DataLoader(dataset_10db, batch_size = 128, shuffle = False)
    dataset_5db = Whisper_sequence(test_5db, test_lab_5db, sequence_length = sequence_length, segment_lengths = test_segments_5db)
    test_dl_5db = DataLoader(dataset_5db, batch_size = 128, shuffle = False)
    dataset_0db = Whisper_sequence(test_0db, test_lab_0db, sequence_length = sequence_length, segment_lengths = test_segments_0db)
    test_dl_0db = DataLoader(dataset_0db, batch_size = 128, shuffle = False)

    final_y_test = np.zeros((len(test_dataset),1))
    true_test_labels = np.zeros((len(test_dataset),1))
    test_idx = 0
    model.eval()
    for x,y in test_dl:
        final_y_test[test_idx:test_idx + x.size(0)] = model(x.type(torch.FloatTensor).to(device)).cpu().detach().numpy()
        true_test_labels[test_idx:test_idx + x.size(0)] = y.detach().numpy()
        test_idx += x.size(0)

    final_test = [1 if y_test > opt_th else 0 for y_test in final_y_test]
    f1_test = f1_score(true_test_labels, final_test)
    acc_test = accuracy_score(true_test_labels, final_test)
    fpr_test, tpr_test,_ = roc_curve(true_test_labels, final_y_test)

    print('Test F1')
    print(f1_test)
    print('Test Accuracy')
    print(acc_test)

    final_y_10db = np.zeros((len(dataset_10db),1))
    true_10db_labels = np.zeros((len(dataset_10db),1))
    test_idx = 0
    model.eval()
    for x,y in test_dl_10db:
        final_y_10db[test_idx:test_idx + x.size(0)] = model(x.type(torch.FloatTensor).to(device)).cpu().detach().numpy()
        true_10db_labels[test_idx:test_idx + x.size(0)] = y.detach().numpy()
        test_idx += x.size(0)

    final_10db = [1 if y_test > opt_th else 0 for y_test in final_y_10db]
    f1_10db = f1_score(true_10db_labels, final_10db)
    acc_10db = accuracy_score(true_10db_labels, final_10db)

    print('Test F1 10dB')
    print(f1_10db)
    print('Test Accuracy 10dB')
    print(acc_10db)

    final_y_5db = np.zeros((len(dataset_5db),1))
    true_5db_labels = np.zeros((len(dataset_5db),1))
    test_idx = 0
    model.eval()
    for x,y in test_dl_5db:
        final_y_5db[test_idx:test_idx + x.size(0)] = model(x.type(torch.FloatTensor).to(device)).cpu().detach().numpy()
        true_5db_labels[test_idx:test_idx + x.size(0)] = y.detach().numpy()
        test_idx += x.size(0)

    final_5db = [1 if y_test > opt_th else 0 for y_test in final_y_5db]
    f1_5db = f1_score(true_5db_labels, final_5db)
    acc_5db = accuracy_score(true_5db_labels, final_5db)

    print('Test F1 5dB')
    print(f1_5db)
    print('Test Accuracy 5dB')
    print(acc_5db)

    final_y_0db = np.zeros((len(dataset_0db),1))
    true_0db_labels = np.zeros((len(dataset_0db),1))
    test_idx = 0
    model.eval()
    for x,y in test_dl_0db:
        final_y_0db[test_idx:test_idx + x.size(0)] = model(x.type(torch.FloatTensor).to(device)).cpu().detach().numpy()
        true_0db_labels[test_idx:test_idx + x.size(0)] = y.detach().numpy()
        test_idx += x.size(0)

    final_0db = [1 if y_test > opt_th else 0 for y_test in final_y_0db]
    f1_0db = f1_score(true_0db_labels, final_0db)
    acc_0db = accuracy_score(true_0db_labels, final_0db)

    print('Test F1 0dB')
    print(f1_0db)
    print('Test Accuracy 0dB')
    print(acc_0db)

    plt.figure()
    plt.plot(final_val, 'b')
    plt.plot(final_val, 'r')
    plt.plot(final_y_val, 'g--')

    plt.figure()
    plt.plot(fpr_val, tpr_val, color = "darkorange")
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.show()

def roc_optimum_th(tpr, fpr, th):
    """
    Define optimum threshold using ROC curve data
    """

    distances = tpr - fpr
    max_idx = np.argmax(distances)
    opt_th = th[max_idx]
    return opt_th

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    main()