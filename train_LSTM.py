import numpy as np
import argparse
import os
import random

RANDOM_SEED = 12345
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic=True

from utils import utils
from utils.readers import DecompensationReader
from utils.preprocessing import Discretizer, Normalizer
from utils import metrics
from utils import common_utils
# from model import StageNet

def parse_arguments(parser):
    parser.add_argument('--test_mode', type=int, default=0, help='Test SA-CRNN on MIMIC-III dataset')
    parser.add_argument('--data_path', type=str, metavar='<data_path>', help='The path to the MIMIC-III data directory')
    parser.add_argument('--file_name', type=str, metavar='<data_path>', help='File name to save model')
    parser.add_argument('--small_part', type=int, default=0, help='Use part of training data')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learing rate')

    parser.add_argument('--input_dim', type=int, default=76, help='Dimension of visit record data')
    parser.add_argument('--rnn_dim', type=int, default=384, help='Dimension of hidden units in RNN')
    parser.add_argument('--output_dim', type=int, default=1, help='Dimension of prediction target')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--dropconnect_rate', type=float, default=0.5, help='Dropout rate in RNN')
    parser.add_argument('--dropres_rate', type=float, default=0.3, help='Dropout rate in residue connection')
    parser.add_argument('--K', type=int, default=10, help='Value of hyper-parameter K')
    parser.add_argument('--chunk_level', type=int, default=3, help='Value of hyper-parameter K')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    if args.test_mode == 1:
        print('Preparing test data ... ')

        train_data_loader = common_utils.DeepSupervisionDataLoader(dataset_dir=os.path.join(
            args.data_path, 'train'), listfile=os.path.join(args.data_path, 'train_listfile.csv'), small_part=True)
        discretizer = Discretizer(timestep=1.0, store_masks=True,
                                impute_strategy='previous', start_time='zero')

        discretizer_header = discretizer.transform(train_data_loader._data["X"][0])[1].split(',')
        cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

        normalizer = Normalizer(fields=cont_channels)
        normalizer_state = 'decomp_normalizer'
        normalizer_state = os.path.join(os.path.dirname(args.data_path), normalizer_state)
        normalizer.load_params(normalizer_state)

        test_data_loader = common_utils.DeepSupervisionDataLoader(dataset_dir=os.path.join(args.data_path, 'test'),
                                                                        listfile=os.path.join(args.data_path, 'test_listfile.csv'), small_part=args.small_part)
        test_data_gen = utils.BatchGenDeepSupervision(test_data_loader, discretizer,
                                                    normalizer, args.batch_size,
                                                    shuffle=False, return_names=True)

        print('Constructing model ... ')
        device = torch.device("cuda:0" if torch.cuda.is_available() == True else 'cpu')
        print("available device: {}".format(device))

        model = nn.LSTM(76, 76, bias=True, batch_first=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        linear = nn.Linear(76, 1)
        
        checkpoint = torch.load('./saved_weights/LSTM', map_location='cpu')
        save_chunk = checkpoint['chunk']
        print("last saved model is in chunk {}".format(save_chunk))
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.eval()
        with torch.no_grad():
            cur_test_loss = []
            test_true = []
            test_pred = []
            
            for each_batch in range(test_data_gen.steps):
                test_data = next(test_data_gen)
                test_name = test_data['names']
                test_data = test_data['data']

                test_x = torch.tensor(test_data[0][0], dtype=torch.float32).to(device)
                test_y = torch.tensor(test_data[1], dtype=torch.float32).to(device)
  
                loss = nn.BCELoss()

                test_output, _ = model(test_x)
                test_output = torch.sigmoid(linear(test_output))
                test_loss = loss(test_output, test_y)
                cur_test_loss.append(test_loss.cpu().detach().numpy()) 
                
                for t, p in zip(test_y.cpu().numpy().flatten(), test_output.cpu().detach().numpy().flatten()):
                    test_true.append(t)
                    test_pred.append(p)
            
            print('Test loss = %.4f'%(np.mean(np.array(cur_test_loss))))
            print('\n')
            test_pred = np.array(test_pred)
            test_pred = np.stack([1 - test_pred, test_pred], axis=1)
            test_ret = metrics.print_metrics_binary(test_true, test_pred)

    else:
        ''' Prepare training data'''
        print('Preparing training data ... ')
        train_data_loader = common_utils.DeepSupervisionDataLoader(dataset_dir=os.path.join(
            args.data_path, 'train'), listfile=os.path.join(args.data_path, 'train_listfile.csv'), small_part=args.small_part)
        val_data_loader = common_utils.DeepSupervisionDataLoader(dataset_dir=os.path.join(
            args.data_path, 'train'), listfile=os.path.join(args.data_path, 'val_listfile.csv'), small_part=args.small_part)
        discretizer = Discretizer(timestep=1.0, store_masks=True,
                                impute_strategy='previous', start_time='zero')

        discretizer_header = discretizer.transform(train_data_loader._data["X"][0])[1].split(',')
        cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

        normalizer = Normalizer(fields=cont_channels)
        normalizer_state = 'decomp_normalizer'
        normalizer_state = os.path.join(os.path.dirname(args.data_path), normalizer_state)
        normalizer.load_params(normalizer_state)

        train_data_gen = utils.BatchGenDeepSupervision(train_data_loader, discretizer,
                                                        normalizer, args.batch_size, shuffle=True, return_names=True)
        val_data_gen = utils.BatchGenDeepSupervision(val_data_loader, discretizer,
                                                    normalizer, args.batch_size, shuffle=False, return_names=True)

        '''Model structure'''
        print('Constructing model ... ')
        device = torch.device("cuda:0" if torch.cuda.is_available() == True else 'cpu')
        print("available device: {}".format(device))

        model = nn.LSTM(76, 76, bias=True, batch_first=True).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        linear = nn.Linear(76, 1).to(device)

        '''Train phase'''
        print('Start training ... ')

        train_loss = []
        val_loss = []
        batch_loss = []
        max_auprc = 0

        file_name = './saved_weights/'+args.file_name
        for each_chunk in range(args.epochs):
            cur_batch_loss = []
            model.train()
            for each_batch in range(train_data_gen.steps):
                batch_data = next(train_data_gen)
                batch_name = batch_data['names']
                batch_data = batch_data['data']

                batch_x = torch.tensor(batch_data[0][0], dtype=torch.float32).to(device)
                batch_y = torch.tensor(batch_data[1], dtype=torch.float32).to(device)
    
                optimizer.zero_grad()
                loss = nn.BCELoss()
                
                cur_output, _ = model(batch_x)
                cur_output = torch.sigmoid(linear(cur_output))
                loss = loss(cur_output, batch_y)

                cur_batch_loss.append(loss.cpu().detach().numpy()) 
                
                loss.backward()
                optimizer.step()
                
                if each_batch % 50 == 0:
                    print('Chunk %d, Batch %d: Loss = %.4f'%(each_chunk, each_batch, cur_batch_loss[-1]))

            batch_loss.append(cur_batch_loss)
            train_loss.append(np.mean(np.array(cur_batch_loss)))
            
            print("\n==>Predicting on validation")
            with torch.no_grad():
                model.eval()
                cur_val_loss = []
                valid_true = []
                valid_pred = []
                for each_batch in range(val_data_gen.steps):
                    valid_data = next(val_data_gen)
                    valid_name = valid_data['names']
                    valid_data = valid_data['data']
                    
                    valid_x = torch.tensor(valid_data[0][0], dtype=torch.float32).to(device)
                    valid_y = torch.tensor(valid_data[1], dtype=torch.float32).to(device)
                    valid_output, _ = model(valid_x)
                    valid_output = torch.sigmoid(linear(valid_output))
                    valid_loss = nn.BCELoss()
                    valid_loss = valid_loss(valid_output, valid_y)

                    cur_val_loss.append(valid_loss.cpu().detach().numpy())

                    for t, p in zip(valid_y.cpu().numpy().flatten(), valid_output.cpu().detach().numpy().flatten()):   
                        valid_true.append(t)
                        valid_pred.append(p)

                val_loss.append(np.mean(np.array(cur_val_loss)))
                print('Valid loss = %.4f'%(val_loss[-1]))
                print('\n')
                valid_pred = np.array(valid_pred)
                valid_pred = np.stack([1 - valid_pred, valid_pred], axis=1)
                ret = metrics.print_metrics_binary(valid_true, valid_pred)
                print()

                cur_auprc = ret['auprc']
                if cur_auprc > max_auprc:
                    max_auprc = cur_auprc
                    state = {
                        'net': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'chunk': each_chunk
                    }
                    torch.save(state, file_name)
                    print('\n------------ Save best model ------------\n')


        '''Evaluate phase'''
        print('Testing model ... ')

        checkpoint = torch.load(file_name)
        save_chunk = checkpoint['chunk']
        print("last saved model is in chunk {}".format(save_chunk))
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.eval()

        test_data_loader = common_utils.DeepSupervisionDataLoader(dataset_dir=os.path.join(args.data_path, 'test'),
                                                                        listfile=os.path.join(args.data_path, 'test_listfile.csv'), small_part=args.small_part)
        test_data_gen = utils.BatchGenDeepSupervision(test_data_loader, discretizer,
                                                    normalizer, args.batch_size,
                                                    shuffle=False, return_names=True)

        with torch.no_grad():
            torch.manual_seed(RANDOM_SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(RANDOM_SEED)
        
            cur_test_loss = []
            test_true = []
            test_pred = []
            
            for each_batch in range(test_data_gen.steps):
                test_data = next(test_data_gen)
                test_name = test_data['names']
                test_data = test_data['data']

                test_x = torch.tensor(test_data[0][0], dtype=torch.float32).to(device)
                test_y = torch.tensor(test_data[1], dtype=torch.float32).to(device)


                test_output, _ = model(test_x)
                test_output = torch.sigmoid(linear(test_output))
                loss = nn.BCELoss()
                test_loss = loss(test_output, test_y)
                cur_test_loss.append(test_loss.cpu().detach().numpy()) 
                
                for t, p in zip(test_y.cpu().numpy().flatten(), test_output.cpu().detach().numpy().flatten()):
                    test_true.append(t)
                    test_pred.append(p)
            
            print('Test loss = %.4f'%(np.mean(np.array(cur_test_loss))))
            print('\n')
            test_pred = np.array(test_pred)
            test_pred = np.stack([1 - test_pred, test_pred], axis=1)
            test_ret = metrics.print_metrics_binary(test_true, test_pred)
