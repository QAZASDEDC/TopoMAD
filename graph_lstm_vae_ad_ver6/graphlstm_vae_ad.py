import logging

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import trange,tqdm
from torch_geometric.nn import GCNConv, GATConv
import IPython
import matplotlib.pyplot as plt
#from sklearn.preprocessing import StandardScaler

from .algorithm_utils import Algorithm, PyTorchUtils
from .graphlstm import GraphLSTM
from .graphlstm_vae import GraphLSTM_VAE

class GraphLSTM_VAE_AD(Algorithm, PyTorchUtils):
    def __init__(self, name: str='GraphLSTM_VAE_AD', num_epochs: int=10, batch_size: int=32, lr: float=1e-3,
                 hidden_dim: int=5, sequence_length: int=30, num_layers: tuple=(2, 2),
                 head: tuple=(1,1), dropout: tuple=(0,0), kind: str='GCN',
                 bias: tuple=(True, True), variational: bool=True,
                 seed: int=None, gpu: int = None, details=True):
        Algorithm.__init__(self, __name__, name, seed, details=details)
        PyTorchUtils.__init__(self, seed, gpu)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr

        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length

        self.num_layers = num_layers
        self.head = head
        self.dropout = dropout
        self.kind = kind
        self.bias = bias
        self.variational = variational

        self.lstmed = None
        #self.sscaler = StandardScaler()

    def fit(self, X: pd.DataFrame, nodes_num: int, edge_index: list, log_step: int = 20, patience: int = 10, selected_indexes = None):
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values

        #data = self.sscaler.fit_transform(data)

        if selected_indexes is None:
            sequences = [data[i:i + self.sequence_length].reshape(self.sequence_length, nodes_num, -1) for i in range(data.shape[0] - self.sequence_length + 1)]
            #train_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=True,shuffle=True,pin_memory=True)
        else:
            sequences = [data[i:i + self.sequence_length].reshape(self.sequence_length, nodes_num, -1) for i in selected_indexes]

        indices = np.random.permutation(len(sequences))
        split_point = int(0.3 * len(sequences))
        train_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=True,
                                  sampler=SubsetRandomSampler(indices[:-split_point]), pin_memory=False)
        valid_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=True,
                                  sampler=SubsetRandomSampler(indices[-split_point:]), pin_memory=False)

        self.lstmed = GraphLSTM_VAE(nodes_num, X.shape[1]//nodes_num, self.hidden_dim,
                                    num_layers=self.num_layers, head=self.head, dropout=self.dropout, kind=self.kind,
                                    bias=self.bias, variational=self.variational, seed=self.seed, gpu=self.gpu)
        #self.lstmed.init_weights()
        self.to_device(self.lstmed)
        optimizer = torch.optim.Adam(self.lstmed.parameters(), lr=self.lr, weight_decay=1e-4)
        edge_index = self.to_var(torch.tensor(edge_index, dtype=torch.long))
        iters_per_epoch = len(train_loader)
        counter = 0
        best_val_loss = np.Inf

        teacher_forcing_ratio = 1

        if self.variational:
            for epoch in range(self.num_epochs):
                logging.debug(f'Epoch {epoch+1}/{self.num_epochs}.')
                self.lstmed.train()
                for (i,ts_batch) in enumerate(tqdm(train_loader)):
                    use_teacher_forcing = random.random() < teacher_forcing_ratio
                    output, enc_hidden, mu, logvar, output_logvar = self.lstmed(self.to_var(ts_batch), edge_index, use_teacher_forcing)
                    total_loss, recon_loss, kl_loss = self.lstmed.loss_function(output, self.to_var(ts_batch.float()), mu, logvar, output_logvar)

                    loss = {}
                    loss['total_loss'] = total_loss.data.item()
                    loss['recon_loss'] = recon_loss.data.item()
                    loss['kl_loss'] = kl_loss.data.item()

                    self.lstmed.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.lstmed.parameters(), 5)
                    optimizer.step()

                    if (i+1) % log_step == 0:
                        IPython.display.clear_output()
                        log = "Epoch [{}/{}], Iter [{}/{}]".format(
                            epoch+1, self.num_epochs, i+1, iters_per_epoch)

                        for tag, value in loss.items():
                            log += ", {}: {:.4f}".format(tag, value)
                        print(log)
                    
                        plt_ctr = 1
                        if not hasattr(self,"loss_logs"):
                            self.loss_logs = {}
                            for loss_key in loss:
                                self.loss_logs[loss_key] = [loss[loss_key]]
                                plt.subplot(2,2,plt_ctr)
                                plt.plot(np.array(self.loss_logs[loss_key]), label=loss_key)
                                plt.legend()
                                plt_ctr += 1
                        else:
                            for loss_key in loss:
                                self.loss_logs[loss_key].append(loss[loss_key])
                                plt.subplot(2,2,plt_ctr)
                                plt.plot(np.array(self.loss_logs[loss_key]), label=loss_key)
                                plt.legend()
                                plt_ctr += 1
                            if 'valid_loss' in self.loss_logs:
                                plt.subplot(2,2,plt_ctr)
                                plt.plot(np.array(self.loss_logs['valid_loss']), label='valid_loss')
                                plt.legend()
                                print("valid_loss:", self.loss_logs['valid_loss'])
                        plt.show()
                
                self.lstmed.eval()
                valid_losses = []
                for (i,ts_batch) in enumerate(tqdm(valid_loader)):
                    output, enc_hidden, mu, logvar, output_logvar = self.lstmed(self.to_var(ts_batch), edge_index)
                    total_loss, recon_loss, kl_loss = self.lstmed.loss_function(output, self.to_var(ts_batch.float()), mu, logvar, output_logvar)
                    valid_losses.append(total_loss.item())
                valid_loss = np.average(valid_losses)
                if 'valid_loss' in self.loss_logs:
                    self.loss_logs['valid_loss'].append(valid_loss)
                else:
                    self.loss_logs['valid_loss'] = [valid_loss]

                if valid_loss < best_val_loss:
                    best_val_loss = valid_loss
                    teacher_forcing_ratio -= 1.0/self.num_epochs
                    torch.save(self.lstmed.state_dict(), self.name+'_'+self.kind+str(self.gpu)+'_'+'checkpoint.pt')
                    counter = 0
                else:
                    counter += 1
                    teacher_forcing_ratio *= 0.5
                    if counter >= patience:
                        self.lstmed.load_state_dict(torch.load(self.name+'_'+self.kind+str(self.gpu)+'_'+'checkpoint.pt'))
                        break          
        else:
            for epoch in trange(self.num_epochs):
                logging.debug(f'Epoch {epoch+1}/{self.num_epochs}.')
                self.lstmed.train()
                for (i,ts_batch) in enumerate(tqdm(train_loader)):
                    use_teacher_forcing = random.random() < teacher_forcing_ratio
                    output, enc_hidden = self.lstmed(self.to_var(ts_batch), edge_index, use_teacher_forcing)
                    total_loss = self.lstmed.loss_function(output, self.to_var(ts_batch.float()))

                    loss = {}
                    loss['total_loss'] = total_loss.data.item()

                    self.lstmed.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.lstmed.parameters(), 5)
                    optimizer.step()

                    if (i+1) % log_step == 0:
                        IPython.display.clear_output()
                        log = "Epoch [{}/{}], Iter [{}/{}]".format(
                            epoch+1, self.num_epochs, i+1, iters_per_epoch)

                        for tag, value in loss.items():
                            log += ", {}: {:.4f}".format(tag, value)
                        print(log)
                    
                        plt_ctr = 1
                        if not hasattr(self,"loss_logs"):
                            self.loss_logs = {}
                            for loss_key in loss:
                                self.loss_logs[loss_key] = [loss[loss_key]]
                                plt.subplot(2,2,plt_ctr)
                                plt.plot(np.array(self.loss_logs[loss_key]), label=loss_key)
                                plt.legend()
                                plt_ctr += 1
                        else:
                            for loss_key in loss:
                                self.loss_logs[loss_key].append(loss[loss_key])
                                plt.subplot(2,2,plt_ctr)
                                plt.plot(np.array(self.loss_logs[loss_key]), label=loss_key)
                                plt.legend()
                                plt_ctr += 1
                            if 'valid_loss' in self.loss_logs:
                                plt.subplot(2,2,plt_ctr)
                                plt.plot(np.array(self.loss_logs['valid_loss']), label='valid_loss')
                                plt.legend()
                                print("valid_loss:", self.loss_logs['valid_loss'])
                        plt.show()

                self.lstmed.eval()
                valid_losses = []
                for (i,ts_batch) in enumerate(tqdm(valid_loader)):
                    output, enc_hidden = self.lstmed(self.to_var(ts_batch), edge_index)
                    total_loss = self.lstmed.loss_function(output, self.to_var(ts_batch.float()))
                    valid_losses.append(total_loss.item())
                valid_loss = np.average(valid_losses)
                if 'valid_loss' in self.loss_logs:
                    self.loss_logs['valid_loss'].append(valid_loss)
                else:
                    self.loss_logs['valid_loss'] = [valid_loss]

                if valid_loss < best_val_loss:
                    best_val_loss = valid_loss
                    teacher_forcing_ratio -= 1.0/self.num_epochs
                    torch.save(self.lstmed.state_dict(), self.name+'_'+self.kind+str(self.gpu)+'_'+'checkpoint.pt')
                    counter = 0
                else:
                    counter += 1
                    teacher_forcing_ratio *= 0.5
                    if counter >= patience:
                        print ("early stoppong")
                        self.lstmed.load_state_dict(torch.load(self.name+'_'+self.kind+str(self.gpu)+'_'+'checkpoint.pt'))
                        break
    
    def load(self, nodes_num, X_shape):
        self.lstmed = GraphLSTM_VAE(nodes_num, X_shape//nodes_num, self.hidden_dim,
                                    num_layers=self.num_layers, head=self.head, dropout=self.dropout, kind=self.kind,
                                    bias=self.bias, variational=self.variational, seed=self.seed, gpu=self.gpu)
        self.lstmed.load_state_dict(torch.load(self.name+'_'+self.kind+str(self.gpu)+'_'+'checkpoint.pt'))
        self.to_device(self.lstmed)
        
    def predict(self, X: pd.DataFrame, nodes_num: int, edge_index: list, sampling_num: int, delay: int = 5):
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values

        #data = self.sscaler.transform(data)

        sequences = [data[i:i + self.sequence_length].reshape(self.sequence_length, nodes_num, -1) for i in range(data.shape[0] - self.sequence_length + 1)]
        data_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, shuffle=False, drop_last=False)

        self.lstmed.eval()

        edge_index = self.to_var(torch.tensor(edge_index, dtype=torch.long))

        scores_sum = []
        scores_max = []
        outputs = []

        with torch.no_grad():
            if self.variational:
                for (i,ts_batch) in enumerate(tqdm(data_loader)):

                    sample_scores_sum = []
                    sample_scores_max = []
                    sample_outputs = []

                    for j in range(sampling_num):
                        output, enc_hidden, mu, logvar, output_logvar = self.lstmed(self.to_var(ts_batch), edge_index)

                        error_origin = torch.div((output - self.to_var(ts_batch.float())) ** 2, output_logvar.exp()) + output_logvar

                        sample_score_sum = torch.sum(error_origin, (2,3))
                        sample_score_max = torch.max(torch.sum(error_origin, 3), 2).values

                        sample_scores_sum.append(sample_score_sum)
                        sample_scores_max.append(sample_score_max)              

                    score_sum = torch.mean(torch.stack(sample_scores_sum,2),2)
                    score_max = torch.mean(torch.stack(sample_scores_max,2),2)

                    scores_sum.append(score_sum.data.cpu().numpy())
                    scores_max.append(score_max.data.cpu().numpy())
                    outputs.append(enc_hidden.data.cpu().numpy())

            else:
                for (i,ts_batch) in enumerate(tqdm(data_loader)):
                    output, enc_hidden = self.lstmed(self.to_var(ts_batch), edge_index)
                    error_origin = nn.MSELoss(reduction = 'none')(output, self.to_var(ts_batch.float()))

                    score_sum = torch.sum(error_origin, (2,3))
                    score_max = torch.max(torch.sum(error_origin, 3), 2).values

                    scores_sum.append(score_sum.data.cpu().numpy())
                    scores_max.append(score_max.data.cpu().numpy())
                    outputs.append(enc_hidden.data.cpu().numpy())

        scores_sum = np.concatenate(scores_sum)
        scores_max = np.concatenate(scores_max)
        outputs = np.concatenate(outputs)


        lattice = np.full((delay, len(sequences)+delay-1), np.nan)
        for i, score in enumerate(scores_sum):
            lattice[i % delay, i:i + delay] = score[-delay:]
        scores_sum = np.nanmean(lattice, axis=0)

        lattice = np.full((delay, len(sequences)+delay-1), np.nan)
        for i, score in enumerate(scores_max):
            lattice[i % delay, i:i + delay] = score[-delay:]
        scores_max = np.nanmean(lattice, axis=0)

        #return scores begin from beginning+seq_len-delay
        return scores_sum, scores_max, outputs

    def interpret(self, X: pd.DataFrame, nodes_num: int, edge_index: list, sampling_num: int, delay: int = 5):
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values

        #data = self.sscaler.transform(data)

        sequences = [data[i:i + self.sequence_length].reshape(self.sequence_length, nodes_num, -1) for i in range(data.shape[0] - self.sequence_length + 1)]
        data_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, shuffle=False, drop_last=False)

        self.lstmed.eval()

        edge_index = self.to_var(torch.tensor(edge_index, dtype=torch.long))

        scores_sum = []
        outputs = []

        if self.variational:
            for (i,ts_batch) in enumerate(tqdm(data_loader)):
                output, enc_hidden, mu, logvar, output_logvar = self.lstmed(self.to_var(ts_batch), edge_index)

                sample_scores_sum = []

                for j in range(sampling_num):
                    error_origin = torch.div((output - self.to_var(ts_batch.float())) ** 2, output_logvar.exp()) + output_logvar
                    
                    sample_score_sum = torch.sum(error_origin, 3)

                    sample_scores_sum.append(sample_score_sum)            
                
                score_sum = torch.mean(torch.stack(sample_scores_sum,3),3)

                scores_sum.append(score_sum.data.cpu().numpy())
                outputs.append(mu.data.cpu().numpy())
                
        else:
            for (i,ts_batch) in enumerate(tqdm(data_loader)):
                output, enc_hidden = self.lstmed(self.to_var(ts_batch), edge_index)
                error_origin = nn.MSELoss(reduction = 'none')(output, self.to_var(ts_batch.float()))
                
                score_sum = torch.sum(error_origin, 3)

                scores_sum.append(score_sum.data.cpu().numpy())
                outputs.append(enc_hidden.data.cpu().numpy())

        scores_sum = np.concatenate(scores_sum)
        outputs = np.concatenate(outputs)


        lattice = np.full((delay, len(sequences)+delay-1, nodes_num), np.nan)
        for i, score in enumerate(scores_sum):
            lattice[i % delay, i:i + delay] = score[-delay:]
        scores_sum = np.nanmean(lattice, axis=0)

        scores_argsort = np.argsort(-scores_sum)

        #return scores begin from beginning+seq_len-delay
        return scores_sum, outputs, scores_argsort