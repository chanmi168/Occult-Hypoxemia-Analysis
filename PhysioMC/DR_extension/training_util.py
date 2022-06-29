import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
from scipy.special import softmax

import numpy as np

from dataIO import *
from setting import *
from stage3_preprocess import *
# from dataset_util import *
from DR_extension.dataset_util import *
from handy_tools import *
import wandb

# def train_resnet(model, dataloader, optimizer, criterion, epoch, training_params):
def train_dann(model, dataloader, training_params):

    optimizer = training_params['optimizer']
    criterion = training_params['criterion']
    epoch = training_params['epoch']
    device = torch.device('cuda:{}'.format(int(training_params['cuda_i'])) if torch.cuda.is_available() else 'cpu')

    dataset_size = len(dataloader.dataset)
#     total_loss = 0
    
    
#     total_AE = dict.fromkeys( ["Apple", "Pear", "Peach", "Banana"], 0)

#     AE_names = []
#     for input_name in model.feature_extractors.keys():
#         for regressor_name in model.regressors.keys():
#             AE_names.append('{}-{}'.format(regressor_name, input_name))

    # total_AE = dict.fromkeys(training_params['model_out_names'], 0)
    # total_AE = {k:v for k,v in total_AE.items() if 'domain' not in k}

    total_losses = dict.fromkeys(training_params['model_out_names']+['total']+['KLD'], 0)


    model.train()
    
    for i, (data, feature, label, meta) in enumerate(dataloader):
#         print('epoch', i)
        # 1. get data
        data = data.to(device=device, dtype=torch.float)
        feature = feature.to(device=device, dtype=torch.float)
        label = label.to(device=device, dtype=torch.float).long()
#         print(label.size())
#         sys.exit()
        
        
#         data = data.to(device).float()
# #         label = label.to(device=device, dtype=torch.float)
#         label = label.to(device=device, dtype=torch.float)

        # 2. infer by net
        output, feature_out, mu, logvar = model(data, feature)
#         out = model(data)
        
        # 3. loss function
        # print(out['race-PPG'].size(), label.size())
        # print(feature_out['PPG'].size())
        # print(label[:,[i_race]].size())
         # output['race-PPG'].size(), feature_out['PPG'].size())

#         print(out, label)
#         print(out, label, training_params['regressor_names'])


        # losses = criterion(out['race-PPG'], label[:, i_race])
        losses = criterion(output, label, mu=mu, logvar=logvar)
    
#         loss = losses['total']
        
#         print(losses.keys())
        
        
#         print(out, losses)
        


#         total_loss += loss.data.detach().cpu().numpy()

        # 3. Backward and optimize
        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()
        

        # 4. accumulate the loss
        for loss_name in total_losses.keys():
            total_losses[loss_name] += losses[loss_name].data.detach().cpu().numpy()
            
        # # 5. compute metric
        # for AE_names in total_AE.keys():
        #     main_task = AE_names.split('-')[0]
        #     label_AE = label[:, [training_params['output_names'].index(main_task.split('-')[0]) ]].data.detach().cpu().numpy()
        #     out_AE = out[AE_names].data.detach().cpu().numpy()
        #     total_AE[AE_names] += np.sum(np.abs(out_AE.squeeze()-label_AE.squeeze()).squeeze())


#         main_task = training_params['output_names'][0]
#         label = label[:, [training_params['output_names'].index(main_task.split('-')[0]) ]].data.detach().cpu().numpy()
#         out = out[main_task].data.detach().cpu().numpy()
    
#         total_AE += np.sum(np.abs(out.squeeze()-label.squeeze()).squeeze())

#         label = {task: label[:, [self.tasks.index(task)]] for task in self.tasks}


#     total_loss = total_loss/dataset_size
    # for loss_name in total_losses.keys():
    #     total_losses[loss_name] == total_losses[loss_name]/dataset_size
            
    # subject_id = training_params['CV_config']['subject_id']
    
    
    performance_dict = {}
    for loss_name in total_losses.keys():
        performance_dict['train_{}'.format(loss_name)] = total_losses[loss_name]

    performance_dict['epoch'] = epoch
    
    
    if training_params['wandb']==True:
        # W&B
        wandb.log(performance_dict)
    
#     performance_dict = {'total_loss': total_loss,
#                        }
    # performance_dict = total_losses
    
    return performance_dict

#     log_dict = {}
#     for loss_name in total_losses.keys():
#         log_dict['train_{}'.format(subject_id, loss_name)] = total_losses[loss_name]

#     log_dict['epoch'] = epoch
    
    
#     if training_params['wandb']==True:
#         # W&B
#         wandb.log(log_dict)
# #         wandb.log({
# # #             '[{}] train_loss'.format(subject_id): total_loss, 
# #             '[{}] train_MAE'.format(subject_id): MAE, 
# #             'epoch': epoch, })
    
    
#     # # TODO: remove performance_dict
# #     performance_dict = {
# #         'total_loss': total_loss,
# #     }
#     performance_dict = total_losses

#     return performance_dict


# def eval_resnet(model, dataloader, optimizer, criterion, epoch, training_params):
def eval_dann(model, dataloader, training_params):

    optimizer = training_params['optimizer']
    criterion = training_params['criterion']
    epoch = training_params['epoch']
    
    device = torch.device('cuda:{}'.format(int(training_params['cuda_i'])) if torch.cuda.is_available() else 'cpu')

    dataset_size = len(dataloader.dataset)

#     total_loss = 0

    total_losses = dict.fromkeys(training_params['model_out_names']+['total']+['KLD'], 0)

    
#     total_AE = {}
#     for model_out_name in training_params['model_out_names']:
#         total_AE[model_out_name] = 0
        
    # total_AE = dict.fromkeys(training_params['model_out_names'], 0)
    # total_AE = {k:v for k,v in total_AE.items() if 'domain' not in k}
        

    model.eval()
#     print('\t\tswitch model to eval')

    for i, (data, feature, label, meta) in enumerate(dataloader):
        # 1. get data        
        data = data.to(device=device, dtype=torch.float)
        feature = feature.to(device=device, dtype=torch.float)
        label = label.to(device=device, dtype=torch.float).long()

        # 2. infer by net
        output, feature_out, mu, logvar = model(data, feature)

            
        # 3. loss function
        losses = criterion(output, label, mu=mu, logvar=logvar)
        # losses = criterion(out['race-PPG'], label[:, i_race])

#         loss = losses['total']

        # 4. accumulate the loss
        for loss_name in total_losses.keys():
            total_losses[loss_name] += losses[loss_name].data.detach().cpu().numpy()
            


        # for AE_names in total_AE.keys():
        #     main_task = AE_names.split('-')[0]
        #     label_AE = label[:, [training_params['output_names'].index(main_task.split('-')[0]) ]].data.detach().cpu().numpy()
        #     out_AE = out[AE_names].data.detach().cpu().numpy()
        #     total_AE[AE_names] += np.sum(np.abs(out_AE.squeeze()-label_AE.squeeze()).squeeze())


    
    
    

#     total_loss = total_loss/dataset_size
#     MAE = total_AE/dataset_size

#     subject_id = training_params['CV_config']['subject_id']
#     if training_params['wandb']==True:
#         # W&B

#         wandb.log({
# #             '[{}] val_loss'.format(subject_id): total_loss, 
#             '[{}] val_MAE'.format(subject_id): MAE, 
#             'epoch': epoch, })



#     total_loss = total_loss/dataset_size
    # for loss_name in total_losses.keys():
    #     total_losses[loss_name] == total_losses[loss_name]
            
    # subject_id = training_params['CV_config']['subject_id']
    
    performance_dict = {}
    for loss_name in total_losses.keys():
        performance_dict['val_{}'.format(loss_name)] = total_losses[loss_name]

    performance_dict['epoch'] = epoch
    
    
    if training_params['wandb']==True:
        # W&B
        wandb.log(performance_dict)
    
#     performance_dict = {'total_loss': total_loss,
#                        }
    # performance_dict = total_losses
    
    return performance_dict




# def pred_resnet(model, dataloader, criterion, epoch, training_params):
def pred_dann(model, dataloader, training_params):

    optimizer = training_params['optimizer']
    criterion = training_params['criterion']
    # print(criterion)
    epoch = training_params['epoch']
    
    device = torch.device('cuda:{}'.format(int(training_params['cuda_i'])) if torch.cuda.is_available() else 'cpu')

    dataset_size = len(dataloader.dataset)

#     total_loss = 0
    total_losses = dict.fromkeys(training_params['model_out_names']+['total']+['KLD'], 0)

#     out_arr = np.empty(0)
#     label_arr = np.empty(0)
#     out_dict = {}
#     label_dict= {}
    out_dict = {}
    for model_out_name in training_params['model_out_names']:
        out_dict[model_out_name] = []

    label_dict = {}
    for model_out_name in training_params['model_out_names']:
        label_dict[model_out_name] = []

#     out_dict = dict.fromkeys(training_params['model_out_names'], list() )
#     label_dict = dict.fromkeys(training_params['model_out_names'], list() )
#     print(out_dict)
#     for task in training_params['tasks']:
#         out_dict[task] = np.empty(0)
#         label_dict[task] = np.empty(0)


    feature_arr = []
    meta_arr = []
    
    mu_arr = []
    logvar_arr = []
    
    model.eval()
#     print('\t\tswitch model to eval')

    for i, (data, feature, label, meta) in enumerate(dataloader):
        # 1. get data
        data = data.to(device=device, dtype=torch.float)
        feature = feature.to(device=device, dtype=torch.float)
        label = label.to(device=device, dtype=torch.float).long()
#         data = data.to(device).float()
#         label = label.to(device=device, dtype=torch.float)

        # 2. infer by net
        # out, _ = model(data, feature)
        output, feature_out, mu, logvar = model(data, feature)

#         out = model(data)
            
        # 3. loss function
#         loss = criterion(out, label)
        losses = criterion(output, label, mu=mu, logvar=logvar)
        # losses = criterion(out['race-PPG'], label[:, i_race])

#         loss = losses['total']
        
        # 4. compute the class loss of features
#         total_loss += loss.data.detach().cpu().numpy()

        # print(losses)
        # 4. accumulate the loss
        for loss_name in total_losses.keys():
            total_losses[loss_name] += losses[loss_name].data.detach().cpu().numpy()
            # print(loss_name)
            


        # TODO: fix this block (out_dict)
#         for output_name in out.keys():
#         print(label)

        for output_name in out_dict.keys():
#             print(output_name)
#             print('===== out =====')
            out_dict[output_name].append(output[output_name].detach().cpu().numpy())
            
        
#             print('===== label =====')
            # i_race = training_params['dataset_dict']['list_label'].index('Race String')
            # i_race = training_params['dataset_dict']['list_label'].index('Race String')
            # label_dict = {}
            # label_dict['race-PPG'] = label[:, i_race]
            
            label_dict[output_name].append( label[:,training_params['output_names'].index(output_name.split('-')[0]) ].detach().cpu().numpy() )


            # if 'domain' in output_name:
            #     input_name = output_name.split('-')[1]
            #     label_dict[output_name].append( np.ones(label.size()[0]) * training_params['modality_dict'][input_name] )
            # else:
            #     label_dict[output_name].append( label[:,training_params['output_names'].index(output_name.split('-')[0]) ].detach().cpu().numpy() )



        feature_arr.append( feature.detach().cpu().numpy())
        meta_arr.append( meta.detach().cpu().numpy())

        mu_arr.append( mu.detach().cpu().numpy())
        logvar_arr.append( logvar.detach().cpu().numpy())


#     print(out_dict[output_name])
    
#     print(np.concatenate(out_dict[output_name]).shape)
#     sys.exit()

    for output_name in out_dict.keys():
        out_dict[output_name] = np.concatenate(out_dict[output_name])
        
        # print(out_dict[output_name].shape)
        m = ( softmax(out_dict[output_name],axis=-1)[:,0]<DEFAULT_ROC_thre ).astype(int) # dimen from Nx2 to N
        # print(m.shape)
        out_dict[output_name] = m
        # out_dict[output_name] = np.concatenate(out_dict[output_name]).squeeze()

        
    for output_name in label_dict.keys():
        label_dict[output_name] = np.concatenate(label_dict[output_name])

    
    feature_arr = np.concatenate(feature_arr,axis=0)
    meta_arr = np.concatenate(meta_arr,axis=0)
    
    mu_arr = np.concatenate(mu_arr,axis=0)
    logvar_arr = np.concatenate(logvar_arr,axis=0)

    performance_dict = {'out_dict': out_dict,
                        'label_dict': label_dict,
                        'feature_arr': feature_arr,
                        'meta_arr': meta_arr,
                        'mu_arr': mu_arr,
                        'logvar_arr': logvar_arr,
                       }
    

    # for loss_name in total_losses.keys():
    #     total_losses[loss_name] == total_losses[loss_name]
    
    performance_dict = Merge(performance_dict, total_losses)

    return performance_dict



# TODO: implement this based on model_features_diagnosis
# ref: https://github.com/chanmi168/Fall-Detection-DAT/blob/master/falldetect/eval_util.py
def visualize_latent(model, training_params, fig_name=None, show_plot=False, outputdir=None, log_wandb=False):
    
    # dataloaders, dataset_sizes = get_loaders(training_params['inputdir'], training_params)
    data = training_params['data'] 
    label = training_params['label']
    dataloaders, dataset_sizes, training_params = get_loaders(data, label, training_params)

    
    data_val = dataloaders['val'].dataset.data
    feature_val = dataloaders['val'].dataset.feature
    # meta = dataloaders['val'].dataset.meta
    label_val = dataloaders['val'].dataset.label

    data_val = torch.from_numpy(data_val)
    feature_val = torch.from_numpy(feature_val)

    data_val = data_val.to(device=training_params['device'], dtype=torch.float)
    feature_val = feature_val.to(device=training_params['device'], dtype=torch.float)
    
    # 2. infer by net
    out, feature_out, mu, logvar = model(data_val, feature_val)

    
    for input_name in feature_out.keys():
        feature_sig = feature_out['input_name'].cpu().detach().numpy()
        feature_np = StandardScaler().fit_transform(feature_sig) # normalizing the features
        print('show standardize mean and std:', np.mean(feature_np),np.std(feature_np))
        
        if DR_mode == 'PCA':
            pca_features = PCA(n_components=10)
            principalComponents_features = pca_features.fit_transform(feature_np)
            var_pca = np.cumsum(np.round(pca_features.explained_variance_ratio_, decimals=3)*100)
            print('PCA var:', var_pca)
            explained_var = var_pca[1]

            fig, ax = plt.subplot(1,1, figsize=(5, 5), dpi=80)

            ax.set_xlabel('Principal Component - 1',fontsize=12)
            ax.set_ylabel('Principal Component - 2',fontsize=12)
            ax.set_title('{} (explained_var: {:.2f}%)'.format(input_name, explained_var),fontsize=15)
            # ax.set_title('PCA of features extracted by Gf ({})'.format(col_name),fontsize=15)
            ax.tick_params(axis='both', which='major', labelsize=12)

            class_ids = [0, 1] # adl, fall
            domain_ids = [0, 1] # src, tgt
            colors = ['r', 'g']
            markers = ['o', 'x']
            legend_dict = {
                '00': 'adl_src',
                '01': 'adl_tgt',
                '10': 'fall_src',
                '11': 'fall_tgt',
            }

            pt_label = ['']

            for class_id, marker in zip(class_ids,markers):
              for domain_id, color in zip(domain_ids,colors):
                indicesToKeep = np.where((labels_np==class_id) & (domain_np==domain_id))[0]

                if class_id == 1:
                  alpha = 0.3
                  ax.scatter(principalComponents_features[indicesToKeep, 0], 
                              principalComponents_features[indicesToKeep, 1], 
                              s = 50, marker=marker, c=color, alpha=alpha,
                            label=legend_dict[str(class_id)+str(domain_id)])
                else:
                  alpha = 0.3
                  ax.scatter(principalComponents_features[indicesToKeep, 0], 
                              principalComponents_features[indicesToKeep, 1], 
                              s = 50, marker=marker, edgecolors=color, facecolors='None', alpha=alpha,
                            label=legend_dict[str(class_id)+str(domain_id)])

            ax.legend(loc='upper right', prop={'size': 15})

        
        
    
    
    # print('out', out)
    # print('feature_out', feature_out)
    # sys.exit()
    
    

#   src_feature, src_class_out, src_domain_out = model(src_data)
    return


def train_model(model, training_params, trainer, evaler, preder):

    inputdir = training_params['inputdir']
    
    

    data = training_params['data'] 
    label = training_params['label']
    
    # dataloaders, dataset_sizes = get_loaders(inputdir, training_params)
    # dataloaders, dataset_sizes, training_params = get_loaders(data, label, training_params)
    dataloaders, dataset_sizes, training_params = get_loaders(training_params)

    # store the total losses
    total_losses_train = dict.fromkeys(training_params['model_out_names']+['total'], np.zeros(training_params['num_epochs']))
    total_losses_val = dict.fromkeys(training_params['model_out_names']+['total'], np.zeros(training_params['num_epochs']))

#     if training_params['wandb']==True:
#         # tell wandb to watch what the model gets up to: gradients, weights, and more!
#         wandb.watch(model, log="all", log_freq=10)
    
    
    print('\t start training.....')

    df_losses_train = pd.DataFrame()
    df_losses_val = pd.DataFrame()
        
    for epoch in range(training_params['num_epochs']):
        if epoch%10==1:
            print('\t[{} epoch]'.format(ordinal(epoch)))
        training_params['epoch'] = epoch

        ##### model training mode ####
        performance_dict_train = trainer(model, dataloaders['train'], training_params)
        
        # df_losses_train = df_losses_train.append(  pd.DataFrame(performance_dict_train, index=[0]), ignore_index=True )
        df_losses_train = pd.concat([df_losses_train,  pd.DataFrame(performance_dict_train, index=[0])])

#         print(performance_dict_train)
#         total_loss_train[epoch] = performance_dict_train['total']

        performance_dict_val = evaler(model, dataloaders['val'], training_params)
        # df_losses_val = df_losses_val.append(  pd.DataFrame(performance_dict_val, index=[0]), ignore_index=True )
        df_losses_val = pd.concat([df_losses_val,  pd.DataFrame(performance_dict_val, index=[0])])

        
        # if epoch==1 or  epoch==training_params['num_epochs']//2 or epoch==training_params['num_epochs']-1:
        #     visualize_latent(model, training_params, fig_name='epoch{}'.format(epoch), show_plot=False, outputdir=None, log_wandb=False)

    print('\t done with training.....')
    
#     print(df_losses_train)
#     print(df_losses_val)

#     sys.exit()
    performance_dict_train = preder(model, dataloaders['train'], training_params)
    performance_dict_val = preder(model, dataloaders['val'], training_params)

    
    CV_dict = {
        'performance_dict_train': performance_dict_train,
#         'total_loss_train': df_losses_train['total'].values,
        'df_losses_train': df_losses_train,
        'performance_dict_val': performance_dict_val,
#         'total_loss_val': df_losses_val['total'].values,
        'df_losses_val': df_losses_val,
        'model': model,
        'CV': training_params['CV_config']['CV'],
        # 'subject_id_val': training_params['CV_config']['subject_id'], 
    }
    
    return CV_dict


def change_output_dim(training_params):
    input_dim = training_params['data_dimensions'][1]
    output_dim = input_dim

    for i_macro in range(training_params['n_block_macro']-1):
        output_dim = np.ceil(output_dim/training_params['stride'])

    output_dim = int(output_dim)
    training_params['output_dim'] = output_dim
    return training_params
