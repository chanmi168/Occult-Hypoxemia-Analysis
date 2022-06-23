import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import sys


# from models_CNN import *
# from models_CNN2 import *
# from models_resnet import *
from DR_extension.models_CNNlight import *

    
class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(torch.nn.Module):
    def __init__(self,  N_ch=5, N_feature=10):
        super(UnFlatten, self).__init__()
        self.N_ch = N_ch
        self.N_feature = N_feature
    def forward(self, input):
        # return input.view(input.size(0), self.size, 1) # number of instances, number of channels, number of samples
        return input.view(input.size(0), self.N_ch, self.N_feature) # number of instances, number of channels, number of samples
    

# high level arch
class PPG_compressor(nn.Module):
    def __init__(self, training_params=None):
        super(PPG_compressor, self).__init__()
        
        input_dim = training_params['data_dimensions'][-1]
        self.input_names = training_params['input_names']

        input_channel = training_params['data_dimensions'][0]
        channel_n = training_params['channel_n']
        kernel_size = training_params['kernel_size']
        self.n_classes = training_params['race_encoder'].classes_.shape[0]

        self.output_names = training_params['output_names']
        self.fusion_type = training_params['fusion_type']
        
        featrue_extractor = training_params['featrue_extractor']

        if self.fusion_type=='late':
            self.encoders = nn.ModuleDict(
                [[input_name, featrue_extractor(training_params=training_params, input_channel=1)] for input_name in self.input_names]
            )
        elif self.fusion_type=='early':
            self.encoders = nn.ModuleDict(
                [[input_name, featrue_extractor(training_params=training_params)] for input_name in ['early_fusion']]
            )

        self.N_features = len(training_params['feature_names'])

        self.dummy_param = nn.Parameter(torch.empty(0))

    
        self.output_channels = {}
        self.encoder_layer_dims = {}
        
        feature_out_dim = 0
#         for input_name in self.input_names:
        for input_name in self.encoders.keys():
#             feature_out_dim += self.feature_extractors[input_name].feature_out_dim
            # feature_out_dim = self.encoders[input_name].feature_out_dim
            self.encoder_layer_dims[input_name] = self.encoders[input_name].encoder_layer_dims
            self.output_channels[input_name] = self.encoders[input_name].output_channels        
        
        
        encoder_layer_dims = self.encoder_layer_dims[input_name]
        output_channels = self.output_channels[input_name]
        
        self.Flatten = Flatten()

        z_dim = 25
        self.fc1 = nn.Linear(encoder_layer_dims[-1]*output_channels[-1], z_dim)
        self.fc2 = nn.Linear(encoder_layer_dims[-1]*output_channels[-1], z_dim)
        self.fc3 = nn.Linear(z_dim, encoder_layer_dims[-1]*output_channels[-1])

        self.UnFlatten = UnFlatten(N_ch=output_channels[-1], N_feature=encoder_layer_dims[-1])

        self.main_task = self.output_names[0]

        if self.main_task=='reconstruction':
            self.decoders = nn.ModuleDict()
            for input_name in self.encoders.keys():
    #             feature_out_dim += self.feature_extractors[input_name].feature_out_dim
                # encoder_layer_dims = self.encoders[input_name].encoder_layer_dims
                # print(encoder_layer_dims)
                # sys.exit()
                self.decoders[self.main_task] = Decoder(training_params=training_params, encoder_layer_dims=encoder_layer_dims, encoder_channels=output_channels, input_channel=1)

        # edited 3/22
        else:
            self.classifiers = []
            self.classifiers.append([self.main_task, BinaryClassification(num_classes=self.n_classes, input_dim=feature_out_dim )])
            self.classifiers = nn.ModuleDict(self.classifiers)
        


        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_().to(mu.device)
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(mu.device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
        
    def forward(self, x, feature):
        
        if len(feature.size())==3:
            feature = feature[:,:,0]

        output = {}
        feature_out = {}
        latent = {}
        
        for i, input_name in enumerate(self.encoders.keys()):
            feature_out[input_name] = self.encoders[input_name](x[:, [i], :])
            z = self.Flatten(feature_out[input_name])
            
            # use linear layers to map to hidden dim for mu and logvar
            # z is the data sampled using mu, logvar (reparameterization), has the same dimension as feature_out[input_name]
            z, mu, logvar = self.bottleneck(z)
            
            latent[input_name] = z # give it back its channel dim
            latent[input_name] = self.fc3(latent[input_name])
            # latent[input_name] = latent[input_name][:, None, :]

            
            latent[input_name] = self.UnFlatten(latent[input_name])


    
            
            if self.main_task=='reconstruction':
                # # for regressor_name in self.classifiers.keys():
                # for i, input_name in enumerate(self.decoders.keys()):
                #     print()
                # print(feature_out[input_name].shape)
                # print(self.decoders[self.main_task])
                

                for regressor_name in self.decoders.keys():
                    # output[regressor_name+'-{}'.format(input_name)] = self.decoders[input_name](feature_out[input_name])
                    # output[regressor_name+'-{}'.format(input_name)] = self.decoders[regressor_name](feature_out[input_name])
                    output[regressor_name+'-{}'.format(input_name)] = self.decoders[regressor_name](latent[input_name])
            else:
                for regressor_name in self.classifiers.keys():
                    # output[regressor_name+'-{}'.format(input_name)] = self.classifiers[regressor_name](feature_out[input_name], feature)
                    output[regressor_name+'-{}'.format(input_name)] = self.classifiers[regressor_name](latent[input_name], feature)

    
            # output['domain-{}'.format(input_name)] = self.domain_classifier(feature_out[input_name], 1)


        return output, feature_out, mu, logvar
    
    
    
def get_KLD(mu, logvar):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD


# loss function
class VAELoss(nn.Module):
    def __init__(self, training_params):
        super(VAELoss, self).__init__()
        self.output_names = training_params['output_names']
        self.device = training_params['device']

        self.loss_weights  = {}
        self.criterions = {}

        main_task = self.output_names[0]

#         for task in self.tasks:
        for task in training_params['regressor_names']:
            if 'race' in task:
                self.criterions[task] = torch.nn.CrossEntropyLoss()
            elif 'reconstruction' in task:
                self.criterions[task] = torch.nn.MSELoss()
                
            if main_task in task:
                self.loss_weights[task] = training_params['loss_weights']['main_task']
            else:
                N_aux_tasks = len(self.output_names)-1
                if N_aux_tasks==0:
                    self.loss_weights[task] = 0
                else:
                    self.loss_weights[task] = training_params['loss_weights']['auxillary_task']/N_aux_tasks

        # self.criterions['KLD'] = torch.nn.KLDivLoss()
        self.criterions['KLD'] = get_KLD
        self.loss_weights['KLD'] = training_params['loss_weights']['auxillary_task']
        
        # self.modality_dict = training_params['modality_dict']
        
        # i_race = training_params['dataset_dict']['list_label'].index('Race String')
        # self.i_race = i_race
        
    def forward(self, output, label, **kwargs):
        label_dict = {}
        for output_name in output.keys():
            # label_dict[output_name] = label[:, i_race]
            label_dict[output_name] = label[:, [self.output_names.index( output_name.split('-')[0] )]]

            # if 'domain' in output_name:
            #     input_name = output_name.split('-')[1]
            #     label_dict[output_name] = torch.ones(label.size()[0]).to(self.device) * self.modality_dict[input_name]
            # else:
            #     print(output_name)

        # for input_name in self.modality_dict:
        #     output['domain-'+input_name] = torch.ones(label.size()[0]).to(self.device).float() * self.modality_dict[input_name]

#         print(output)

#         print(label_dict)
        
        losses = {}
        for output_name in output.keys():
            # losses[output_name] = self.criterions[ output_name.split('-')[0] ](output[output_name], label_dict[output_name].to(device=self.device, dtype=torch.float).long()).squeeze().float()
            
            if 'race' in output_name:
                losses[output_name] = self.criterions[ output_name.split('-')[0] ](output[output_name], label_dict[output_name].to(device=self.device, dtype=torch.float).long().squeeze()).float()
            elif 'reconstruction' in output_name:
                losses[output_name] = self.criterions[ output_name.split('-')[0] ](output[output_name], label_dict[output_name].to(device=self.device, dtype=torch.float).squeeze()).float()
        
        
        losses['KLD'] = self.criterions['KLD'](kwargs['mu'], kwargs['logvar'])


        list_loss = []
        
        for output_name in list(output.keys()) + ['KLD']:
#             print(output_name)
            print(output_name,  self.loss_weights[ output_name.split('-')[0] ], losses[output_name] )
            l = self.loss_weights[ output_name.split('-')[0] ] * losses[output_name] 
            list_loss.append(l)
#             list_loss.append(self.loss_weights[ output_name.split('-')[0] ] * losses[output_name] for output_name in output.keys())

        losses['total'] = torch.sum(torch.stack(list_loss))

#         losses = {output_name: self.criterions[ output_name ](output[output_name].squeeze(), label_dict[output_name].to(device=self.device, dtype=torch.float).squeeze()) for output_name in output.keys()}

#         losses['total'] = torch.sum(torch.stack([self.loss_weights[ output_name.split('-')[0] ] * losses[output_name] for output_name in output.keys()]))

        return losses

    
    
    
class BinaryClassification(nn.Module):
    def __init__(self, num_classes=10, input_dim=10, feature_dim=0):
        super(BinaryClassification, self).__init__()

        self.bn1 = nn.BatchNorm1d(input_dim+feature_dim)
        self.bn2 = nn.BatchNorm1d(50)
        self.relu = nn.ReLU()

#         self.fc1 = nn.Linear(input_dim, 50+feature_dim)
        self.fc1 = nn.Linear(input_dim+feature_dim, 50)

        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x, feature):  
        
#         if feature is not None:
#             print(x.size(), feature.size())
        x = torch.cat((x, feature), 1)

        out = self.bn1(x)
        out = self.relu(out)
        out = self.fc1(out)
        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.fc2(out)
        
        return out
    
    
class RespiratoryRegression(nn.Module):
    def __init__(self, num_classes=10, input_dim=50, feature_dim=0):
        super(RespiratoryRegression, self).__init__()

        self.bn1 = nn.BatchNorm1d(input_dim+feature_dim)
        self.bn2 = nn.BatchNorm1d(50)
        self.relu = nn.ReLU()

#         self.fc1 = nn.Linear(input_dim, 50+feature_dim)
        self.fc1 = nn.Linear(input_dim+feature_dim, 50)

        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x, feature):  
        
#         if feature is not None:
#             print(x.size(), feature.size())
        x = torch.cat((x, feature), 1)
        
#         print(x.size(), feature.size())

        out = self.bn1(x)
        out = self.relu(out)
        out = self.fc1(out)
        
        out = self.bn2(out)
        out = self.relu(out)
#         out = torch.cat((out, feature), 1)
        out = self.fc2(out)
        
#         out = self.relu(out)

        return out
    
class DominantFreqRegression(nn.Module):
    def __init__(self, training_params, num_classes=10, input_dim=50, feature_dim=0):
        super(DominantFreqRegression, self).__init__()

    
#         self.xf = training_params['xf']
#         self.xf_masked = training_params['xf_masked']
        xf_masked = torch.from_numpy(training_params['xf_masked'])
        self.xf_masked = xf_masked.float()
        
        self.dominantFreq_detect = training_params['dominantFreq_detect']
#         self.xf_masked = xf_masked.to(device=training_params['device'], dtype=torch.float)
        
#         self.bn1 = nn.BatchNorm1d(input_dim+feature_dim)
#         self.bn2 = nn.BatchNorm1d(50)
#         self.relu = nn.ReLU()

# #         self.fc1 = nn.Linear(input_dim, 50+feature_dim)
#         self.fc1 = nn.Linear(input_dim+feature_dim, 50)

#         self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x, feature):  
        
#         x = torch.from_numpy(x)
#         xf_masked = torch.from_numpy(self.xf_masked)
#         print(x.get_device(), self.xf_masked.get_device())
#         print(x, self.xf_masked)
        

        if self.dominantFreq_detect=='argmax':
#             print('x', x)
#             print('x size', x.size())
            index_dominant = torch.argmax(x,axis=1).squeeze()
#             print('index_dominant', index_dominant.size())
#             print('self.xf_masked', self.xf_masked.size())
            xf_repeated = torch.tile(self.xf_masked.to(x.device), (x.shape[0], 1)).T
#             print('xf_repeated', xf_repeated.size())

            out = xf_repeated[index_dominant,  range(xf_repeated.shape[1])][:,None]
#             label_mapped = xf_repeated[ indices, range(xf_repeated.shape[1])]

    
        elif self.dominantFreq_detect=='expectation':
            x_normed = x / torch.sum(x, axis=1)[:,None]
            out = torch.sum(x_normed * self.xf_masked.to(x.device), axis=1)[:,None]
#         print(out.size())

        return out
#         print(x.size(),  self.xf_masked.size())
#         sys.exit()
# #         print(x.size(),  self.xf_masked)
        
#         # dim of x: 
        
# #         if feature is not None:
# #             print(x.size(), feature.size())
#         x = torch.cat((x, feature), 1)
        
# #         print(x.size(), feature.size())

#         out = self.bn1(x)
#         out = self.relu(out)
#         out = self.fc1(out)
        
#         out = self.bn2(out)
#         out = self.relu(out)
# #         out = torch.cat((out, feature), 1)
#         out = self.fc2(out)
        
#         out = self.relu(out)

#         return out
    
# GRL
class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)


# domain classifier neural network (fc layers)
class DomainClassifier(nn.Module):
    def __init__(self, num_classes=10, input_dim=50, hidden_dim=50, p_dropout=0.5):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.drop = nn.Dropout(p=p_dropout)
        self.relu = nn.ReLU()
        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        #       print('DomainClassifier_total_params:', pytorch_total_params)

    def forward(self, x, constant):
        out1 = GradReverse.grad_reverse(x.float(), constant)
        out1 = self.relu(self.drop(self.fc1(out1)))
        out2 = self.fc2(out1)
        return out2

    
    

    
# loss function
class CompressionLoss(nn.Module):
    def __init__(self, training_params):
        super(CompressionLoss, self).__init__()
        self.output_names = training_params['output_names']
        self.device = training_params['device']

        self.loss_weights  = {}
        self.criterions = {}

        main_task = self.output_names[0]

#         for task in self.tasks:
        for task in training_params['regressor_names']:
            if 'race' in task:
                self.criterions[task] = torch.nn.CrossEntropyLoss()
            elif 'reconstruction' in task:
                self.criterions[task] = torch.nn.MSELoss()
                
            if main_task in task:
                self.loss_weights[task] = training_params['loss_weights']['main_task']
            else:
                N_aux_tasks = len(self.output_names)-1
                if N_aux_tasks==0:
                    self.loss_weights[task] = 0
                else:
                    self.loss_weights[task] = training_params['loss_weights']['auxillary_task']/N_aux_tasks

        # self.modality_dict = training_params['modality_dict']
        
        # i_race = training_params['dataset_dict']['list_label'].index('Race String')
        # self.i_race = i_race
        
    def forward(self, output, label):
        label_dict = {}
        for output_name in output.keys():
            # label_dict[output_name] = label[:, i_race]
            label_dict[output_name] = label[:, [self.output_names.index( output_name.split('-')[0] )]]

            # if 'domain' in output_name:
            #     input_name = output_name.split('-')[1]
            #     label_dict[output_name] = torch.ones(label.size()[0]).to(self.device) * self.modality_dict[input_name]
            # else:
            #     print(output_name)

        # for input_name in self.modality_dict:
        #     output['domain-'+input_name] = torch.ones(label.size()[0]).to(self.device).float() * self.modality_dict[input_name]

#         print(output)

#         print(label_dict)
        
        losses = {}
        for output_name in output.keys():
            # losses[output_name] = self.criterions[ output_name.split('-')[0] ](output[output_name], label_dict[output_name].to(device=self.device, dtype=torch.float).long()).squeeze().float()
            
            if 'race' in output_name:
                losses[output_name] = self.criterions[ output_name.split('-')[0] ](output[output_name], label_dict[output_name].to(device=self.device, dtype=torch.float).long().squeeze()).float()
            else:
                losses[output_name] = self.criterions[ output_name.split('-')[0] ](output[output_name], label_dict[output_name].to(device=self.device, dtype=torch.float).squeeze()).float()
                


        list_loss = []
        
        for output_name in output.keys():
#             print(output_name)
            
            l = self.loss_weights[ output_name.split('-')[0] ] * losses[output_name] 
            list_loss.append(l)
#             list_loss.append(self.loss_weights[ output_name.split('-')[0] ] * losses[output_name] for output_name in output.keys())

        losses['total'] = torch.sum(torch.stack(list_loss))

#         losses = {output_name: self.criterions[ output_name ](output[output_name].squeeze(), label_dict[output_name].to(device=self.device, dtype=torch.float).squeeze()) for output_name in output.keys()}

#         losses['total'] = torch.sum(torch.stack([self.loss_weights[ output_name.split('-')[0] ] * losses[output_name] for output_name in output.keys()]))

        return losses

    
    
# loss function
class AdversarialLoss(nn.Module):
    def __init__(self, training_params):
        super(AdversarialLoss, self).__init__()
        self.output_names = training_params['output_names']
        self.device = training_params['device']

        self.loss_weights  = {}
        self.criterions = {}

        main_task = self.output_names[0]

#         for task in self.tasks:
        for task in training_params['regressor_names']:
            self.criterions[task] = torch.nn.MSELoss()
            if main_task in task:
                self.loss_weights[task] = training_params['loss_weights']['main_task']
            else:
                
                N_aux_tasks = len(self.output_names)-1
                if N_aux_tasks==0:
                    self.loss_weights[task] = 0
                else:
                    self.loss_weights[task] = training_params['loss_weights']['auxillary_task']/N_aux_tasks
            
            
        task = 'domain'
        self.criterions[task] = torch.nn.CrossEntropyLoss()
        self.loss_weights[task] = 1




        self.modality_dict = training_params['modality_dict']
        
    def forward(self, output, label):
        

#         label = {output_name: label[:, [self.output_names.index( output_name.split('-')[0] )]] for output_name in output.keys()}

        label_dict = {}
        for output_name in output.keys():
            if 'domain' in output_name:
                input_name = output_name.split('-')[1]
                label_dict[output_name] = torch.ones(label.size()[0]).to(self.device) * self.modality_dict[input_name]
            else:
#                 print(output_name)
                label_dict[output_name] = label[:, [self.output_names.index( output_name.split('-')[0] )]]

#         for input_name in self.modality_dict:
#             output['domain-'+input_name] = torch.ones(label.size()[0]).to(self.device).float() * self.modality_dict[input_name]

#         print(output)

#         print(label_dict)
        
        losses = {}
        for output_name in output.keys():
#             print(output_name)
#             print(output[output_name].squeeze().size(), label_dict[output_name].to(device=self.device, dtype=torch.float).squeeze().size())
#             losses[output_name] = self.criterions[ output_name.split('-')[0] ](output[output_name].squeeze(), label_dict[output_name].to(device=self.device, dtype=torch.float).squeeze())

            if 'domain' in output_name:
                losses[output_name] = self.criterions[ output_name.split('-')[0] ](output[output_name], label_dict[output_name].to(device=self.device, dtype=torch.float).long()).squeeze().float()
            else:
                losses[output_name] = self.criterions[ output_name.split('-')[0] ](output[output_name], label_dict[output_name].to(device=self.device, dtype=torch.float)).squeeze().float()
    
        
    
        list_loss = []
        
        for output_name in output.keys():
#             print(output_name)
            
            l = self.loss_weights[ output_name.split('-')[0] ] * losses[output_name] 
            list_loss.append(l)
#             list_loss.append(self.loss_weights[ output_name.split('-')[0] ] * losses[output_name] for output_name in output.keys())

        losses['total'] = torch.sum(torch.stack(list_loss))

#         losses = {output_name: self.criterions[ output_name ](output[output_name].squeeze(), label_dict[output_name].to(device=self.device, dtype=torch.float).squeeze()) for output_name in output.keys()}

#         losses['total'] = torch.sum(torch.stack([self.loss_weights[ output_name.split('-')[0] ] * losses[output_name] for output_name in output.keys()]))

        return losses
