from argparse import ArgumentParser

import torch
import torchmetrics
import pytorch_lightning as pl

from .model_selection import load_model
from .scheduler import WarmupCosineSchedule

class LitVisionTransformer(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

        self.backbone = load_model(args)
        
    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.backbone(x)
        
        return embedding

    def training_step(self, batch, batch_idx):
        # forward and backward pass and log
        x, y = batch
        
        y_hat = self.backbone(x)
        
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_epoch=True, on_step=True)

        self.train_acc(y_hat.softmax(dim=1), y)
        self.log('train_acc', self.train_acc, on_epoch=True, on_step=False)
                
        curr_lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', curr_lr, on_epoch=False, on_step=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        y_hat = self.backbone(x)
        
        loss = self.criterion(y_hat, y)
        self.val_acc(y_hat.softmax(dim=-1), y)
        
        metrics = {'val_acc': self.val_acc, 'val_loss': loss}
        self.log_dict(metrics, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        x, y = batch
        
        y_hat = self.backbone(x)
        
        self.test_acc(y_hat.softmax(dim=-1), y)
        
        self.log('test_acc', self.test_acc, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        if self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), 
            lr=self.args.learning_rate, weight_decay=self.args.weight_decay)  
        else: 
            optimizer = torch.optim.SGD(self.parameters(), lr=self.args.learning_rate, 
            momentum=0.9, weight_decay=self.args.weight_decay)
        
        scheduler = {'scheduler': WarmupCosineSchedule(
        optimizer, warmup_steps=self.args.warmup_steps, 
        t_total=self.args.total_steps),
        'name': 'learning_rate', 'interval': 'step', 'frequency': 1}
        
        return [optimizer], [scheduler]
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--optimizer', choices=['sgd', 'adam'], default='sgd')
        parser.add_argument('--learning_rate', default=0.001, type=float,
                            help='Initial learning rate.')  
        parser.add_argument('--weight_decay', type=float, default=0.00)
        parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps for LR scheduler.')
        
        parser.add_argument('--model_name',
                        choices=['Ti_4', 'Ti_8', 'Ti_16', 'Ti_32', 'S_4', 'S_8', 'S_16', 'S_32',
                                 'B_4', 'B_8', 'B_16', 'B_32', 'L_16', 'L_32', 'B_16_in1k'],
                        default='B_16_in1k', help='Which model architecture to use')
        parser.add_argument('--pretrained',action='store_true',
                            help='Loads pretrained model if available')
        parser.add_argument('--checkpoint_path', type=str, default=None)     
        parser.add_argument('--transfer_learning', action='store_true',
                            help='Load partial state dict for transfer learning'
                            'Resets the [embeddings, logits and] fc layer for ViT')    
        parser.add_argument('--load_partial_mode', choices=['full_tokenizer', 'patchprojection', 
                            'posembeddings', 'clstoken', 'patchandposembeddings', 
                            'patchandclstoken', 'posembeddingsandclstoken', None], default=None,
                            help='Load pre-processing components to speed up training')

        parser.add_argument('--interm_features_fc', action='store_true', 
                        help='If use this flag creates FC using intermediate features instead of only last layer.')
        parser.add_argument('--conv_patching', action='store_true', 
                        help='If use this flag uses a small convolutional stem instead of single large-stride convolution for patch projection.')
        
        return parser

