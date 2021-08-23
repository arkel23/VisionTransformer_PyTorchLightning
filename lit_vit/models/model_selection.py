import torch 
import torch.nn as nn

import einops
from einops.layers.torch import Rearrange

from pytorch_pretrained_vit import ViT, ViTConfigExtended, PRETRAINED_CONFIGS

def load_model(args):
    # initiates model and loss     
    model = VisionTransformer(args)
    
    if args.checkpoint_path:
        if args.load_partial_mode:
            model.model.load_partial(weights_path=args.checkpoint_path, 
                pretrained_image_size=self.configuration.pretrained_image_size, 
                pretrained_mode=args.load_partial_mode, verbose=True)
        else:
            state_dict = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
            expected_missing_keys = []

            load_patch_embedding = (
                (self.configuration.num_channels==self.configuration.pretrained_num_channels) and
                (not(args.conv_patching))
            )
            load_fc_layer = (
                (self.configuration.num_classes==self.configuration.pretrained_num_classes) and
                (not(args.transfer_learning)) and 
                (not(args.interm_features_fc))
            )
            
            if ('patch_embedding.weight' in state_dict and load_patch_embedding):
                expected_missing_keys += ['model.patch_embedding.weight', 'model.patch_embedding.bias']
            
            if ('pre_logits.weight' in state_dict and self.configuration.load_repr_layer==False):
                expected_missing_keys += ['model.pre_logits.weight', 'model.pre_logits.bias']
                    
            if ('model.fc.weight' in state_dict and load_fc_layer):
                expected_missing_keys += ['model.fc.weight', 'model.fc.bias']
            
            for key in expected_missing_keys:
                state_dict.pop(key)
                        
            ret = model.load_state_dict(state_dict, strict=False)
            print('''Missing keys when loading pretrained weights: {}
                Expected missing keys: {}'''.format(ret.missing_keys, expected_missing_keys))
            print('Unexpected keys when loading pretrained weights: {}'.format(ret.unexpected_keys))
            
            print('Loaded from custom checkpoint.')

    return model


class VisionTransformer(nn.Module):
    def __init__(self, args):
        super(VisionTransformer, self).__init__()
        
        def_config = PRETRAINED_CONFIGS['{}'.format(args.model_name)]['config']
        self.configuration = ViTConfigExtended(**def_config)
        self.configuration.num_classes = args.num_classes
        self.configuration.image_size = args.image_size
        
        self.model = ViT(self.configuration, name=args.model_name, 
            pretrained=args.pretrained, load_fc_layer=not(args.interm_features_fc),
            ret_interm_repr=args.interm_features_fc, conv_patching=args.conv_patching)

        if args.interm_features_fc:
            self.inter_class_head = nn.Sequential(
                            nn.Linear(self.configuration.num_hidden_layers, 1),
                            Rearrange(' b d 1 -> b d'),
                            nn.ReLU(),
                            nn.LayerNorm(self.configuration.hidden_size, eps=self.configuration.layer_norm_eps),
                            nn.Linear(self.configuration.hidden_size, self.configuration.num_classes)
                        )
    
    def forward(self, images, mask=None):
        if hasattr(self, 'inter_class_head'):
            features, interm_features = self.model(images, mask)
        else:
            logits = self.model(images, mask)

        if hasattr(self, 'inter_class_head'):
            interm_features = torch.stack(interm_features, dim=-1)
            logits = self.inter_class_head(interm_features[:, 0])
        return logits
