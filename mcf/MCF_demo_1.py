# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 16:42:44 2024

@author: Achun
"""

 
import torch 
from torch import nn
import torch.nn.functional as F
from torchvision import models


class HSI_CNN(nn.Module):

    def __init__(self,in_channels):
        super().__init__()
        
        self._model = models.mobilenet_v3_small(pretrained=True)
        _tmp1 = self._model.features[0][0]
        _tmp2 = self._model.features[1].block[0][0]

        use_bias = (_tmp1.bias != None)
        self._model.features[0][0] = nn.Conv2d(in_channels, out_channels=_tmp1.out_channels,
            kernel_size= 1, stride= 1, padding= 0, bias=use_bias)
        self._model.features[1].block[0][0] = nn.Conv2d(in_channels = _tmp2.in_channels, out_channels=_tmp2.out_channels,
            kernel_size= _tmp2.kernel_size, stride= 1, padding= _tmp2.padding, groups = _tmp2.groups, bias=use_bias)
            
        torch.cuda.empty_cache()

        del _tmp1 , _tmp2
        

class Lidar_CNN(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self._model = models.mobilenet_v3_small()

        _tmp1 = self._model.features[0][0]
        _tmp2 = self._model.features[1].block[0][0]


        use_bias = (_tmp1.bias != None)
        self._model.features[0][0] = nn.Conv2d(in_channels, out_channels=_tmp1.out_channels,
            kernel_size= 1, stride= 1, padding= 0, bias=use_bias)
        self._model.features[1].block[0][0] = nn.Conv2d(in_channels = _tmp2.in_channels, out_channels=_tmp2.out_channels,
            kernel_size= _tmp2.kernel_size, stride= 1, padding= _tmp2.padding, groups = _tmp2.groups, bias=use_bias)


        torch.cuda.empty_cache()

        del _tmp1 , _tmp2



class SelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, dim_head, attn_pdrop, resid_pdrop):
        super().__init__()
        inner_dim = dim_head * n_head
        project_out = not (n_head == 1 and dim_head == n_embd)
        self.n_head = n_head
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, inner_dim,bias=False)
        self.query = nn.Linear(n_embd, inner_dim,bias=False)
        self.value = nn.Linear(n_embd, inner_dim,bias=False)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Sequential(nn.Linear(inner_dim, n_embd),
                                  self.resid_drop
        ) if project_out else nn.Identity()


    
    def forward(self, x):
        B, T, C = x.size()

        k = self.key(x).view(B, T, self.n_head, self.dim_head).transpose(1, 2) 
        q = self.query(x).view(B, T, self.n_head, self.dim_head).transpose(1, 2) 
        v = self.value(x).view(B, T, self.n_head, self.dim_head).transpose(1, 2) 


        att = (q @ k.transpose(-2, -1)) * self.scale
        #with mask att
        m_r = torch.ones_like(att) * 0.1
        att = att + torch.bernoulli(m_r) * -1e12
        
        att = F.softmax(att, dim=-1)

        y = att @ v 
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head*self.dim_head) 

        # output projection
        y = self.proj(y)
        return y    
    


class Block(nn.Module):

    def __init__(self, n_embd, n_head, dim_head, block_exp, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, dim_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True), # changed from GELU
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x




class MVT(nn.Module):


    def __init__(self, n_embd, n_head, dim_head, block_exp, n_layer, 
                    block_w, block_h, embd_pdrop, attn_pdrop, resid_pdrop):
        super().__init__()
        self.n_embd = n_embd
        self.block_w = block_w
        self.block_h = block_h
        
        self.inner_proj = nn.Linear(block_w * block_h, n_embd)
        
        # positional embedding parameter (learnable), image + lidar
        self.pos_emb = nn.Parameter(torch.zeros(1, (1 + 1) * block_w * block_h + n_embd, n_embd))
        

        self.drop = nn.Dropout(embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, dim_head,
                        block_exp, attn_pdrop, resid_pdrop)
                        for layer in range(n_layer)])
        

        self.ln_f = nn.LayerNorm(n_embd)

        self.apply(self._init_weights)



    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            # nn.init.constant_(m.bias, 0)
            # module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.BatchNorm2d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        no_decay.add('pos_emb')

        # create the pytorch optimizer object
        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        return optim_groups

    def forward(self, image_tensor, lidar_tensor):

        
        bz = lidar_tensor.shape[0]
        # h, w = lidar_tensor.shape[2:4]
        breakpoint()
        h = self.block_h    # 5
        w = self.block_w    # 5
        
        image_tensor = image_tensor.view(bz, -1, h, w)
        lidar_tensor = lidar_tensor.view(bz, -1, h, w)
        # pixel embd
        image_1 = image_tensor.view(bz, -1, h*w)
        #spectral embd
        token_embed_1 = image_1.permute(0,2,1).contiguous()
        
        token_embed_2 = self.inner_proj(image_1)
        # token_embed_2 = self.inner_proj(token_embed_1)
        #lidar embd
        lidar_1 = lidar_tensor.view(bz, -1, h*w)
        token_embed_3 = lidar_1.permute(0,2,1).contiguous()
        
        token_embeddings = torch.cat([token_embed_1, token_embed_2,token_embed_3], dim=1)
        
        x = self.drop(self.pos_emb + token_embeddings) 

        x = self.blocks(x)
        x = self.ln_f(x)

        x = x.permute(0,2,1).contiguous()

        image_tensor_out = x[:, :, :h*w].contiguous().view(bz , -1, h, w)
        lidar_tensor_out = x[:, :, h*w+self.n_embd:].contiguous().view(bz , -1, h, w)
        
        return image_tensor_out, lidar_tensor_out



class MCF(nn.Module):


    def __init__(self, HSIband, lidarband, num_classes):
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.avgpool_2 = nn.AdaptiveAvgPool2d((5, 5))
        
        self.image_encoder = HSI_CNN(HSIband)
        self.lidar_encoder = Lidar_CNN(lidarband)

        self.mlp_head = nn.Linear(24, num_classes)
        torch.nn.init.xavier_uniform_(self.mlp_head.weight)
        torch.nn.init.normal_(self.mlp_head.bias, std=1e-6)  

        

        self.transformer1 = MVT(n_embd=16,
                            n_head=4, 
                            dim_head = 16,
                            block_exp=4, 
                            n_layer=2, 
                            block_w=5, 
                            block_h=5, 
                            embd_pdrop=0.1, 
                            attn_pdrop=0.1, 
                            resid_pdrop=0.1)
        
        self.transformer2 = MVT(n_embd=24,
                            n_head=4, 
                            dim_head = 24,
                            block_exp=4, 
                            n_layer=2, 
                            block_w=5, 
                            block_h=5, 
                            embd_pdrop=0.1, 
                            attn_pdrop=0.1, 
                            resid_pdrop=0.1)

        
    def forward(self, hsi, lidar):

        bz, _, h, w = hsi.shape
        img_channel = hsi.shape[1]
        lidar_channel = lidar.shape[1]

        image_tensor = hsi.view(bz, img_channel, h, w)
        lidar_tensor = lidar.view(bz, lidar_channel, h, w)

        image_features = self.image_encoder._model.features[0](image_tensor)
        image_features1 = self.image_encoder._model.features[1](image_features)

        lidar_features = self.lidar_encoder._model.features[0](lidar_tensor)
        lidar_features1 = self.lidar_encoder._model.features[1](lidar_features)

        # fusion at (B, 24, 11, 11)
        image_features2 = self.avgpool(image_features1)
        lidar_features2 = self.avgpool(lidar_features1)
        image_features_layer1, lidar_features_layer1 = self.transformer1(image_features2, lidar_features2)
        image_features_layer1 = F.interpolate(image_features_layer1, size=[h, w], mode='bilinear')
        lidar_features_layer1 = F.interpolate(lidar_features_layer1, size=[h, w], mode='bilinear')
        
        image_features = image_features1 + image_features_layer1
        lidar_features = lidar_features1 + lidar_features_layer1


        image_features2 = self.image_encoder._model.features[2](image_features)
        lidar_features2 = self.lidar_encoder._model.features[2](lidar_features)
        bz, _, h2, w2 = image_features2.shape
        # fusion at (B, 24, 6, 6)
        image_features3 = self.avgpool_2(image_features2)
        lidar_features3 = self.avgpool_2(lidar_features2)        
        
        image_features_layer2, lidar_features_layer2 = self.transformer2(image_features3, lidar_features3)
#         image_features_layer2 = F.interpolate(image_features_layer2, size=[h2, w2], mode='bilinear')
#         lidar_features_layer2 = F.interpolate(lidar_features_layer2, size=[h2, w2], mode='bilinear')
        
        image_features = image_features3 + image_features_layer2
        lidar_features = lidar_features3 + lidar_features_layer2


        image_features = self.image_encoder._model.avgpool(image_features)
        image_features = torch.flatten(image_features, 1)
        image_features = image_features.view(bz, 1, -1)
        lidar_features = self.lidar_encoder._model.avgpool(lidar_features)
        lidar_features = torch.flatten(lidar_features, 1)
        lidar_features = lidar_features.view(bz, 1, -1)

        fused_features = torch.cat([image_features, lidar_features], dim=1)
        fused_features = torch.sum(fused_features, dim=1)
        fused_features = fused_features.view(bz,-1)
        pred = self.mlp_head(fused_features)
           
        return pred




