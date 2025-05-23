
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load
from timm import create_model
from transformers import CLIPProcessor, CLIPModel
import clip

# =============================================================================
# Backbone architectures used in the SProd paper experiments.
# =============================================================================


class ResNet18(nn.Module):
    def __init__(self, freeze_pretrained=True, num_classes=2):
        super(ResNet18, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', f'resnet18', pretrained=True)
        if freeze_pretrained:
            self.freeze_pretrained()
        d = self.model.fc.in_features
        self.model.fc = nn.Linear(d, num_classes)

    def freeze_pretrained(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x, return_feature=False, return_feature_list=False):
        features = self.model.conv1(x)
        features = self.model.bn1(features)
        features = self.model.relu(features)
        features = self.model.maxpool(features)

        features1 = self.model.layer1(features)
        features2 = self.model.layer2(features1)
        features3 = self.model.layer3(features2)
        features4 = self.model.layer4(features3)

        global_avg_pool = F.adaptive_avg_pool2d(features4, (1, 1))
        global_avg_pool = global_avg_pool.view(global_avg_pool.size(0), -1)

        output = self.model.fc(global_avg_pool)

        if return_feature:
            return output, global_avg_pool
        elif return_feature_list:
            return output, [features1, features2, features3, features4, global_avg_pool]
        else:
            return output

    def forward_threshold(self, x, threshold):
        output = self.forward(x)
        output = F.threshold(output, threshold, 0)
        return output

    def intermediate_forward(self, x, layer_index):
        out = self.model.relu(self.model.bn1(self.model.conv1(x)))
        out = self.model.maxpool(out)

        out = self.model.layer1(out)
        if layer_index == 1:
            return out

        out = self.model.layer2(out)
        if layer_index == 2:
            return out

        out = self.model.layer3(out)
        if layer_index == 3:
            return out

        out = self.model.layer4(out)
        if layer_index == 4:
            return out

        raise ValueError("Invalid layer_index. Supported values are 1, 2, 3, and 4.")

    def get_fc(self):
        fc = self.model.fc
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.model.fc
    
    def load_trained_fc(self, fc_state_dict_path):
        trained_fc_state_dict = torch.load(fc_state_dict_path)
        
        new_state_dict = {}
        for k, v in trained_fc_state_dict.items():
            new_key = k.replace('linearhead.', '')
            new_state_dict[new_key] = v

        self.model.fc.load_state_dict(new_state_dict)

class ResNet34(nn.Module):
    def __init__(self, freeze_pretrained=True, num_classes=2):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
        if freeze_pretrained:
            self.freeze_pretrained()
        d = self.model.fc.in_features
        self.model.fc = nn.Linear(d, num_classes)

    def freeze_pretrained(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x, return_feature=False, return_feature_list=False):
        features = self.model.conv1(x)
        features = self.model.bn1(features)
        features = self.model.relu(features)
        features = self.model.maxpool(features)

        features1 = self.model.layer1(features)
        features2 = self.model.layer2(features1)
        features3 = self.model.layer3(features2)
        features4 = self.model.layer4(features3)

        global_avg_pool = self.model.avgpool(features4)
        global_avg_pool = global_avg_pool.view(global_avg_pool.size(0), -1)

        output = self.model.fc(global_avg_pool)

        if return_feature:
            return output, global_avg_pool
        elif return_feature_list:
            return output, [features1, features2, features3, features4, global_avg_pool]
        else:
            return output

    def forward_threshold(self, x, threshold):
        features = self.model.conv1(x)
        features = self.model.bn1(features)
        features = self.model.relu(features)
        features = self.model.maxpool(features)

        features1 = self.model.layer1(features)
        features2 = self.model.layer2(features1)
        features3 = self.model.layer3(features2)
        features4 = self.model.layer4(features3)

        global_avg_pool = self.model.avgpool(features4)
        global_avg_pool = global_avg_pool.view(global_avg_pool.size(0), -1)

        output = self.model.fc(global_avg_pool)

        output = F.threshold(output, threshold, 0)

        return output

    def intermediate_forward(self, x, layer_index):
        if layer_index == 1:
            return self.model.layer1(x)
        elif layer_index == 2:
            return self.model.layer2(self.model.layer1(x))
        elif layer_index == 3:
            return self.model.layer3(self.model.layer2(self.model.layer1(x)))
        elif layer_index == 4:
            return self.model.layer4(self.model.layer3(self.model.layer2(self.model.layer1(x))))
        else:
            raise ValueError("Invalid layer_index. Supported values are 1, 2, 3, and 4.")

    def get_fc(self):
        fc = self.model.fc
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.model.fc

    def load_trained_fc(self, fc_state_dict_path):
        trained_fc_state_dict = torch.load(fc_state_dict_path)
        
        new_state_dict = {}
        for k, v in trained_fc_state_dict.items():
            new_key = k.replace('linearhead.', '')
            new_state_dict[new_key] = v

        self.model.fc.load_state_dict(new_state_dict)


class ResNet50(nn.Module):
    def __init__(self, freeze_pretrained=True,num_classes=2):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', f'resnet50', pretrained=True)
        if freeze_pretrained:
            self.freeze_pretrained()
        d = self.model.fc.in_features
        self.model.fc = nn.Linear(d, num_classes)

    def freeze_pretrained(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x, return_feature=False, return_feature_list=False):
        features = self.model.conv1(x)
        features = self.model.bn1(features)
        features = self.model.relu(features)
        features = self.model.maxpool(features)

        features1 = self.model.layer1(features)
        features2 = self.model.layer2(features1)
        features3 = self.model.layer3(features2)
        features4 = self.model.layer4(features3)

        global_avg_pool = self.model.avgpool(features4)
        global_avg_pool = global_avg_pool.view(global_avg_pool.size(0), -1)

        output = self.model.fc(global_avg_pool)

        if return_feature:
            return output, global_avg_pool
        elif return_feature_list:
            return output, [features1, features2, features3, features4, global_avg_pool]
        else:
            return output

    def forward_threshold(self, x, threshold):
        features = self.model.conv1(x)
        features = self.model.bn1(features)
        features = self.model.relu(features)
        features = self.model.maxpool(features)

        features1 = self.model.layer1(features)
        features2 = self.model.layer2(features1)
        features3 = self.model.layer3(features2)
        features4 = self.model.layer4(features3)

        global_avg_pool = self.model.avgpool(features4)
        global_avg_pool = global_avg_pool.view(global_avg_pool.size(0), -1)

        output = self.model.fc(global_avg_pool)

        output = F.threshold(output, threshold, 0)

        return output

    def intermediate_forward(self, x, layer_index):
        if layer_index == 1:
            return self.model.layer1(x)
        elif layer_index == 2:
            return self.model.layer2(self.model.layer1(x))
        elif layer_index == 3:
            return self.model.layer3(self.model.layer2(self.model.layer1(x)))
        elif layer_index == 4:
            return self.model.layer4(self.model.layer3(self.model.layer2(self.model.layer1(x))))
        else:
            raise ValueError("Invalid layer_index. Supported values are 1, 2, 3, and 4.")

    def get_fc(self):
        fc = self.model.fc
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.model.fc
    
    def load_trained_fc(self, fc_state_dict_path):
        trained_fc_state_dict = torch.load(fc_state_dict_path)
        
        new_state_dict = {}
        for k, v in trained_fc_state_dict.items():
            new_key = k.replace('linearhead.', '')
            new_state_dict[new_key] = v

        self.model.fc.load_state_dict(new_state_dict)


class ResNet101(nn.Module):
    def __init__(self, freeze_pretrained=True, num_classes=2):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', f'resnet101', pretrained=True)
        if freeze_pretrained:
            self.freeze_pretrained()
        d = self.model.fc.in_features
        self.model.fc = nn.Linear(d, num_classes)

    def freeze_pretrained(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x, return_feature=False, return_feature_list=False):
        features = self.model.conv1(x)
        features = self.model.bn1(features)
        features = self.model.relu(features)
        features = self.model.maxpool(features)

        features1 = self.model.layer1(features)
        features2 = self.model.layer2(features1)
        features3 = self.model.layer3(features2)
        features4 = self.model.layer4(features3)

        global_avg_pool = self.model.avgpool(features4)
        global_avg_pool = global_avg_pool.view(global_avg_pool.size(0), -1)

        output = self.model.fc(global_avg_pool)

        if return_feature:
            return output, global_avg_pool
        elif return_feature_list:
            return output, [features1, features2, features3, features4, global_avg_pool]
        else:
            return output

    def forward_threshold(self, x, threshold):
        features = self.model.conv1(x)
        features = self.model.bn1(features)
        features = self.model.relu(features)
        features = self.model.maxpool(features)

        features1 = self.model.layer1(features)
        features2 = self.model.layer2(features1)
        features3 = self.model.layer3(features2)
        features4 = self.model.layer4(features3)

        global_avg_pool = self.model.avgpool(features4)
        global_avg_pool = global_avg_pool.view(global_avg_pool.size(0), -1)

        output = self.model.fc(global_avg_pool)

        output = F.threshold(output, threshold, 0)

        return output

    def intermediate_forward(self, x, layer_index):
        if layer_index == 1:
            return self.model.layer1(x)
        elif layer_index == 2:
            return self.model.layer2(self.model.layer1(x))
        elif layer_index == 3:
            return self.model.layer3(self.model.layer2(self.model.layer1(x)))
        elif layer_index == 4:
            return self.model.layer4(self.model.layer3(self.model.layer2(self.model.layer1(x))))
        else:
            raise ValueError("Invalid layer_index. Supported values are 1, 2, 3, and 4.")

    def get_fc(self):
        fc = self.model.fc
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.model.fc

    def load_trained_fc(self, fc_state_dict_path):
        trained_fc_state_dict = torch.load(fc_state_dict_path)
        
        new_state_dict = {}
        for k, v in trained_fc_state_dict.items():
            new_key = k.replace('linearhead.', '')
            new_state_dict[new_key] = v

        self.model.fc.load_state_dict(new_state_dict)


class BiT_M_R50x1(nn.Module):
    def __init__(self, num_classes=2, freeze_pretrained=True):
        super().__init__()
        self.model = create_model('resnetv2_50x1_bitm', pretrained=True)
        
        if freeze_pretrained:
            self.freeze_pretrained()

        d = self.model.head.in_features
        self.model.head = nn.Linear(d, num_classes)

    def freeze_pretrained(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x, return_feature=False, return_feature_list=False):
        x = self.model.stem(x)  
        x = self.model.stages(x)  
        features = self.model.norm(x)  
        features = torch.mean(features, dim=[2, 3])
        output = self.model.head(features)  

        if return_feature:
            return output, features
        else:
            return output

    def forward_threshold(self, x, threshold):
        output = self.forward(x)
        output = F.threshold(output, threshold, 0)
        return output


    def get_fc(self):
        fc = self.model.head  
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.model.head  

    def load_trained_fc(self, fc_state_dict_path):
        trained_fc_state_dict = torch.load(fc_state_dict_path)
        
        new_state_dict = {}
        for k, v in trained_fc_state_dict.items():
            new_key = k.replace('linearhead.', '')  
            new_state_dict[new_key] = v

        self.model.head.load_state_dict(new_state_dict) 


class BiT_M_R50x3(nn.Module):
    def __init__(self, num_classes=2, freeze_pretrained=True):
        super().__init__()
        self.model = create_model('resnetv2_50x3_bitm', pretrained=True)
        
        if freeze_pretrained:
            self.freeze_pretrained()

        d = self.model.head.in_features
        self.model.head = nn.Linear(d, num_classes)

    def freeze_pretrained(self):
        for param in self.model.parameters():
            param.requires_grad = False


    def forward(self, x, return_feature=False, return_feature_list=False):
        
        x = self.model.stem(x)  
        x = self.model.stages(x)  
        features = self.model.norm(x)  
        features = torch.mean(features, dim=[2, 3])
        output = self.model.head(features)  
        if return_feature:
            return output, features

        else:
            return output

    def forward_threshold(self, x, threshold):
        output = self.forward(x)
        output = F.threshold(output, threshold, 0)
        return output

    def get_fc(self):
        fc = self.model.head  
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.model.head  

    def load_trained_fc(self, fc_state_dict_path):
        trained_fc_state_dict = torch.load(fc_state_dict_path)
        
        new_state_dict = {}
        for k, v in trained_fc_state_dict.items():
            new_key = k.replace('linearhead.', '')  
            new_state_dict[new_key] = v

        self.model.head.load_state_dict(new_state_dict) 


class BiT_M_R101x1(nn.Module):
    def __init__(self, num_classes=2, freeze_pretrained=True):
        super().__init__()
        self.model = create_model('resnetv2_101x1_bitm', pretrained=True)
        
        if freeze_pretrained:
            self.freeze_pretrained()

        d = self.model.head.in_features
        self.model.head = nn.Linear(d, num_classes)

    def freeze_pretrained(self):
        for param in self.model.parameters():
            param.requires_grad = False


    def forward(self, x, return_feature=False, return_feature_list=False):
        
        x = self.model.stem(x)  
        x = self.model.stages(x)  
        features = self.model.norm(x)  
        features = torch.mean(features, dim=[2, 3])
        output = self.model.head(features)  
        if return_feature:
            return output, features

        else:
            return output

    def forward_threshold(self, x, threshold):
        output = self.forward(x)
        output = F.threshold(output, threshold, 0)
        return output

    def get_fc(self):
        fc = self.model.head  
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.model.head  

    def load_trained_fc(self, fc_state_dict_path):
        trained_fc_state_dict = torch.load(fc_state_dict_path)
        
        new_state_dict = {}
        for k, v in trained_fc_state_dict.items():
            new_key = k.replace('linearhead.', '')  
            new_state_dict[new_key] = v

        self.model.head.load_state_dict(new_state_dict)


class ViT_Ti(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.feature_size = 192  

        self.model = create_model('vit_tiny_patch16_224', pretrained=True)

        self.model.head = nn.Linear(self.feature_size, num_classes)

    def forward(self, x, return_feature=False):
        features = self.model.forward_features(x)  

        class_token = features[:, 0, :]  

        logits = self.model.head(class_token) 

        if return_feature:
            return logits, class_token  
        return logits

    def forward_threshold(self, x, threshold):
        features = self.model.forward_features(x)
        cls_token = features[:, 0, :]
        logits = self.model.head(cls_token)

        thresholded_logits = F.threshold(logits, threshold, 0)
        return thresholded_logits
    
    def get_fc(self):
        fc = self.model.head
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.model.head
    
    def load_trained_fc(self, fc_state_dict_path):
        trained_fc_state_dict = torch.load(fc_state_dict_path)
        
        new_state_dict = {}
        for k, v in trained_fc_state_dict.items():
            new_key = k.replace('linearhead.', '')  
            new_state_dict[new_key] = v

        self.model.head.load_state_dict(new_state_dict)

class ViT_S(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.feature_size = 384  

        self.model = create_model('vit_small_patch16_224', pretrained=True)

        self.model.head = nn.Linear(self.feature_size, num_classes)

    def forward(self, x, return_feature=False):
        features = self.model.forward_features(x)  

        class_token = features[:, 0, :]  

        logits = self.model.head(class_token)  

        if return_feature:
            return logits, class_token 
        return logits


    def forward_threshold(self, x, threshold):
        features = self.model.forward_features(x)
        cls_token = features[:, 0, :]
        logits = self.model.head(cls_token)

        thresholded_logits = F.threshold(logits, threshold, 0)
        return thresholded_logits

    def get_fc(self):
        fc = self.model.head
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.model.head
    
    def load_trained_fc(self, fc_state_dict_path):
        trained_fc_state_dict = torch.load(fc_state_dict_path)
        
        new_state_dict = {}
        for k, v in trained_fc_state_dict.items():
            new_key = k.replace('linearhead.', '')  
            new_state_dict[new_key] = v

        self.model.head.load_state_dict(new_state_dict)

class ViT_B(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.feature_size = 768  

        self.model = create_model('vit_base_patch16_224', pretrained=True)

        self.model.head = nn.Linear(self.feature_size, num_classes)

    def forward(self, x, return_feature=False):
        features = self.model.forward_features(x)  
        
        class_token = features[:, 0, :] 

        logits = self.model.head(class_token)  

        if return_feature:
            return logits, class_token  
        return logits


    def forward_threshold(self, x, threshold):
        features = self.model.forward_features(x)
        cls_token = features[:, 0, :]
        logits = self.model.head(cls_token)

        thresholded_logits = F.threshold(logits, threshold, 0)
        return thresholded_logits

    def get_fc(self):
        fc = self.model.head
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.model.head
    
    def load_trained_fc(self, fc_state_dict_path):
        trained_fc_state_dict = torch.load(fc_state_dict_path)
        
        new_state_dict = {}
        for k, v in trained_fc_state_dict.items():
            new_key = k.replace('linearhead.', '')  
            new_state_dict[new_key] = v

        self.model.head.load_state_dict(new_state_dict)



class DeiT_Ti(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.feature_size = 192  

        self.model = create_model('deit_tiny_patch16_224', pretrained=True)

        self.model.head = nn.Linear(self.feature_size, num_classes)

    def forward(self, x, return_feature=False):
        features = self.model.forward_features(x)  

        class_token = features[:, 0, :]  

        logits = self.model.head(class_token)  

        if return_feature:
            return logits, class_token  
        return logits


    def forward_threshold(self, x, threshold):
        features = self.model.forward_features(x)
        cls_token = features[:, 0, :]
        logits = self.model.head(cls_token)

        thresholded_logits = F.threshold(logits, threshold, 0)
        return thresholded_logits

    def get_fc(self):
        fc = self.model.head
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.model.head
    
    def load_trained_fc(self, fc_state_dict_path):
        trained_fc_state_dict = torch.load(fc_state_dict_path)
        
        new_state_dict = {}
        for k, v in trained_fc_state_dict.items():
            new_key = k.replace('linearhead.', '')  
            new_state_dict[new_key] = v

        self.model.head.load_state_dict(new_state_dict)

class DeiT_S(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.feature_size = 384  

        self.model = create_model('deit_small_patch16_224', pretrained=True)

        self.model.head = nn.Linear(self.feature_size, num_classes)

    def forward(self, x, return_feature=False):
        features = self.model.forward_features(x)  

        class_token = features[:, 0, :]  

        logits = self.model.head(class_token)  

        if return_feature:
            return logits, class_token  
        return logits
    

    def forward_threshold(self, x, threshold):
        features = self.model.forward_features(x)
        cls_token = features[:, 0, :]
        logits = self.model.head(cls_token)

        thresholded_logits = F.threshold(logits, threshold, 0)
        return thresholded_logits

    def get_fc(self):
        fc = self.model.head
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.model.head
    
    def load_trained_fc(self, fc_state_dict_path):
        trained_fc_state_dict = torch.load(fc_state_dict_path)
        
        new_state_dict = {}
        for k, v in trained_fc_state_dict.items():
            new_key = k.replace('linearhead.', '')  
            new_state_dict[new_key] = v

        self.model.head.load_state_dict(new_state_dict)

class DeiT_B(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.feature_size = 768  

        self.model = create_model('deit_base_patch16_224', pretrained=True)

        self.model.head = nn.Linear(self.feature_size, num_classes)

    def forward(self, x, return_feature=False):
        features = self.model.forward_features(x)  

        class_token = features[:, 0, :]  

        logits = self.model.head(class_token)  

        if return_feature:
            return logits, class_token  
        return logits


    def forward_threshold(self, x, threshold):
        features = self.model.forward_features(x)
        cls_token = features[:, 0, :]
        logits = self.model.head(cls_token)

        thresholded_logits = F.threshold(logits, threshold, 0)
        return thresholded_logits

    def get_fc(self):
        fc = self.model.head
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.model.head
    
    def load_trained_fc(self, fc_state_dict_path):
        trained_fc_state_dict = torch.load(fc_state_dict_path)
        
        new_state_dict = {}
        for k, v in trained_fc_state_dict.items():
            new_key = k.replace('linearhead.', '')  
            new_state_dict[new_key] = v

        self.model.head.load_state_dict(new_state_dict)


class DINOv2_ViT_S_14(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.feature_size = 384

        self.model = load('facebookresearch/dinov2', 'dinov2_vits14')

        self.model.head = nn.Linear(self.feature_size, num_classes)

    def forward(self, x, return_feature=False, return_feature_list=False):
        features = self.model.forward_features(x)  
        if isinstance(features, dict):
            cls_token = features['x_norm_clstoken']
        else:
            raise ValueError("Features format not recognized.")

        output = self.model.head(cls_token)

        if return_feature:
            return output, cls_token
        elif return_feature_list:
            return output, features
        else:
            return output


    def forward_threshold(self, x, threshold):
        features = self.model.forward_features(x)
        if isinstance(features, dict):
            cls_token = features['x_norm_clstoken']
        else:
            raise ValueError("Features format not recognized.")

        output = self.model.head(cls_token)

        output = F.threshold(output, threshold, 0)

        return output

    def intermediate_forward(self, x, layer_index):
        if layer_index < 1 or layer_index > len(self.model.blocks):
            raise ValueError("Invalid layer_index. Supported values are from 1 to {}.".format(len(self.model.blocks)))

        x = self.model.patch_embed(x)
        x = x + self.model.pos_embed[:, :x.size(1), :]
        x = self.model.pos_drop(x)

        for i, block in enumerate(self.model.blocks):
            x = block(x)
            if i + 1 == layer_index:
                return x

        return x  

    def get_fc(self):
        fc = self.model.head
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.model.head
    
    def load_trained_fc(self, fc_state_dict_path):
        trained_fc_state_dict = torch.load(fc_state_dict_path)
        
        new_state_dict = {}
        for k, v in trained_fc_state_dict.items():
            new_key = k.replace('linearhead.', '')
            new_state_dict[new_key] = v

        self.model.head.load_state_dict(new_state_dict)


class DINOv2_ViT_B_14(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.feature_size = 768

        self.model = load('facebookresearch/dinov2', 'dinov2_vitb14')

        self.model.head = nn.Linear(self.feature_size, num_classes)

    def forward(self, x, return_feature=False, return_feature_list=False):
        features = self.model.forward_features(x)  
        if isinstance(features, dict):
            cls_token = features['x_norm_clstoken']
        else:
            raise ValueError("Features format not recognized.")

        output = self.model.head(cls_token)

        if return_feature:
            return output, cls_token
        elif return_feature_list:
            return output, features
        else:
            return output


    def forward_threshold(self, x, threshold):
        features = self.model.forward_features(x)
        if isinstance(features, dict):
            cls_token = features['x_norm_clstoken']
        else:
            raise ValueError("Features format not recognized.")

        output = self.model.head(cls_token)

        output = F.threshold(output, threshold, 0)

        return output

    def intermediate_forward(self, x, layer_index):
        if layer_index < 1 or layer_index > len(self.model.blocks):
            raise ValueError("Invalid layer_index. Supported values are from 1 to {}.".format(len(self.model.blocks)))

        x = self.model.patch_embed(x)
        x = x + self.model.pos_embed[:, :x.size(1), :]
        x = self.model.pos_drop(x)

        for i, block in enumerate(self.model.blocks):
            x = block(x)
            if i + 1 == layer_index:
                return x

        return x  

    def get_fc(self):
        fc = self.model.head
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.model.head
    
    def load_trained_fc(self, fc_state_dict_path):
        trained_fc_state_dict = torch.load(fc_state_dict_path)
        
        new_state_dict = {}
        for k, v in trained_fc_state_dict.items():
            new_key = k.replace('linearhead.', '')
            new_state_dict[new_key] = v

        self.model.head.load_state_dict(new_state_dict)


class DINOv2_ViT_L_14(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.feature_size = 1024

        self.model = load('facebookresearch/dinov2', 'dinov2_vitl14')

        self.model.head = nn.Linear(self.feature_size, num_classes)

    def forward(self, x, return_feature=False, return_feature_list=False):
        features = self.model.forward_features(x)  
        if isinstance(features, dict):
            cls_token = features['x_norm_clstoken']
        else:
            raise ValueError("Features format not recognized.")

        output = self.model.head(cls_token)

        if return_feature:
            return output, cls_token
        elif return_feature_list:
            return output, features
        else:
            return output


    def forward_threshold(self, x, threshold):
        features = self.model.forward_features(x)
        if isinstance(features, dict):
            cls_token = features['x_norm_clstoken']
        else:
            raise ValueError("Features format not recognized.")

        output = self.model.head(cls_token)

        output = F.threshold(output, threshold, 0)

        return output

    def intermediate_forward(self, x, layer_index):
        if layer_index < 1 or layer_index > len(self.model.blocks):
            raise ValueError("Invalid layer_index. Supported values are from 1 to {}.".format(len(self.model.blocks)))

        x = self.model.patch_embed(x)
        x = x + self.model.pos_embed[:, :x.size(1), :]
        x = self.model.pos_drop(x)

        for i, block in enumerate(self.model.blocks):
            x = block(x)
            if i + 1 == layer_index:
                return x

        return x  

    def get_fc(self):
        fc = self.model.head
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.model.head
    
    def load_trained_fc(self, fc_state_dict_path):
        trained_fc_state_dict = torch.load(fc_state_dict_path)
        
        new_state_dict = {}
        for k, v in trained_fc_state_dict.items():
            new_key = k.replace('linearhead.', '')
            new_state_dict[new_key] = v

        self.model.head.load_state_dict(new_state_dict)




class CLIP_ViT_B_16(nn.Module):
    def __init__(self, num_classes=2, freeze_pretrained=True):
        super().__init__()
        
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

        if freeze_pretrained:
            for param in self.clip.parameters():
                param.requires_grad = False

        hidden_size = self.clip.config.projection_dim
        self.proj = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values, return_feature=False, return_feature_list=False):
        outputs = self.clip.get_image_features(pixel_values)  # shape: (batch_size, hidden_size)

        logits = self.proj(outputs)

        if return_feature:
            return logits, outputs
        elif return_feature_list:
            return logits, outputs
        else:
            return logits

    def forward_threshold(self, x, threshold):
        output = self.forward(x)
        return F.threshold(output, threshold, 0)

    def get_fc(self):
        return self.proj.weight.cpu().detach().numpy(), self.proj.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.proj

    def load_trained_fc(self, fc_state_dict_path):
        trained_fc_state_dict = torch.load(fc_state_dict_path)
        new_state_dict = {}
        for k, v in trained_fc_state_dict.items():
            new_key = k.replace('linearhead.', '')
            new_state_dict[new_key] = v
        self.proj.load_state_dict(new_state_dict)
    


class CLIP_RN50(nn.Module):
    def __init__(self, num_classes=2, freeze_pretrained=True):
        super().__init__()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("RN50", device=device)

        if freeze_pretrained:
            for param in self.model.parameters():
                param.requires_grad = False

        hidden_size = self.model.visual.output_dim  
        self.proj = nn.Linear(hidden_size, num_classes)

    def forward(self, images, return_feature=False, return_feature_list=False):
        with torch.no_grad():
            image_features = self.model.encode_image(images)  
        image_features = image_features.float()

        logits = self.proj(image_features)

        if return_feature:
            return logits, image_features
        elif return_feature_list:
            return logits, image_features
        else:
            return logits

    def forward_threshold(self, x, threshold):
        output = self.forward(x)
        return F.threshold(output, threshold, 0)

    def get_fc(self):
        return self.proj.weight.cpu().detach().numpy(), self.proj.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.proj

    def load_trained_fc(self, fc_state_dict_path):
        trained_fc_state_dict = torch.load(fc_state_dict_path)
        new_state_dict = {}
        for k, v in trained_fc_state_dict.items():
            new_key = k.replace('linearhead.', '')
            new_state_dict[new_key] = v
        self.proj.load_state_dict(new_state_dict)




class SwinBase(nn.Module):
    def __init__(self, model_name: str, feature_size: int, num_classes: int = 2):
        super().__init__()
        self.feature_size = feature_size
        self.model = create_model(model_name, pretrained=True)
        self.model.head = nn.Linear(self.feature_size, num_classes)

    def forward(self, x, return_feature=False):
        features = self.model.forward_features(x)  # (B, H, W, C)
        if features.ndim == 4:
            features = features.mean(dim=(1, 2))  # Global average pooling
        logits = self.model.head(features)
        return (logits, features) if return_feature else logits

    def forward_threshold(self, x, threshold):
        logits = self.forward(x)
        return F.threshold(logits, threshold, 0)

    def get_fc(self):
        fc = self.model.head
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.model.head

    def load_trained_fc(self, fc_state_dict_path):
        state = torch.load(fc_state_dict_path)
        new_state = {k.replace('linearhead.', ''): v for k, v in state.items()}
        self.model.head.load_state_dict(new_state)


class Swin_T(SwinBase):
    def __init__(self, num_classes=2):
        super().__init__('swin_tiny_patch4_window7_224', feature_size=768, num_classes=num_classes)


class Swin_S(SwinBase):
    def __init__(self, num_classes=2):
        super().__init__('swin_small_patch4_window7_224', feature_size=768, num_classes=num_classes)


class Swin_B(SwinBase):
    def __init__(self, num_classes=2):
        super().__init__('swin_base_patch4_window7_224', feature_size=1024, num_classes=num_classes)



class ConvNeXt_Base(nn.Module):
    def __init__(self, model_name='convnext_tiny', feature_size=768, num_classes=2):
        super().__init__()
        self.feature_size = feature_size
        self.model = create_model(model_name, pretrained=True)
        self.model.head = nn.Linear(self.feature_size, num_classes)

    def forward(self, x, return_feature=False, return_feature_list=False):
        features = self.model.forward_features(x)  # shape: [B, C, H, W]
        pooled = features.mean(dim=[2, 3])  # global average pooling -> shape: [B, C]
        logits = self.model.head(pooled)

        if return_feature:
            return logits, pooled
        elif return_feature_list:
            return logits, features
        else:
            return logits

    def forward_threshold(self, x, threshold):
        logits = self.forward(x)
        return F.threshold(logits, threshold, 0)

    def get_fc(self):
        fc = self.model.head
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()
    
    def get_fc_layer(self):
        return self.model.head

    def load_trained_fc(self, fc_state_dict_path):
        state = torch.load(fc_state_dict_path)
        new_state = {k.replace('linearhead.', ''): v for k, v in state.items()}
        self.model.head.load_state_dict(new_state)



class ConvNeXt_T(ConvNeXt_Base):
    def __init__(self, num_classes=2):
        super().__init__('convnext_tiny', feature_size=768, num_classes=num_classes)

class ConvNeXt_S(ConvNeXt_Base):
    def __init__(self, num_classes=2):
        super().__init__('convnext_small', feature_size=768, num_classes=num_classes)

class ConvNeXt_B(ConvNeXt_Base):
    def __init__(self, num_classes=2):
        super().__init__('convnext_base', feature_size=1024, num_classes=num_classes)
