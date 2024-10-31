import torch
import models_vit
from util.pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_

# call the model
model = models_vit.__dict__['vit_multi_head_classifiers'](
    num_outputs = 8,
    drop_path_rate=0.2,
    global_pool=True,
)
num_params = sum(p.numel() for p in model.parameters())
print(num_params)

# load RETFound weights
checkpoint = torch.load('RETFound_oct_weights.pth', map_location='cpu')
checkpoint_model = checkpoint['model']
# print(checkpoint_model['pos_embed'].shape)
# print(checkpoint_model['head.bias'])
state_dict = model.state_dict()
for k in ['head.weight', 'head.bias']:
    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        print(f"Removing key {k} from pretrained checkpoint")
        del checkpoint_model[k]

# interpolate position embedding
# print(model.pos_embed.shape)
interpolate_pos_embed(model, checkpoint_model)

# # load pre-trained model
msg = model.load_state_dict(checkpoint_model, strict=False)

# assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
# assert set(msg.missing_keys) == {'classifichead.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}

# # manually initialize fc layer
# trunc_normal_(model.head.weight, std=2e-5)
# print(model)
x = torch.randn(100,3,224,224)
y =model(x)
print(y.shape)
num_params = sum(p.numel() for p in model.parameters())
print(num_params)
# print(y.shape)
# print("Model = %s" % str(model))