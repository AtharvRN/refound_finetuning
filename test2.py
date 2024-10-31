from models_mae import mae_vit_large_patch16_dec512d8b
import models_vit
import torch
from util.pos_embed import interpolate_pos_embed,interpolate_decoder_pos_embed
from timm.models.layers import trunc_normal_
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF

model = mae_vit_large_patch16_dec512d8b(img_size=224)
print(model.pos_embed.shape)
# print(model)
num_params = sum(p.numel() for p in model.parameters())
print(num_params)

# load RETFound weights
checkpoint = torch.load('RETFound_oct_weights.pth', map_location='cpu')
checkpoint_model = checkpoint['model']
# print(checkpoint_model)
# print(checkpoint_model['pos_embed'].shape)
# print(checkpoint_model['head.bias'])
state_dict = model.state_dict()

for k in ['head.weight', 'head.bias']:
    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        print(f"Removing key {k} from pretrained checkpoint")
        del checkpoint_model[k]

# checkpoint_model = interpolate_pos_embed(model, checkpoint_model)
# checkpoint_model = interpolate_decoder_pos_embed(model, checkpoint_model)
# # load pre-trained model
msg = model.load_state_dict(checkpoint_model, strict=False)

# x = torch.randn(1,3,224,224)
# path = "/home/tejadhith/Project/OCT/Dataset/stavan_images_March16/RID_1001000629_20181112_153114_L_CIRRUS_HD-OCT_5000_512x1024x128_ORG_IMG_JPG_0009.jpg"

path = "/home/tejadhith/Project/OCT/Dataset/segregated_28-sep-2023_kath/RID_1001000001_20181231_155031_R_CIRRUS_HD-OCT_5000_512x1024x128_ORG_IMG_JPG_0089.jpg"

path = "/home/tejadhith/Project/OCT/Dataset/segregated_28-sep-2023_kath/RID_1001007923_20180417_162800_L_CIRRUS_HD-OCT_5000_512x1024x128_ORG_IMG_JPG_0001.jpg"
test_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to the desired size
        # transforms.Resize((512, 512)),  # Resize images to the desired size
        transforms.Grayscale(num_output_channels=3),  # Convert single-channel images to 3 channels
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # Normalize pixel values
    ])

img = Image.open(path)
plt.imsave("OCT_image.png",img,cmap="gray")


img_tensor = test_transform(img)
img_tensor = img_tensor.unsqueeze(0)

resized_img = img_tensor.squeeze().cpu().detach().numpy()[0]
plt.imsave("OCT_resized_image.png",resized_img,cmap="gray")


# mask_ratios = np.arange(0,100,10)
# for mask_ratio in mask_ratios:
#     loss,pred,mask =model(img_tensor,mask_ratio=0)
#     # print(pred.shape)
#     recon_img = model.unpatchify(pred)
#     recon_img_np = recon_img.squeeze().cpu().detach().numpy()  # Assuming recon_img is a single-channel image
#     recon_img_np = recon_img_np[0]
#     file_name = f"OCT_recon_mask_{mask_ratio}.png"
#     plt.imsave(file_name,recon_img_np,cmap="gray")
mask_ratios = np.arange(0, 101, 10)
num_subplots = len(mask_ratios) + 1  # Include the original image

# Create subplots
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 9))  # Adjust dimensions as needed
axes = axes.flatten()  # Flatten for easier iteration

# Original image 
axes[0].imshow(img_tensor.squeeze().cpu().detach().numpy()[0], cmap='gray')
axes[0].set_title('Original Image')

# Reconstructed images
for i, mask_ratio in enumerate(mask_ratios):
    # print(mask_ratio)
    print(i)
    print(img_tensor.shape)
    loss, pred, mask = model(img_tensor, mask_ratio=mask_ratio/100)

    recon_img = model.unpatchify(pred)
    # print(recon_img.shape)
    recon_img_np = recon_img.squeeze().cpu().detach().numpy()
    recon_img_np = recon_img_np[0]
    axes[i + 1].imshow(recon_img_np, cmap='gray')
    axes[i + 1].set_title(f'Reconstructed (Mask Ratio: {mask_ratio}%)')
    axes[i+1].axis("off")

# Adjust layout and save
plt.tight_layout()
plt.savefig("OCT_reconstructions.png")
    # recon_img_np = recon_img.squeeze().cpu().detach().numpy()[0]  # Assuming recon_img is a single-channel image
    # mask_np = mask.squeeze().cpu().detach().numpy()  # Assuming mask is 1-dimensional
    # mask_np = mask_np.reshape(32,32)
    # # print(img_tensor.shape)
    # # print(recon_img_np.shape)
    # # print(mask_np)
    # for ph in range(len(mask_np)):
    #     for pw in range(len(mask_np)):
    #         if mask_np[ph,pw] == 1:
    #             recon_img_np[ph*16:(ph+1)*16, pw*16:(pw+1)*16] = resized_img[ph*16:(ph+1)*16, pw*16:(pw+1)*16]

    # plt.imsave("OCT_recon_patches_mask_{mask_ratio}.png",recon_img_np,cmap="gray")



# z_latent = torch.random.randn(1,50,1024)
# print()
     
# print(recon_img_np[:,:16,:16].shape)
# print(img_tensor[:,:,:16,:16].squeeze().cpu().detach().numpy().shape)
# # print(recon_img_np[:,:16,:16] - img_tensor[:,:,:16,:16].squeeze().cpu().detach().numpy())
# print(recon_img_np[0,:16,:16]-resized_img[:16,:16])
# print(recon_img_np[:,:16,:16] -img_tensor[0,:,:16,:16].)


# recon_img_np = recon_img_np.transpose(1,2,0)
# # print(recon_img_np.shape)
# recon_img_np1 =  recon_img_np[0]
# recon_img_np2 =  recon_img_np[1]
# recon_img_np3 =  recon_img_np[2]

# recon_img_np1 = (recon_img_np1 * 255).astype(np.uint8)  # Convert to uint8
# recon_img_np2 = (recon_img_np2 * 255).astype(np.uint8)  # Convert to uint8
# recon_img_np3 = (recon_img_np3 * 255).astype(np.uint8)  # Convert to uint8

# plt.imsave("OCT_recon_patches1.png",recon_img_np1,cmap="gray")
# plt.imsave("OCT_recon_patches2.png",recon_img_np2,cmap="gray")
# plt.imsave("OCT_recon_patches3.png",recon_img_np3,cmap="gray")

# recon_img_pil = TF.to_pil_image(recon_img_np)

# # Save or visualize the reconstructed image with replaced patches
# recon_img_pil.save("OCT_recon_with_patches.png")
