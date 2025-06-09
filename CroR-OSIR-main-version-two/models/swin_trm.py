import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformer
from timm.layers import to_2tuple

class SwinTransformerCustom(nn.Module):
    def __init__(self, img_size=64, output_channels=128, output_size=8):
        super(SwinTransformerCustom, self).__init__()
        # Initialize the Swin Transformer
        self.swin_transformer = SwinTransformer(img_size=img_size,
                                                patch_size=4,
                                                in_chans=3,
                                                embed_dim=96,
                                                depths=(2, 2, 6, 2),
                                                num_heads=(3, 6, 12, 24),
                                                window_size=7,
                                                mlp_ratio=4.,
                                                qkv_bias=True,
                                                drop_rate=0.,
                                                attn_drop_rate=0.,
                                                drop_path_rate=0.1,
                                                norm_layer=nn.LayerNorm,
                                                ape=False,
                                                patch_norm=True,
                                                use_checkpoint=False)
        
        # Modify the patch embedding layer to accept 64x64 input
        self.swin_transformer.patch_embed.img_size = to_2tuple(img_size)
        self.swin_transformer.patch_embed.grid_size = (img_size // self.swin_transformer.patch_embed.patch_size[0], 
                                                       img_size // self.swin_transformer.patch_embed.patch_size[1])
        num_patches = self.swin_transformer.patch_embed.grid_size[0] * self.swin_transformer.patch_embed.grid_size[1]
        self.swin_transformer.patch_embed.num_patches = num_patches

        # Update positional embeddings
        self.swin_transformer.pos_drop = nn.Dropout(p=0.0)
        self.swin_transformer.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.swin_transformer.embed_dim))

        # Remove the classification head
        self.swin_transformer.head = nn.Identity()

        # Add a convolutional layer to reshape the output
        self.conv = nn.ConvTranspose2d(in_channels=3072, out_channels=128, kernel_size=8, stride=8)
        self.output_size = output_size

    def forward(self, x):
        # Forward pass through Swin Transformer
        x = self.swin_transformer(x) #[batchsize,3,64,64] -> [batchsize,3072,1,1]
        
        # Reshape the output to be a 4D tensor for the convolutional layer
        batch_size, num_features = x.shape[0], x.shape[3]*4
        x = x.view(batch_size, num_features, 1, 1)  # (batch_size, num_features, 1, 1)

        # Apply the convolutional layer
        x = self.conv(x) #[batchszie,3072,1,1] -> [batchsize,128,1,1]
        
        # Resize to the desired output shape
        x = nn.functional.interpolate(x, size=(self.output_size, self.output_size), mode='bilinear', align_corners=False) #进行插值和采样，从而提高分辨率的技术
        #[batchsize,128,1,1] -> [batchsize,128,8,8]
        return x

# Create an instance of the modified Swin Transformer
model = SwinTransformerCustom()

# Print the model architecture
# print(model)
