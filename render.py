import torch
from meshgpt_pytorch import (
    MeshAutoencoder,
    MeshTransformer,
    mesh_render
)

device = "cuda" if torch.cuda.is_available() else "cpu"
autoencoder = MeshAutoencoder(
        decoder_dims_through_depth=(128,) * 6 + (192,) * 12 + (256,) * 24 + (384,) * 6,
        dim_codebook=192,
        dim_area_embed=16,
        dim_coor_embed=16,
        dim_normal_embed=16,
        dim_angle_embed=8,
        attn_decoder_depth=4,
        attn_encoder_depth=2
    ).to(device)

transformer = MeshTransformer(
    autoencoder,
    dim=768,
    coarse_pre_gateloop_depth=2,
    fine_pre_gateloop_depth=2,
    attn_depth=12,
    attn_heads=12,
    dropout=0.0,
    max_seq_len=1500,
    condition_on_text=True,
    gateloop_use_heinsen=False,
    text_condition_model_types="bge",
    text_condition_cond_drop_prob=0.0,
).to(device)
pkg = torch.load('/media/ry/rayan/mesh_postprocess/checkpoints/mesh-encoder_16k_2_4_0.339.pt') 
autoencoder.load_state_dict(pkg['model'],strict=False)
pkg = torch.load('/media/ry/rayan/mesh_postprocess/checkpoints/mesh-transformer.16k_768_12_12_loss_2.335.pt') 
transformer.load_state_dict(pkg['model'],strict=False)

output = []  
output.append((transformer.generate(texts = ['sofa','bed', 'computer screen', 'bench', 'chair', 'table' ] , temperature = 0.0) ))   
output.append((transformer.generate(texts = ['milk carton', 'door', 'shovel', 'heart', 'trash can', 'ladder'], temperature = 0.0) )) 
output.append((transformer.generate(texts = ['hammer', 'pedestal', 'pickaxe', 'wooden cross', 'coffee bean', 'crowbar'],  temperature = 0.0) )) 
output.append((transformer.generate(texts = ['key', 'minecraft character', 'dragon head', 'open book', 'minecraft turtle', 'wooden table'], temperature = 0.0) )) 
output.append((transformer.generate(texts = ['gun', 'ice cream cone', 'axe', 'helicopter', 'shotgun', 'plastic bottle'], temperature = 0.0) )) 

mesh_render.save_rendering(f'./render.obj', output)