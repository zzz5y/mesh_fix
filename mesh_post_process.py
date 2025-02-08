import torch
import argparse
import trimesh
import logging
from meshgpt_pytorch import (
    MeshAutoencoder,
    MeshTransformer,
    mesh_render
)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_weights(model, path, device):
    """加载模型权重"""
    logger.info(f'Loading weights from {path}')
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--transformer_path', type=str, required=True, help='Transformer 权重路径')
    parser.add_argument('--autoencoder_path', type=str, required=True, help='Autoencoder 权重路径')
    return parser.parse_args()

def load_mesh_obj(file_path):
    """加载 OBJ 格式的 mesh"""
    logger.info(f'Loading OBJ mesh from {file_path}')
    mesh = trimesh.load_mesh(file_path)
    return torch.tensor(mesh.vertices, dtype=torch.float32), torch.tensor(mesh.faces, dtype=torch.long)

def load_mesh_ply(file_path):
    """加载 PLY 格式的 mesh"""
    logger.info(f'Loading PLY mesh from {file_path}')
    mesh = trimesh.load_mesh(file_path)
    return torch.tensor(mesh.vertices, dtype=torch.float32), torch.tensor(mesh.faces, dtype=torch.long)

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f'Using device: {device}')
    
    transformer = MeshTransformer.from_pretrained("").to(device)
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
    
    transformer = load_weights(transformer, args.transformer_path, device)
    autoencoder = load_weights(autoencoder, args.autoencoder_path, device)
    
    # 示例：加载一个 OBJ 或 PLY mesh
    vertices, faces = load_mesh_obj("example.obj")  # 或 load_mesh_ply("example.ply")
    
    logger.info('Mesh loaded successfully')
    
    # 可选：进行 mesh 处理或推理
    # output = transformer.generate(texts=['sofa', 'chair'], temperature=0.0)
    # mesh_render.save_rendering('./render.obj', output)
    
    logger.info('Processing complete')

if __name__ == "__main__":
    main()