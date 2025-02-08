import torch
import argparse
import trimesh
import logging
from meshgpt_pytorch import (
    MeshAutoencoder,
    MeshTransformer,
    mesh_render
)
import os
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
    parser.add_argument('--transformer_path', type=str, default="/media/ry/rayan/mesh_postprocess/checkpoints/mesh-encoder_16k_2_4_0.339.pt", help='Transformer 权重路径')
    parser.add_argument('--autoencoder_path', type=str, default="/media/ry/rayan/mesh_postprocess/checkpoints/mesh-transformer.16k_768_12_12_loss_2.335.pt", help='Autoencoder 权重路径')
    parser.add_argument('--obj_input_path', type=str, default="./obj_input/human10.obj", help='input path')
    parser.add_argument('--obj_output_path', type=str, default="./obj_output", help='output path')
    return parser.parse_args()

def load_mesh_obj(file_path):
    """加载 OBJ 格式的 mesh"""
    logger.info(f'Loading OBJ mesh from {file_path}')
    mesh = trimesh.load_mesh(file_path)
    return torch.tensor(mesh.vertices, dtype=torch.float32), torch.tensor(mesh.faces, dtype=torch.long),torch.tensor(mesh.edges, dtype=torch.long)

def load_mesh_ply(file_path):
    """加载 PLY 格式的 mesh"""
    logger.info(f'Loading PLY mesh from {file_path}')
    mesh = trimesh.load_mesh(file_path)
    return torch.tensor(mesh.vertices, dtype=torch.float32), torch.tensor(mesh.faces, dtype=torch.long),torch.tensor(mesh.edges, dtype=torch.long)

def load_mesh(file_path):
    """根据文件扩展名选择加载 OBJ 或 PLY"""
    if file_path.endswith('.obj'):
        return load_mesh_obj(file_path)
    elif file_path.endswith('.ply'):
        return load_mesh_ply(file_path)
    else:
        raise ValueError(f'Unsupported file format: {file_path}')
    
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f'Using device: {device}')
    
    #transformer = MeshTransformer.from_pretrained("").to(device)
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
    
    #transformer = load_weights(transformer, args.transformer_path, device)
    #autoencoder = load_weights(autoencoder, args.autoencoder_path, device)
    pkg = torch.load(args.autoencoder_path) 
    autoencoder.load_state_dict(pkg['model'],strict=False)
    ##pkg = torch.load(args.transformer_path) 
    #transformer.load_state_dict(pkg['model'],strict=False)
    # 设置为评估模式
    autoencoder.eval()
    #pkg = torch.load(args.transformer_path) 
    #transformer.load_state_dict(pkg['model'],strict=False)
    
    # 根据 obj_input_path 选择加载函数
    vertices, faces,face_edges = load_mesh(args.obj_input_path)
    vertices = vertices.to(device)
    faces = faces.to(device)
    face_edges = face_edges.to(device)
    logger.info(f'vertices.shape:{vertices.shape}')
    logger.info(f'faces.shape:{faces.shape}')
    logger.info(f'Mesh loaded successfully')
    with torch.no_grad():
        # 1. Tokenize mesh
        codes = autoencoder.tokenize(vertices=vertices, faces=faces, face_edges=face_edges)
        codes = codes.flatten().unsqueeze(0)
        # 确保 codes 的长度是 num_quantizers 的整数倍
        num_quantizers = autoencoder.num_quantizers
        codes = codes[:, :codes.shape[-1] // num_quantizers * num_quantizers]

        # 2. 解码为 mesh faces
        coords, mask = autoencoder.decode_from_codes_to_faces(codes)
        # 获取原始 mesh 数据对应的 faces 的顶点坐标
        orgs = vertices[faces].unsqueeze(0)

        # 3. 计算 MSE（仅作为指标，可按需删除）
        mse = torch.mean((orgs.view(-1, 3).cpu() - coords.view(-1, 3).cpu()) ** 2)
        logger.info(f'MSE: {mse.item()}')

        # 如有需要，可将结果保存到列表中
        all_random_samples = [(coords, orgs)]

    # 构造输出文件路径，并保存渲染结果
    input_basename = os.path.basename(args.obj_input_path)
    input_no_ext = os.path.splitext(input_basename)[0]
    os.makedirs(args.obj_output_path, exist_ok=True)
    output_file_path = os.path.join(args.obj_output_path, f"{input_no_ext}_render.obj")
    mesh_render.save_rendering(output_file_path, all_random_samples)

    logger.info('Processing complete')
    # min_mse, max_mse = float('inf'), float('-inf')
    # min_coords, min_orgs, max_coords, max_orgs = None, None, None, None
    # random_samples, random_samples_pred, all_random_samples = [], [], []
    # total_mse, sample_size = 0.0, 200
    # codes = autoencoder.tokenize(vertices=vertices, faces=faces, face_edges=face_edges) 
    
    # codes = codes.flatten().unsqueeze(0)
    # codes = codes[:, :codes.shape[-1] // autoencoder.num_quantizers * autoencoder.num_quantizers] 
 
    # coords, mask = autoencoder.decode_from_codes_to_faces(codes)
    # orgs = vertices[faces].unsqueeze(0)

    # mse = torch.mean((orgs.view(-1, 3).cpu() - coords.view(-1, 3).cpu())**2)
    # total_mse += mse 

    # if mse < min_mse: min_mse, min_coords, min_orgs = mse, coords, orgs
    # if mse > max_mse: max_mse, max_coords, max_orgs = mse, coords, orgs
 
    # if len(random_samples) <= 30:
    #     random_samples.append(coords)
    #     random_samples_pred.append(orgs)
    # else:
    #     all_random_samples.extend([random_samples_pred, random_samples])
    #     random_samples, random_samples_pred = [], []
    
    
    # # 可选：进行 mesh 处理或推理
    # #output = transformer.generate(texts=['sofa', 'chair'], temperature=0.0)
    # # output = transformer.generate(
    # #                                 #texts=['sofa', 'chair'],
    # #                                 vertices = vertices,
    # #                                 faces = faces,
    # #                                 temperature=0.0
    # #                               )
    # # 从输入路径中提取基名并去掉扩展名
    # input_basename = os.path.basename(args.obj_input_path)  # 获取文件名带扩展
    # input_no_ext = os.path.splitext(input_basename)[0]      # 去掉扩展名
    
    # # 确保输出目录存在
    # os.makedirs(args.obj_output_path, exist_ok=True)    
    # # 生成输出文件路径
    # output_file_path = os.path.join(args.obj_output_path, f"{input_no_ext}_render.obj")


    # # 调用 save_rendering 函数并使用从命令行获取的输出路径
    # mesh_render.save_rendering(output_file_path, all_random_samples)
    
    # logger.info('Processing complete')

if __name__ == "__main__":
    main()