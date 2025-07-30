import os
import itertools
import subprocess
import json
from datetime import datetime

def create_config_file(params, config_id):
    """创建临时配置文件"""
    config_content = f"""
seed = {params['seed']}
multi_gpu = "0"
dataset_name = "fisv"
fisv_dir = "./fis-v"
label = "{params['label']}"

subset = 0
bs_train = {params['bs_train']}
bs_test = 32
num_workers = 4

backbone = "vivit"
i3d = dict(backbone="I3D", neck="", evaluator="", I3D_ckpt_path="model_rgb.pth")
vivit = dict(backbone="ViViT")

neck = "TQN"
head = "weighted"
q_number = {params['q_number']}
query_var = {params['query_var']}
pe = "{params['pe']}"
att_loss = {params['att_loss']}
dino_loss = {params['dino_loss']}
max_len = 136
num_layers = {params['num_layers']}

split_feats = True
epoch_num = {params['epoch_num']}
lr = {params['lr']}
"""
    
    config_path = f"configs/quick_search_{config_id}.py"
    with open(config_path, 'w') as f:
        f.write(config_content)
    return config_path

def main():
    # 重点搜索关键超参数
    param_grid = {
        'seed': [999],
        'label': ['PCS'],  # 先搜索PCS
        'bs_train': [32, 48],
        'q_number': [136, 150],
        'query_var': [5, 8],
        'pe': ['query_pe'],
        'att_loss': [True],
        'dino_loss': [True, False],
        'num_layers': [1, 2],
        'epoch_num': [200],  # 快速验证
        'lr': [1e-4, 5e-5]
    }
    
    os.makedirs('quick_search_results', exist_ok=True)
    results_file = f'quick_search_results/fisv_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl'
    
    combinations = list(itertools.product(*param_grid.values()))
    print(f"快速搜索: {len(combinations)} 个实验")
    
    best_result = {'params': None, 'rho': 0.0}
    
    for i, combination in enumerate(combinations):
        params = dict(zip(param_grid.keys(), combination))
        config_path = create_config_file(params, f"quick_{i:03d}")
        
        print(f"\n运行 {i+1}/{len(combinations)}: {params}")
        
        try:
            result = subprocess.run([
                'python3', 'main.py', '--config', config_path
            ], capture_output=True, text=True, timeout=3600)
            
            # 简单解析输出获取最佳相关性
            output_lines = result.stdout.split('\n')
            best_rho = 0.0
            for line in output_lines:
                if 'best:' in line:
                    try:
                        best_rho = float(line.split('best:')[1].strip())
                    except:
                        pass
            
            if best_rho > best_result['rho']:
                best_result = {'params': params.copy(), 'rho': best_rho}
                print(f"新最佳: {best_rho}")
            
            # 记录结果
            with open(results_file, 'a') as f:
                f.write(json.dumps({'params': params, 'rho': best_rho}) + '\n')
                
        except Exception as e:
            print(f"实验失败: {e}")
        finally:
            if os.path.exists(config_path):
                os.remove(config_path)
    
    print(f"\n快速搜索完成!")
    print(f"最佳SRCC: {best_result['rho']}")
    print(f"最佳参数: {best_result['params']}")

if __name__ == "__main__":
    main()