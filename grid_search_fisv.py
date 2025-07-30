import os
import itertools
import subprocess
import json
from datetime import datetime

def create_config_file(base_config, params, config_id):
    """创建临时配置文件"""
    config_content = f"""
seed = {params['seed']}
multi_gpu = "0"

# dataset config (logo, gym, fisv)
dataset_name = "fisv"
fisv_dir = "./fis-v"
label = "{params['label']}"

# dataloader config
subset = 0
bs_train = {params['bs_train']}
bs_test = 32
num_workers = 4

# network config (i3d, vivit)
backbone = "vivit"

i3d = dict(
    backbone="I3D",
    neck="",
    evaluator="",
    I3D_ckpt_path="model_rgb.pth" 
)

vivit = dict(
    backbone="ViViT",
)

neck = "TQN"
head = "weighted"
# query number
q_number = {params['q_number']}
# variange for initilize query
query_var = {params['query_var']}
# positional embedding method ["query_pe","query_memory_pe","no"]
pe = "{params['pe']}" 
att_loss = {params['att_loss']}
dino_loss = {params['dino_loss']}
max_len = 136

num_layers = {params['num_layers']}

# training config
split_feats = True
epoch_num = {params['epoch_num']}
lr = {params['lr']}
"""
    
    config_path = f"configs/grid_search_fisv_{config_id}.py"
    with open(config_path, 'w') as f:
        f.write(config_content)
    return config_path

def run_experiment(config_path, params, results_file):
    """运行单个实验"""
    print(f"运行实验: {params}")
    
    try:
        # 运行训练
        result = subprocess.run([
            'python3', 'main.py', '--config', config_path
        ], capture_output=True, text=True, timeout=7200)  # 2小时超时
        
        # 解析日志获取最佳性能
        log_file = f"exp/{params['seed']}_{params['dataset_name']}_{params['label']}_{params['att_loss']}_{params['query_var']}_{params['pe']}.log"
        best_rho = extract_best_rho(log_file)
        
        # 保存结果
        result_entry = {
            'params': params,
            'best_rho': best_rho,
            'status': 'completed',
            'timestamp': datetime.now().isoformat()
        }
        
        with open(results_file, 'a') as f:
            f.write(json.dumps(result_entry) + '\n')
            
        print(f"实验完成，最佳SRCC: {best_rho}")
        return best_rho
        
    except subprocess.TimeoutExpired:
        print("实验超时")
        return 0.0
    except Exception as e:
        print(f"实验失败: {e}")
        return 0.0
    finally:
        # 清理临时配置文件
        if os.path.exists(config_path):
            os.remove(config_path)

def extract_best_rho(log_file):
    """从日志文件中提取最佳SRCC值"""
    best_rho = 0.0
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if 'best:' in line and 'correlation:' in line:
                    parts = line.split('best:')
                    if len(parts) > 1:
                        best_rho = max(best_rho, float(parts[1].strip()))
    except:
        pass
    return best_rho

def main():
    # 定义超参数搜索空间
    param_grid = {
        'seed': [999, 666, 123],
        'label': ['PCS'],
        'bs_train': [16, 32, 48],
        'q_number': [100, 136, 150],
        'query_var': [2, 5, 8],
        'pe': ['query_pe', 'query_memory_pe'],
        'att_loss': [True, False],
        'dino_loss': [True, False],
        'num_layers': [1, 2, 3],
        'epoch_num': [1000],  # 减少训练轮数以加快搜索
        'lr': [1e-4, 5e-5, 2e-4],
        'dataset_name': ['fisv']
    }
    
    # 创建结果目录
    os.makedirs('grid_search_results', exist_ok=True)
    results_file = f'grid_search_results/fisv_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl'
    
    # 生成所有参数组合
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))
    
    print(f"总共需要运行 {len(combinations)} 个实验")
    
    best_result = {'params': None, 'rho': 0.0}
    
    for i, combination in enumerate(combinations):
        params = dict(zip(keys, combination))
        config_id = f"{i:04d}"
        
        print(f"\n进度: {i+1}/{len(combinations)}")
        
        # 创建配置文件
        config_path = create_config_file(None, params, config_id)
        
        # 运行实验
        rho = run_experiment(config_path, params, results_file)
        
        # 更新最佳结果
        if rho > best_result['rho']:
            best_result['params'] = params.copy()
            best_result['rho'] = rho
            print(f"发现新的最佳结果! SRCC: {rho}")
    
    # 输出最终结果
    print(f"\n网格搜索完成!")
    print(f"最佳SRCC: {best_result['rho']}")
    print(f"最佳参数: {best_result['params']}")
    
    # 保存最佳配置
    if best_result['params']:
        best_config_path = create_config_file(None, best_result['params'], 'best')
        print(f"最佳配置已保存到: {best_config_path}")

if __name__ == "__main__":
    main()