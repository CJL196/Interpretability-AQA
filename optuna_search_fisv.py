import os
import optuna
import subprocess
import json
from datetime import datetime
import tempfile

def create_config_file(params, config_id):
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
q_number = 136
# variange for initilize query
query_var = {params['query_var']}
# positional embedding method ["query_pe","query_memory_pe","no"]
pe = "query_pe"
att_loss = True
dino_loss = True
max_len = 136

num_layers = {params['num_layers']}

# training config
split_feats = True
epoch_num = 1000
lr = {params['lr']}
"""
    
    config_path = f"configs/optuna_search_{config_id}.py"
    with open(config_path, 'w') as f:
        f.write(config_content)
    return config_path

def extract_best_rho(log_file):
    """从日志文件中提取最佳SRCC值"""
    best_rho = 0.0
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if 'best:' in line and 'correlation:' in line:
                    parts = line.split('best:')
                    if len(parts) > 1:
                        rho_value = float(parts[1].strip())
                        best_rho = max(best_rho, rho_value)
    except Exception as e:
        print(f"解析日志文件失败: {e}")
    return best_rho

def objective(trial):
    """Optuna优化目标函数"""
    
    # 定义超参数搜索空间
    params = {
        'seed': trial.suggest_categorical('seed', [999, 666, 123]),
        'label': trial.suggest_categorical('label', ['PCS']),
        'bs_train': trial.suggest_categorical('bs_train', [16, 32, 48]),
        'query_var': trial.suggest_float('query_var', 1.0, 10.0, step=0.5),
        'num_layers': trial.suggest_int('num_layers', 1, 4),
        'lr': trial.suggest_float('lr', 1e-5, 5e-4, log=True),
        'dataset_name': 'fisv',
        'att_loss': True,
        'pe': 'query_pe',
        'dino_loss': True
    }
    
    config_id = f"trial_{trial.number:04d}"
    config_path = create_config_file(params, config_id)
    
    try:
        print(f"\n试验 {trial.number}: {params}")
        
        # 运行训练
        result = subprocess.run([
            'python3', 'main.py', '--config', config_path
        ], capture_output=True, text=True, timeout=7200)
        
        print(result)
        
        # 解析日志获取最佳性能
        log_file = f"exp/{params['seed']}_{params['dataset_name']}_{params['label']}_{params['att_loss']}_{params['query_var']}_{params['pe']}.log"
        best_rho = extract_best_rho(log_file)
        
        print(f"试验 {trial.number} 完成，SRCC: {best_rho}")
        
        return best_rho
        
    except subprocess.TimeoutExpired:
        print(f"试验 {trial.number} 超时")
        return 0.0
    except Exception as e:
        print(f"试验 {trial.number} 失败: {e}")
        return 0.0
    finally:
        # 清理临时配置文件
        if os.path.exists(config_path):
            os.remove(config_path)

def main():
    # 创建Optuna研究，使用固定的数据库文件名
    storage_url = "sqlite:///optuna.db"
    study_name = "fisv_optimization"
    
    study = optuna.create_study(
        direction='maximize',  # 最大化SRCC
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),  # 使用TPE采样器
        pruner=optuna.pruners.MedianPruner(  # 使用中位数剪枝器
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=1
        )
    )
    
    print(f"开始Optuna超参数优化")
    print(f"研究名称: {study_name}")
    print(f"数据库: {storage_url}")
    
    # 运行优化
    try:
        study.optimize(objective, n_trials=100, timeout=86400)  # 100次试验或24小时
    except KeyboardInterrupt:
        print("优化被用户中断")
    
    # 输出结果
    print(f"\n优化完成!")
    print(f"最佳SRCC: {study.best_value}")
    print(f"最佳参数: {study.best_params}")
    print(f"完成试验数: {len(study.trials)}")
    
    # 保存最佳配置
    best_config_path = create_config_file(study.best_params, 'best_optuna')
    print(f"最佳配置已保存到: {best_config_path}")
    
    # 保存优化历史到CSV
    trials_df = study.trials_dataframe()
    trials_df.to_csv('optimization_history.csv', index=False)
    print(f"优化历史已保存到: optimization_history.csv")
    
    # 生成优化可视化图表
    try:
        import optuna.visualization as vis
        import plotly
        
        # 优化历史图
        fig1 = vis.plot_optimization_history(study)
        fig1.write_html('optimization_history.html')
        
        # 参数重要性图
        fig2 = vis.plot_param_importances(study)
        fig2.write_html('param_importances.html')
        
        # 参数关系图
        fig3 = vis.plot_parallel_coordinate(study)
        fig3.write_html('parallel_coordinate.html')
        
        print("可视化图表已保存: optimization_history.html, param_importances.html, parallel_coordinate.html")
        
    except ImportError:
        print("安装 plotly 以生成可视化图表: pip install plotly")

if __name__ == "__main__":
    main()
