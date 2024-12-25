import subprocess
import yaml
import os

def modify_yaml(config_file, **kwargs):
    """
    修改 YAML 配置文件中的参数。
    
    :param config_file: 配置文件路径
    :param kwargs: 要修改的键值对
    """
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    # 修改配置中的参数
    for key, value in kwargs.items():
        if key in config:
            config[key] = value
        else:
            print(f"Warning: Key '{key}' not found in the config file.")
    
    # 将修改后的配置写回到文件
    with open(config_file, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    print(f"Updated {config_file} with new parameters.")

def run_solver(cfg_file, qaf_file):
    """
    执行 solver_demo.py 脚本。

    :param cfg_file: 配置文件路径
    :param qaf_file: QAF 文件路径
    """
    command = [
        'python', 'solver_demo.py',
        '--custom_cfg', cfg_file,
        '--qaf', qaf_file
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    
    # 输出命令执行的结果
    if result.returncode == 0:
        print("Solver ran successfully.")
        print(result.stdout)
    else:
        print("Error occurred while running the solver.")
        print(result.stderr)

def main():
    config_file = 'configs/mcts_sft.yaml'
    qaf_file = '../MARIO_EVAL/data/math_test.json'

    # 定义要测试的参数组合
    batch_sizes = [4, 8, 16, 32]
    n_generate_samples = [8, 16, 32]
    question_ranges = [16, 64, 128]
    iterations = 40

    # 循环生成不同的参数组合并执行
    for batch_size in batch_sizes:
        for n_generate_sample in n_generate_samples:
            for question_range in question_ranges:
                # 定义新的参数集
                new_params = {
                    'batch_size': batch_size,
                    'n_generate_sample': n_generate_sample,
                    "best_of": n_generate_sample,
                    'iterations': iterations,
                    'question_range': question_range
                }

                # 修改配置文件
                modify_yaml(config_file, **new_params)
                
                # 执行 solver_demo.py
                run_solver(config_file, qaf_file)

if __name__ == "__main__":
    main()
