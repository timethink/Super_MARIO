from __future__ import annotations

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from termcolor import colored
from tqdm import tqdm
import time
from mcts_math.agents import SBSREACT
from mcts_math.agents import MCTS
from mcts_math.solver import Solver
from mcts_math.config import BaseConfig
from react_demo import load_qaf
from react_batch_demo import batch
import shutil
import subprocess
import yaml
import torch
from vllm.distributed import cleanup_dist_env_and_memory

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--custom_cfg', type=str, default="configs/sbs_sft.yaml")
    args.add_argument(
        "--qaf", "--question-answer-file", 
        type=str, 
        required=True,
        help="the file includes question / partial solution (optional) / answer (optional)")

    args = args.parse_args()
    return args

def read_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    """        data = {
            "step": final_step,
            "seq_len": final_seq_len,
            "average_prefill_len": final_prefill_len,
            "average_decode_len": final_decode_len,
            "hfu": final_pre_mfu,
            "mfu": final_nopre_mfu,
            "time": final_time,
            "unfinished": final_unfinished,
            "pre_flops": final_pre_flops,
            "nopre_flops": final_nopre_flops,
            "pre_linear_flops": final_pre_linear_flops,
            "pre_attention_flops": final_pre_attention_flops,
            "nopre_linear_flops": final_nopre_linear_flops,
            "nopre_attention_flops": final_nopre_attention_flops
        }"""
    return data["step"],data["seq_len"], data['average_prefill_len'],data['average_decode_len'],data["hfu"], data["mfu"],data["time"], data["unfinished"],data["pre_flops"],data["nopre_flops"],data["pre_linear_flops"],data["pre_attention_flops"],data["nopre_linear_flops"],data["nopre_attention_flops"]

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

#修改成main()
#if __name__ == '__main__':
def main():
    args = parse_args()
    config = OmegaConf.structured(BaseConfig)
    start_time = datetime.now()
    if args.custom_cfg:
        custom_config = OmegaConf.load(args.custom_cfg)
        config = OmegaConf.merge(config, custom_config)
    config = OmegaConf.create(OmegaConf.to_yaml(config, resolve=True))
    #将config的enable_prefix_caching改为True得到config2
    #config2 = OmegaConf.create(OmegaConf.to_yaml(config, resolve=True).replace("enable_prefix_caching: false", "enable_prefix_caching: true"))
     #打印config
    config_file = "/workspace/MARIO_EVAL/data/runtime/config.yaml"
    with open(config_file, "w") as f:
        f.write(OmegaConf.to_yaml(config, resolve=True))

    llm_version = os.path.basename(config.model_dir.rstrip("/"))#提取模型路径的最后一个文件夹名

    #提取qaf的前question_range个问题作为新的文件保存
    question_range = config.question_range
    with open(args.qaf, "r") as f:
        data = json.load(f)
    data = data[:question_range]
    new_qaf = f"/workspace/MARIO_EVAL/data/runtime_data/qaf_{question_range}.json"
    with open(new_qaf, "w") as f:
        json.dump(data, f, ensure_ascii=False)
    #data = load_qaf(args.qaf)#加载问题答案文件
    data = load_qaf(new_qaf)#加载问题答案文件
    
    #先清空"//workspace/MARIO_EVAL/data/pic_tree"文件夹 
    
    """
    # 删除并重新创建文件夹
    folder_path = '/workspace/MARIO_EVAL/data/pic_tree'
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

    os.makedirs(folder_path)
    """

    folder_path1 = '/workspace/MARIO_EVAL/data/runtime_tree'
    if os.path.exists(folder_path1):
        shutil.rmtree(folder_path1)
    
    #folder_path100 = '/workspace/MARIO_EVAL/data/runtime_tree1'
    #if os.path.exists(folder_path100):
    #    shutil.rmtree(folder_path100)

    os.makedirs(folder_path1)

    
    folder_path2 = '/workspace/MARIO_EVAL/data/runtime_prompt'
    if os.path.exists(folder_path2):
        shutil.rmtree(folder_path2)

    os.makedirs(folder_path2)

    folder_path3 = '/workspace/MARIO_EVAL/data/runtime_output'
    if os.path.exists(folder_path3):
        shutil.rmtree(folder_path3)

    os.makedirs(folder_path3)



    foldername = f'/workspace/MARIO_EVAL/data/runtime_data/{config.run_tool}_{config.batch_size}b_{config.n_generate_sample}sample_{config.iterations}iter_{config.question_range}_qaf_{config.num_few_shot}example_{config.batch_id}batch_id'
    if os.path.exists(foldername) and config.enable_prefix_caching == False:
        shutil.rmtree(foldername)
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    
    #修改

    solver = Solver(config=config)
    
    """
    folder_path5 = '/workspace/MARIO_EVAL/data/runtime_seq'
    if os.path.exists(folder_path5):
        shutil.rmtree(folder_path5)

    os.makedirs(folder_path5)

    folder_path6 = '/workspace/MARIO_EVAL/data/runtime_time'
    if os.path.exists(folder_path6):
        shutil.rmtree(folder_path6)

    os.makedirs(folder_path6)

    folder_path7 = '/workspace/MARIO_EVAL/data/runtime_percent'
    if os.path.exists(folder_path7):
        shutil.rmtree(folder_path7)

    os.makedirs(folder_path7)
    """


    """
    folder_path5 = '/workspace/MARIO_EVAL/data/pic_mfu_step'
    if os.path.exists(folder_path5):
        shutil.rmtree(folder_path5) 

    os.makedirs(folder_path5)
    """
    log_filename = "/workspace/MARIO_EVAL/data/llm_stats.txt"
    if os.path.exists(log_filename):
        os.remove(log_filename)

    stats_filename = "/workspace/MARIO_EVAL/data/runtime_data/stats.json"
    if os.path.exists(stats_filename):
        os.remove(stats_filename)

    # init method
    if config.mode == "mcts":
        method = MCTS
    elif config.mode == "sbs":
        method = SBSREACT
    else:
        raise NotImplementedError
    
    saved_jsonl_file = f"mcts_result.jsonl" 
    with open(saved_jsonl_file, "w") as writer:
        batch_file = "/workspace/MARIO_EVAL/data/runtime/batch.json"
        #输出batch(data, config.batch_size)到batch_file
        with open(batch_file, "w") as f:
            for cur_data in tqdm(batch(data, config.batch_size), desc="Main Processing"):
                f.write(json.dumps(cur_data, ensure_ascii=False) + '\n')
                f.flush()
        #batch_size默认为-1,即只有一个batch,如果batch_size>0,则每个batch的大小为batch_size
        #batch(data, config.batch_size)返回一个生成器,每次生成一个batch的数据
        #由于data是一个列表，每个元素是一个字典，包含问题、答案等信息，所以cur_data是一个包含了多个题目的列表，每个元素是问题，列表长度为batch_size
        batch_id = 0
        for cur_data in tqdm(batch(data, config.batch_size), desc="Main Processing"):
            agents = [method(config=config, question=d["question"], ground_truth=d["answer"] if config.is_sampling else None) 
                      for d in cur_data]#每个d是一个字典，包含问题、答案等信息
            #添加agents2
            #agents2 = [method(config=config2, question=d["question"], ground_truth=d["answer"] if config.is_sampling else None) 
            #            for d in cur_data]#每个d是一个字典，包含问题、答案等信息
            #agents是一个列表，每个元素是一个agent对象，每个agent对象包含一个method对象，一个question，一个ground_truth，这里的method是MCTS
            #获取当前是第几个batch
            
            agent_file = "/workspace/MARIO_EVAL/data/runtime/agent"
            #直接print(agents)到agent_file
            with open(agent_file, "w") as f:
                f.write(str(agents))
                f.flush()

            print(f"batch_id: {batch_id}\n")
                
            jsonlines = solver.solve(agents, batch_id)
            batch_id += 1
            #print(jsonlines)
            """
            for d in cur_data:
                question = d["question"]
                d["react"] = jsonlines[question]
                writer.write(json.dumps(d, ensure_ascii=False) + '\n')
                writer.flush()
       
             """
            
    end_time = datetime.now()
    print(f"Time cost: {end_time - start_time}")
    solver.llm_shutdown()
    return config
    """
    #清楚之前的solver等数据，避免内存溢出
    del solver
    del agents
    solver2 = Solver(config=config2)
    saved_jsonl_file = f"mcts_result1.jsonl" 
    with open(saved_jsonl_file, "w") as writer:
        batch_file = "/workspace/MARIO_EVAL/data/runtime/batch1.json"
        #输出batch(data, config.batch_size)到batch_file
        with open(batch_file, "w") as f:
            for cur_data in tqdm(batch(data, config.batch_size), desc="Main Processing"):
                f.write(json.dumps(cur_data, ensure_ascii=False) + '\n')
                f.flush()
        #batch_size默认为-1,即只有一个batch,如果batch_size>0,则每个batch的大小为batch_size
        #batch(data, config.batch_size)返回一个生成器,每次生成一个batch的数据
        #由于data是一个列表，每个元素是一个字典，包含问题、答案等信息，所以cur_data是一个包含了多个题目的列表，每个元素是问题，列表长度为batch_size
        for cur_data in tqdm(batch(data, config.batch_size), desc="Main Processing"):
            #添加agents2
            agents2 = [method(config=config2, question=d["question"], ground_truth=d["answer"] if config.is_sampling else None) 
                        for d in cur_data]#每个d是一个字典，包含问题、答案等信息
            #agents是一个列表，每个元素是一个agent对象，每个agent对象包含一个method对象，一个question，一个ground_truth，这里的method是MCTS
            agent_file = "/workspace/MARIO_EVAL/data/runtime/agent"
            #直接print(agents)到agent_file
            with open(agent_file, "w") as f:
                f.write(str(agents2))
                f.flush()
                
            jsonlines2 = solver2.solve(agents2)
    """


def draw_pic(config):
        #输出数据对比文件
    batch_num = config.question_range // config.batch_size
    #分多个batch输出数据
    for batch_id in range(batch_num):
        foldername = f'/workspace/MARIO_EVAL/data/runtime_data/{config.run_tool}_{config.batch_size}b_{config.n_generate_sample}sample_{config.iterations}iter_{config.question_range}_qaf_{config.num_few_shot}example_{batch_id}batch_id'
        datafile0 = f"{foldername}/final_data{0}.json"
        datafile1 = f"{foldername}/final_data{1}.json"
        """with open(data_filename, "w") as f:
                f.write(str(final_seq_len))
                f.write("\n")
                f.write(str(final_mfu))
                f.write("\n")
                f.write(str(final_time))
                f.write("\n")
                f.write(str(final_unfinished))
                f.write("\n")"""
        #datafile中分别包含了final_seq_len, final_mfu, final_time, final_unfinished
        #将两个datafile中的四个数值分别取出，画在同一张图上，得到四个图，横轴为step=[0,1,2,...]
        steps0,  seq_len0, average_prefill_len0, average_decode_len0, hfu0, mfu0, time0, unfinished0, pre_flops0, nopre_flops0, pre_linear_flops0, pre_attention_flops0, nopre_linear_flops0, nopre_attention_flops0 = read_data(datafile0)
        steps1, seq_len1, average_prefill_len1, average_decode_len1, hfu1, mfu1, time1, unfinished1 , pre_flops1, nopre_flops1, pre_linear_flops1, pre_attention_flops1, nopre_linear_flops1, nopre_attention_flops1 = read_data(datafile1)
        #画图
        #将seq_len0, seq_len1画在一张图上
        
        plt.plot(steps0, seq_len0, label="without_prefix_caching")
        plt.plot(steps1, seq_len1, label="with_prefix_caching")
        plt.legend()
        plt.xlabel("step")
        plt.ylabel("seq tokens num") 
        plt.title("seq_len")
        plt.savefig(f"{foldername}/seq_len.png")
        #关闭当前图
        plt.close()

        #将average_prefill_len0, average_prefill_len1画在一张图上
        plt.plot(steps0, average_prefill_len0, label="without_prefix_caching")
        plt.plot(steps1, average_prefill_len1, label="with_prefix_caching")
        plt.legend()
        plt.xlabel("step")
        plt.ylabel("prefill tokens num")
        plt.title("average prefill_len")
        plt.savefig(f"{foldername}/prefill_len.png")
        plt.close()

        #将average_decode_len0, average_decode_len1画在一张图上
        plt.plot(steps0, average_decode_len0, label="without_prefix_caching")
        plt.plot(steps1, average_decode_len1, label="with_prefix_caching")
        plt.legend()
        plt.xlabel("step")
        plt.ylabel("decode tokens num")
        plt.title("decode_len")
        plt.savefig(f"{foldername}/decode_len.png")
        plt.close()
        
        #将final_mfu0, final_mfu1画在一张图上
        plt.plot(steps0, hfu0, label="without_prefix_caching")
        plt.plot(steps1, hfu1, label="with_prefix_caching")
        plt.legend()
        plt.xlabel("step")
        plt.ylabel("hfu")
        plt.title("hfu")
        plt.savefig(f"{foldername}/hfu.png")
        plt.close()


        plt.plot(steps0, mfu0, label="without_prefix_caching")
        plt.plot(steps1, mfu1, label="with_prefix_caching")
        plt.legend()
        plt.xlabel("step")
        plt.ylabel("mfu")
        plt.title("mfu")
        plt.savefig(f"{foldername}/mfu.png")
        plt.close()

        #将time0, time1画在一张图上
        plt.plot(steps0, time0, label="without_prefix_caching")
        plt.plot(steps1, time1, label="with_prefix_caching")
        plt.legend()
        plt.xlabel("step")
        plt.ylabel("time/s")
        plt.title("time")
        plt.savefig(f"{foldername}/time.png")
        plt.close()
        #将finished0, finished1画在一张图上
        finished0 = [1 - i for i in unfinished0]
        finished1 = [1 - i for i in unfinished1]
        plt.plot(steps0, finished0, label="without_prefix_caching")
        plt.plot(steps1, finished1, label="with_prefix_caching")
        plt.legend()
        plt.xlabel("step")
        plt.ylabel("finished num/total num")
        plt.title("finished problems")
        plt.savefig(f"{foldername}/finished.png")
        plt.close()
        #将pre_flops0,pre_linear_flops0, pre_attention_flops0, pre_flops1,pre_linear_flops1, pre_attention_flops1画在一张图上
        #plt.plot(steps0, pre_flops0, label="pre_flops w/o")
        #plt.plot(steps1, pre_flops1, label="pre_flops w")
        plt.plot(steps0, pre_linear_flops0, label="hfu_linear_flops w/o")
        plt.plot(steps1, pre_linear_flops1, label="hfu_linear_flops w")
        plt.plot(steps0, pre_attention_flops0, label="hfu_attention_flops w/o")
        plt.plot(steps1, pre_attention_flops1, label="hfu_attention_flops w")
        plt.legend()
        plt.xlabel("step")
        plt.ylabel("flops")
        plt.title("pre_flops")
        plt.savefig(f"{foldername}/pre_flops.png")
        plt.close()
        #将nopre_flops0,nopre_linear_flops0, nopre_attention_flops0, nopre_flops1,nopre_linear_flops1, nopre_attention_flops1画在一张图上
        #plt.plot(steps0, nopre_flops0, label="nopre_flops w/o")
        #plt.plot(steps1, nopre_flops1, label="nopre_flops w")
        plt.plot(steps0, nopre_linear_flops0, label="mfu_linear_flops w/o")
        plt.plot(steps1, nopre_linear_flops1, label="mfu_linear_flops w")
        plt.plot(steps0, nopre_attention_flops0, label="mfu_attention_flops w/o")
        plt.plot(steps1, nopre_attention_flops1, label="mfu_attention_flops w")
        plt.legend()
        plt.xlabel("step")
        plt.ylabel("flops")
        plt.title("nopre_flops")
        plt.savefig(f"{foldername}/nopre_flops.png")
        plt.close()

def sglang_vllm_pic(config):
    batch_num = config.question_range // config.batch_size
    #分多个batch输出数据
    for batch_id in range(batch_num):
        foldername1 = f'/workspace/MARIO_EVAL/data/runtime_data/vllm_{config.batch_size}b_{config.n_generate_sample}sample_{config.iterations}iter_{config.question_range}_qaf_{config.num_few_shot}example_{batch_id}batch_id'
        foldername2 = f'/workspace/MARIO_EVAL/data/runtime_data/sglang_{config.batch_size}b_{config.n_generate_sample}sample_{config.iterations}iter_{config.question_range}_qaf_{config.num_few_shot}example_{batch_id}batch_id'

        foldername3 = f'/workspace/MARIO_EVAL/data/runtime_data/compare_{config.batch_size}b_{config.n_generate_sample}sample_{config.iterations}iter_{config.question_range}_qaf_{config.num_few_shot}example_{batch_id}batch_id'

        if os.path.exists(foldername3):
            shutil.rmtree(foldername3)

        os.makedirs(foldername3)

        datafile0 = f"{foldername1}/final_data{1}.json"
        datafile1 = f"{foldername2}/final_data{1}.json"
        #datafile中分别包含了final_seq_len, final_mfu, final_time, final_unfinished
        #将两个datafile中的四个数值分别取出，画在同一张图上，得到四个图，横轴为step=[0,1,2,...]
        steps0,  seq_len0, average_prefill_len0, average_decode_len0, hfu0, mfu0, time0, unfinished0, pre_flops0, nopre_flops0, pre_linear_flops0, pre_attention_flops0, nopre_linear_flops0, nopre_attention_flops0= read_data(datafile0)
        steps1, seq_len1, average_prefill_len1, average_decode_len1, hfu1, mfu1, time1, unfinished1, pre_flops1, nopre_flops1, pre_linear_flops1, pre_attention_flops1, nopre_linear_flops1, nopre_attention_flops1  = read_data(datafile1)      

        #将average_prefill_len0, average_prefill_len1画在一张图上
        plt.plot(steps0, average_prefill_len0, label="vllm")
        plt.plot(steps1, average_prefill_len1, label="sglang")
        plt.legend()
        plt.xlabel("step")
        plt.ylabel("prefill tokens num")
        plt.title("average prefill_len")
        plt.savefig(f"{foldername3}/enable_compare_prefill_len.png")
        plt.close()
        #将average_decode_len0, average_decode_len1画在一张图上
        plt.plot(steps0, average_decode_len0, label="vllm")
        plt.plot(steps1, average_decode_len1, label="sglang")
        plt.legend()
        plt.xlabel("step")
        plt.ylabel("decode tokens num")
        plt.title("decode_len")
        plt.savefig(f"{foldername3}/enable_compare_decode_len.png")
        plt.close()  
        #将final_mfu0, final_mfu1画在一张图上
        plt.plot(steps0, hfu0, label="vllm")
        plt.plot(steps1, hfu1, label="sglang")
        plt.legend()
        plt.xlabel("step")
        plt.ylabel("hfu")
        plt.title("hfu")
        plt.savefig(f"{foldername3}/enable_compare_hfu.png")
        plt.close()
        #将hfu0, hfu1画在一张图上
        plt.plot(steps0, mfu0, label="vllm")
        plt.plot(steps1, mfu1, label="sglang")
        plt.legend()
        plt.xlabel("step")
        plt.ylabel("mfu")
        plt.title("mfu")
        plt.savefig(f"{foldername3}/enable_compare_mfu.png")
        plt.close()
        #将time0, time1画在一张图上
        plt.plot(steps0, time0, label="vllm")
        plt.plot(steps1, time1, label="sglang")
        plt.legend()
        plt.xlabel("step")
        plt.ylabel("time/s")
        plt.title("time")
        plt.savefig(f"{foldername3}/enable_compare_time.png")
        plt.close()
        #将finished0, finished1画在一张图上
        finished0 = [1 - i for i in unfinished0]
        finished1 = [1 - i for i in unfinished1]
        plt.plot(steps0, finished0, label="vllm")
        plt.plot(steps1, finished1, label="sglang")
        plt.legend()
        plt.xlabel("step")
        plt.ylabel("finished num/total num")
        plt.title("finished problems")
        plt.savefig(f"{foldername3}/enable_compare_finished.png")
        plt.close()

        #将开关打开前的对比放在vllm中
        datafile0 = f"{foldername1}/final_data{0}.json"
        datafile1 = f"{foldername2}/final_data{0}.json"
        #datafile中分别包含了final_seq_len, final_mfu, final_time, final_unfinished
        #将两个datafile中的四个数值分别取出，画在同一张图上，得到四个图，横轴为step=[0,1,2,...]
        steps0,  seq_len0, average_prefill_len0, average_decode_len0, hfu0, mfu0, time0, unfinished0, pre_flops0, nopre_flops0, pre_linear_flops0, pre_attention_flops0, nopre_linear_flops0, nopre_attention_flops0= read_data(datafile0)
        steps1, seq_len1, average_prefill_len1, average_decode_len1, hfu1, mfu1, time1, unfinished1, pre_flops1, nopre_flops1, pre_linear_flops1, pre_attention_flops1, nopre_linear_flops1, nopre_attention_flops1  = read_data(datafile1)      

        #将average_prefill_len0, average_prefill_len1画在一张图上
        plt.plot(steps0, average_prefill_len0, label="vllm")
        plt.plot(steps1, average_prefill_len1, label="sglang")
        plt.legend()
        plt.xlabel("step")
        plt.ylabel("prefill tokens num")
        plt.title("average prefill_len")
        plt.savefig(f"{foldername3}/without_compare_prefill_len.png")
        plt.close()
        #将average_decode_len0, average_decode_len1画在一张图上
        plt.plot(steps0, average_decode_len0, label="vllm")
        plt.plot(steps1, average_decode_len1, label="sglang")
        plt.legend()
        plt.xlabel("step")
        plt.ylabel("decode tokens num")
        plt.title("decode_len")
        plt.savefig(f"{foldername3}/without_compare_decode_len.png")
        plt.close()  
        #将final_mfu0, final_mfu1画在一张图上
        plt.plot(steps0, hfu0, label="vllm")
        plt.plot(steps1, hfu1, label="sglang")
        plt.legend()
        plt.xlabel("step")
        plt.ylabel("hfu")
        plt.title("hfu")
        plt.savefig(f"{foldername3}/without_compare_hfu.png")
        plt.close()
        #将hfu0, hfu1画在一张图上
        plt.plot(steps0, mfu0, label="vllm")
        plt.plot(steps1, mfu1, label="sglang")
        plt.legend()
        plt.xlabel("step")
        plt.ylabel("mfu")
        plt.title("mfu")
        plt.savefig(f"{foldername3}/without_compare_mfu.png")
        plt.close()
        #将time0, time1画在一张图上
        plt.plot(steps0, time0, label="vllm")
        plt.plot(steps1, time1, label="sglang")
        plt.legend()
        plt.xlabel("step")
        plt.ylabel("time/s")
        plt.title("time")
        plt.savefig(f"{foldername3}/without_compare_time.png")
        plt.close()
        #将finished0, finished1画在一张图上
        finished0 = [1 - i for i in unfinished0]
        finished1 = [1 - i for i in unfinished1]
        plt.plot(steps0, finished0, label="vllm")
        plt.plot(steps1, finished1, label="sglang")
        plt.legend()
        plt.xlabel("step")
        plt.ylabel("finished num/total num")
        plt.title("finished problems")
        plt.savefig(f"{foldername3}/without_compare_finished.png")
        plt.close()

    

if __name__ == '__main__':
    #sglang_vllm_pic()
    
    test_batch_size = [1]
    test_n_generate_sample = [2]
    test_iterations = [40]
    test_question_range = [4]
    num_few_shots = [0]
    run_tool = [ "sglang", "vllm"]

    for batch_size in test_batch_size:
        for n_generate_sample in test_n_generate_sample:
            for iterations in test_iterations:
                for question_range in test_question_range:
                    for num_few_shot in num_few_shots:
                        for tool in run_tool:
                            new_params = {
                                'batch_size': batch_size,
                                'n_generate_sample': n_generate_sample,
                                'iterations': iterations,
                                'question_range': question_range,
                                'enable_prefix_caching': False,
                                'best_of': n_generate_sample,
                                "num_few_shot": num_few_shot,
                                "run_tool": tool
                            }
                            modify_yaml("configs/mcts_sft.yaml", **new_params)
                            print(f"batch_size: {batch_size}, n_generate_sample: {n_generate_sample}, iterations: {iterations}, question_range: {question_range}, num_few_shot: {num_few_shot}, run_tool: {tool}")
                            
                            
                            config = main()
                            
                            #cleanup_dist_env_and_memory()
                            enable_params = {
                                'enable_prefix_caching': True
                            }
                            
                            modify_yaml("configs/mcts_sft.yaml", **enable_params)
                            print(f"batch_size: {batch_size}, n_generate_sample: {n_generate_sample}, iterations: {iterations}, question_range: {question_range}, num_few_shot: {num_few_shot}, run_tool: {tool}")
                            config2 = main()
                            draw_pic(config)
                        sglang_vllm_pic(config)
    
