from __future__ import annotations

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import argparse
from datetime import datetime

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


if __name__ == '__main__':
    args = parse_args()
    config = OmegaConf.structured(BaseConfig)
    start_time = datetime.now()
    if args.custom_cfg:
        custom_config = OmegaConf.load(args.custom_cfg)
        config = OmegaConf.merge(config, custom_config)
    config = OmegaConf.create(OmegaConf.to_yaml(config, resolve=True))
     #打印config
    config_file = "/workspace/MARIO_EVAL/data/runtime/config.yaml"
    with open(config_file, "w") as f:
        f.write(OmegaConf.to_yaml(config, resolve=True))

    llm_version = os.path.basename(config.model_dir.rstrip("/"))#提取模型路径的最后一个文件夹名

    data = load_qaf(args.qaf)#加载问题答案文件
    solver = Solver(config=config)
    #先清空"//workspace/MARIO_EVAL/data/pic_tree"文件夹 
    
    # 删除并重新创建文件夹
    folder_path = '/workspace/MARIO_EVAL/data/pic_tree'
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

    os.makedirs(folder_path)

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

    folder_path4 = '/workspace/MARIO_EVAL/data/runtime_mfu'
    if os.path.exists(folder_path4):
        shutil.rmtree(folder_path4)

    os.makedirs(folder_path4)

    folder_path5 = '/workspace/MARIO_EVAL/data/pic_mfu_step'
    if os.path.exists(folder_path5):
        shutil.rmtree(folder_path5) 

    os.makedirs(folder_path5)

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
        for cur_data in tqdm(batch(data, config.batch_size), desc="Main Processing"):
            agents = [method(config=config, question=d["question"], ground_truth=d["answer"] if config.is_sampling else None) 
                      for d in cur_data]#每个d是一个字典，包含问题、答案等信息
            #agents是一个列表，每个元素是一个agent对象，每个agent对象包含一个method对象，一个question，一个ground_truth，这里的method是MCTS
            agent_file = "/workspace/MARIO_EVAL/data/runtime/agent"
            #直接print(agents)到agent_file
            with open(agent_file, "w") as f:
                f.write(str(agents))
                f.flush()
                
            jsonlines = solver.solve(agents)
            print(jsonlines)
            """
            for d in cur_data:
                question = d["question"]
                d["react"] = jsonlines[question]
                writer.write(json.dumps(d, ensure_ascii=False) + '\n')
                writer.flush()
            """
    end_time = datetime.now()
    print(f"Time cost: {end_time - start_time}")
    
