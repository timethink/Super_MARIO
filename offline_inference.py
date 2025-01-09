"""
author: lovecambi
email: interfk@gmail.com
"""
from __future__ import annotations

import argparse
import json
import numpy as np

from typing import Any, Dict, Type, Optional, List
from pydantic import BaseModel
from omegaconf import OmegaConf
from tqdm import tqdm
from graphviz import Digraph
from mcts_math.constants import (
    NO_VALID_CHILD, 
    TOO_MANY_STEPS, 
    TOO_MANY_CODE_ERRORS, 
)
from mcts_math.config import BaseConfig
from mcts_math.agents.utils import math_is_equiv
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer,AutoTokenizer

class InferNode(BaseModel):

    tag: str = "0"

    text: str = ""
    extra_info: str = ""
    action: str = ""
    action_input: str = ""
    final_answer: str = ""

    c_puct: float = 1.25
    depth: int = 0

    prior: float = 1.0
    value: float = 0
    q_value: float = 0
    visit_count: int = 0

    parent: Optional[Any] = None
    children: List[Any] = []

    prune: bool = False
    pre_token_num: int = 0#添加，从根节点到当前节点的token数量（不包括当前节点）
    token_num: int = 0#添加，节点的token数量
    total_flops: int = 0#添加,表示当前节点在产生child的过程中的flops数量
    linear_flops: int = 0#添加 
    attention_flops: int = 0#添加
    def puct(self) -> float:
        q_value = self.q_value if self.visit_count > 0 else 0
        u_value = self.c_puct * self.prior * np.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return q_value + u_value


def rebuild_tree(
    tree_dict: Dict[str, Any], 
    max_num_children: int,
    c_puct: float,
    root_tag: str = "0",
) -> Tuple[Type[InferNoder], int]:
    root = InferNode(
        parent=None,
        tag=root_tag,
        c_puct=c_puct,
        **tree_dict[root_tag],
    )
    candidates = [root]
    max_depth = 0
    while candidates:
        node = candidates.pop(0)
        for idx in range(max_num_children):
            tag = f"{node.tag}.{idx}"
            depth = node.depth + 1
            if tag in tree_dict:
                child = InferNode(
                    parent=node,
                    tag=tag,
                    depth=depth,
                    c_puct=c_puct,
                    **tree_dict[tag],
                )
                max_depth = max(max_depth, depth)
                node.children.append(child)
                candidates.append(child)
    return root, max_depth


def is_valid_final_answer_node(node: Type[InferNode]) -> bool:
    if not node.children and node.final_answer and \
        node.final_answer not in [NO_VALID_CHILD, TOO_MANY_STEPS, TOO_MANY_CODE_ERRORS]:
        return True
    return False


def prune_node(node: Type[InferNode]) -> bool:
    if node.children:
        children_prune = []
        for child in node.children:
            children_prune.append(prune_node(child))
        if all(children_prune):
            node.prune = True
    else:
        # for leaf node
        if not is_valid_final_answer_node(node): 
            node.prune = True
    return node.prune


def select_non_prune(current_nodes: List[Type[InferNode]]) -> List[Type[InferNode]]:
        candidate_nodes = []
        for node in current_nodes:
            candidate_nodes.extend([child for child in node.children if not child.prune])
        return candidate_nodes


def sort_by_strategy(
    candidate_nodes: List[Type[InferNode]],
    strategy: str = "q_value",
) -> List[Type[InferNode]]:
    if strategy == "value":
        return sorted(candidate_nodes, key=lambda x: x.value, reverse=True)
    elif strategy == "q_value":
        return sorted(candidate_nodes, key=lambda x: x.q_value, reverse=True)
    elif strategy == "visit_count":
        return sorted(candidate_nodes, key=lambda x: x.visit_count, reverse=True)
    elif strategy == "puct":
        return sorted(candidate_nodes, key=lambda x: x.puct(), reverse=True)
    else:
        raise NotImplementedError(f"strategy {strategy} not implemented")

def get_solution( 
    full_tree_dict: Dict[str, Any], 
    prune: bool = False,
    b1: int = 1,
    b2: int = 5,
    strategy: str = "q_value",
    c_puct: float = 1.25,
) -> Optional[Dict[str, Any]]:
    """
    This function is used to extract solution from a built tree.
    It is mainly used for MCTS, but also works for saved tree from step_beam.
    """
    question = full_tree_dict["question"]
    ground_truth = full_tree_dict.get("answer", None)
    tree_dict = full_tree_dict["react"]

    # rebuild tree
    root, tree_depth = rebuild_tree(tree_dict, max_num_children=b1*b2, c_puct=c_puct)

    # pruning tree
    if prune:
        prune_node(root)
        if root.prune:
            # no valid leaf node for the entire tree
            return None
    
    # search in tree
    final_answer_nodes = []
    current_top_num = b1
    current_nodes = [root] 

    for _ in range(tree_depth):
        candidate_nodes = select_non_prune(current_nodes)
        candidate_nodes = sort_by_strategy(candidate_nodes, strategy)
        current_nodes = candidate_nodes[:current_top_num]

        for current_node in current_nodes[:]:
            if is_valid_final_answer_node(current_node):
                final_answer_nodes.append(current_node)
                current_nodes.remove(current_node)
                current_top_num -= 1
            elif not current_node.children:
                current_nodes.remove(current_node)
                current_top_num -= 1
    
    if not final_answer_nodes:
        return None

    final_answer_nodes = sort_by_strategy(final_answer_nodes, strategy)
    top_final_answer_node = final_answer_nodes[0]

    # for node in final_answer_nodes:
    #     print(node.tag)

    return {
        "question": question,
        "ground_truth": ground_truth,
        "final_answer": top_final_answer_node.final_answer,
        "tag": top_final_answer_node.tag,
    }


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--custom_cfg", type=str, default="configs/offline_inference.yaml")
    args.add_argument("--tree_jsonl", type=str, required=True, default="saved tree jsonl file.")

    args = args.parse_args()
    return args

def cal_tree():
    tree_file_name = "/workspace/MARIO_EVAL/mcts_tree.json"
    i = 0
    with open(tree_file_name, "r") as f:
        for line in f:
            full_tree_dict = json.loads(line)
            #建树
            tree_dict = full_tree_dict["react"]
            root, tree_depth = rebuild_tree(tree_dict, max_num_children=16, c_puct=1.25)
            #tree_filename1 = f"/workspace/MARIO_EVAL/data/runtime_tree/question{i}_tree.json"
            tree_pic_name =  f"/workspace/MARIO_EVAL/data/pic_tree/question{i}_tree"
            
            #计算node的token数量
            get_token_num(root)
            #计算flops
            total_flops, linear_flops, attention_flops = calculate_tree_flops(root)

            
            print(f"question {i} total_flops: {total_flops}, linear_flops: {linear_flops}, attention_flops: {attention_flops}")
            tree_flops_file = f"/workspace/MARIO_EVAL/data/pic_tree/question{i}_tree_flops.txt"
            with open(tree_flops_file, "w") as f:
                #使用科学计数法
                f.write(f"total_flops: {total_flops:.2e}\n")
                f.write(f"linear_flops: {linear_flops:.2e}\n")
                f.write(f"attention_flops: {attention_flops:.2e}\n")
            draw_tree_pic(root, tree_pic_name)
            i += 1

def get_token_num(root: Type[InferNode]):
    #加载tokenizer
    tokenizer_path = "/workspace/Qwen2.5-Math-7B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": ['<code>', '<end_of_step>', '<end_of_code>', '<output>', '<end_of_output>', '<answer>', '<end_of_answer>', '<|user|>', '<|assistant|>', '<refine>', '<end_of_refine>', '\n<|assistant|>']
        },
        replace_additional_special_tokens=False,
    )
    candidates = [root]
    while candidates:
        node = candidates.pop(0)
        text = node.text
        tokens = tokenizer.tokenize(text)
        node.token_num = len(tokens)
        if node.children:
            for child in node.children:
                child.pre_token_num = node.pre_token_num + node.token_num
                candidates.append(child)


def draw_tree_pic(root: Type[InferNode], file_name):
        output_image_path = file_name
        candidates = [root]
        states = {}
        dot = Digraph(comment='MCTS Tree')
        dot.attr(rankdir='500')
        while candidates:
            node = candidates.pop(0)
            node_label = f"Tag: {node.tag}\nValue: {node.value}\nQ: {node.q_value}\nP: {node.prior}\nVisits: {node.visit_count}\nTotal FLOPS: {node.total_flops:.2e}\n"
            dot.node(node.tag, node_label)
            if node.children:
                for child in node.children:
                    dot.edge(node.tag, child.tag)
                candidates.extend(node.children)
        dot.render(output_image_path, format="png", cleanup=True)
    
def calculate_tree_flops(root: Type[InferNode]):
    candidates = [root]
    total_flops = 0
    linear_flops = 0
    attention_flops = 0
    while candidates:
        node = candidates.pop(0)
        if node.children:
            for child in node.children:
                candidates.append(child)
                tmp_flops, tmp_linear_flops, tmp_attention_flops = calculate_decode_flops(child.pre_token_num, child.token_num)
                node.total_flops += tmp_flops
                node.linear_flops += tmp_linear_flops
                node.attention_flops += tmp_attention_flops
        total_flops += node.total_flops
        linear_flops += node.linear_flops
        attention_flops += node.attention_flops
    return total_flops, linear_flops, attention_flops
            
        
            
def calculate_decode_flops( prefill_length: int, seq_length: int) -> float:
    hidden_size = 3584
    num_hidden_layers = 28
    vocab_size = 151680
    flops_per_forward = 0
    linear_flops = 0
    attention_flops = 0
    # 估算每次前向传播的 FLOPS 数量
    for i in range(seq_length):
        flops_per_forward += num_hidden_layers * (24 * hidden_size ** 2 + 4 * (prefill_length + i) * hidden_size) + 2 * hidden_size * vocab_size
        linear_flops += num_hidden_layers * (24 * hidden_size ** 2)
        attention_flops += num_hidden_layers * (4 * (prefill_length + i) * hidden_size)

    return flops_per_forward, linear_flops, attention_flops


if __name__ == '__main__':
    cal_tree()

    """
    args = parse_args()

    config = OmegaConf.structured(BaseConfig)
    if args.custom_cfg:
        custom_config = OmegaConf.load(args.custom_cfg)
        config = OmegaConf.merge(config, custom_config)
    config = OmegaConf.create(OmegaConf.to_yaml(config, resolve=True))
    print(config)


    cnt, total = 0, 0
    with open(args.tree_jsonl, "r") as f:
        for line in tqdm(f):
            full_tree_dict = json.loads(line)
            solution = get_solution(
                full_tree_dict,
                prune=config.prune,
                b1=config.step_beam_width,
                b2=config.n_generate_sample,
                strategy=config.mcts_infer_strategy,            
            )

            if solution and math_is_equiv(solution["ground_truth"], solution["final_answer"]):
                cnt += 1
            total += 1

    print(cnt, total, f"Accuracy: {cnt / total}")
    """