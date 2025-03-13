"""
author: lmp-decaderan
email: ldecaderan@gmail.com

reviewed: lovecambi
email: interfk@gmail.com
"""
from __future__ import annotations

import os
import random
import torch
import numpy as np
from graphviz import Digraph
from termcolor import colored
from typing import Dict, Any, Optional, Type, List, Tuple, Callable, Union
from functools import partial
from pydantic import field_validator

from vllm.outputs import CompletionOutput, RequestOutput

from mcts_math.agents.utils import math_is_equiv as is_equiv

from mcts_math.nodes import MCTSNode
from mcts_math.constants import (
    TOO_MANY_CODE_ERRORS, 
    TOO_MANY_STEPS, 
    NO_VALID_CHILD, 
    SOLUTION_COLOR, 
    OBSERVATION_COLOR,
    WARNING_COLOR,
)

from .tree import BaseTree, code_execution
from .step_beam import SBSREACT

def calculate_prefill_flops(seq_length: int) -> float:
    hidden_size = 4096
    num_hidden_layers = 30
    vocab_size = 102400

    # 估算每次前向传播的 FLOPS 数量
    flops_per_token = num_hidden_layers * (24 * hidden_size ** 2 + 4 * seq_length * hidden_size) + 2 * hidden_size * vocab_size
    flops_per_forward = flops_per_token * seq_length
    
    return flops_per_forward

def calculate_decode_flops( prefill_length: int, seq_length: int) -> float:
    hidden_size = 4096
    num_hidden_layers = 30
    vocab_size = 102400

    # 估算每次前向传播的 FLOPS 数量
    flops_per_token = num_hidden_layers * (24 * hidden_size ** 2 + 4 * (prefill_length + seq_length) * hidden_size) + 2 * hidden_size * vocab_size
    flops_per_forward = flops_per_token * seq_length
    return flops_per_forward


def calculate_mfu(seq_length: int,time: float) -> float:
    batch_size = 1
    hidden_size = 4096
    num_hidden_layers = 30
    vocab_size = 102400

    # 估算每次前向传播的 FLOPS 数量
    flops_per_token = num_hidden_layers * (24 * hidden_size ** 2 + 4 * seq_length * hidden_size) + 2 * hidden_size * vocab_size
    flops_per_forward = flops_per_token * seq_length
    
    # 估算每次迭代的 FLOPS 数量
    flops_per_iter = flops_per_forward * batch_size
    #7B
    #flops_per_iter = 7e9 * 34 * 2
    # A100 GPU 的峰值 FLOPS（以 bfloat16 为单位）
    a100_peak_flops = 312e12  # 312 TFLOPS

    # 计算 MFU
    mfu = flops_per_iter / (a100_peak_flops * time)
    return mfu

class MCTS(SBSREACT):

    @field_validator("config")
    def validate_config(cls, cfg: Any):
        BaseTree.validate_config(cfg)
        if not cfg.mode == "mcts":
            raise ValueError(f"Wrong value for config mode.")
        if cfg.stop is None:
            raise ValueError(f"Wrong value for config stop, cannot be None")
        return cfg

    def create_node(self, parent: Optional[Type[MCTSNode]] = None) -> Type[MCTSNode]:
        return MCTSNode(
            parent=parent, 
            additional_state_keys=self.REACT_NODE_KEYS,
            c_puct=self.config.c_puct,
        )

    def generate(self) -> None:
        print
        self.search()

    @torch.inference_mode()
    def search(self) -> None:
        for idx in range(self.config.iterations):
            # node selection starting from root
            node = self.selection()
            # expansion_evaluation_backpropagation
            self.expansion_evaluation_backpropagation(node)

    def selection(self) -> Optional[Type[MCTSNode]]:
        node = self.root
        while node.has_children() or node.is_terminal:
            next_node = self.select_child(node)     # To encourage exploration, select from non-terminal children
            if next_node is None:                   # if None，it mean all children are terminal
                node.is_terminal = True
                break
            node = next_node
    
        return None if node.is_terminal else node

    def select_child(self, node: Type[MCTSNode]) -> Optional[Type[MCTSNode]]:
        # TODO: implement multi-strategy
        # select the best child according to the puct
        best_value = -float("inf")
        best_childs = []

        for child in node.children:
            if child.is_terminal:
                continue

            puct_value = child.puct()
            if puct_value == best_value:
                best_childs.append(child)
            elif puct_value > best_value:
                best_value = puct_value
                best_childs = [child]

        return random.choice(best_childs) if best_childs else None

    def expansion_evaluation_backpropagation(self, node: Type[MCTSNode]) -> None:
        """
        This function is only used for single example inference, required to set `create_local_llm` as True.
        """
        assert self.config.create_local_llm, "llm must be created within MCTS class."
        prompt = self.create_prompt()
        # expand and evaluate
        outputs, value_estimate = self.llm(prompt, n=self.n_generate_sample, stop=self.stop)
        #输出得到的outputs和value_estimate到终端
        #print(f"outputs: {outputs}\n")
        print(f"value_estimate: {value_estimate}\n")
        print("here is ok\n")
        if value_estimate is not None:  # input exceeds 4096, output '' and None
            self.expand_node(outputs, node)
        else:
            value_estimate = self.config.negative_reward
            node.is_terminal = True
        # backup
        node.update_recursive(value_estimate, self.root)
        #添加
        self.add_value_to_sglang(node)

    #添加,用于处理value相关
    def add_value_to_sglang(self, node: Type[MCTSNode]) -> None:
        if node.is_terminal:#如果是终止节点，value应当为0或者负数
            input_value = 0
        else:
            input_value = node.puct()
        #收集partial_solution
        partial_solution = self.collect_partial_solution(node)
        #创建prompt
        prompt = self.prompt_wrap(
            self.question,
            partial_solution,
            self.config,
        )
        #将prompt和input_value传入llm
        outputs, value_estimate = self.llm(prompt, with_value=True, input_value=input_value)
        #print(f"outputs: {outputs}\n")

    def expand_node(self, outputs: List[CompletionOutput], node: Type[MCTSNode]) -> int:
        #print("This is expand node")

        children_length = 0
        if self.config.remove_duplicate:#去重
            dedup_outputs = []
            dedup_keys = set()
            for output in outputs:
                key = output.text.strip()
                if not key in dedup_keys:
                    dedup_keys.add(key)
                    dedup_outputs.append(output)
            outputs = dedup_outputs
            #打印输出
            #print(f"dedup_outputs: {dedup_outputs}")
            print("here ded is ok")
        for idx, output in enumerate(outputs):
            #print("This is expand node loop")
            #这里出现了问题导致出错
            try:
            # 这里出现了问题导致出错
              if len(output.token_ids) == 0:
                prior_prob = 0.0
              else:
                prior_prob = np.exp(output.cumulative_logprob / len(output.token_ids))
              #print(f"output.cumulative_logprob: {output.cumulative_logprob}\n")
              #print(f"len(output.token_ids): {len(output.token_ids)}\n")
              #print(f"prior_prob: {prior_prob}\n")
              
            except Exception as e:
              print(f"Error calculating prior_prob: {e}")
              continue  # 跳过当前循环，继续下一个循环
            step_result, parser_result = self.step_unwrap(output.text.strip())
            #print(f"step_result: {step_result}\n")
            #print(f"parser_result: {parser_result}\n")
            #print(f"step_result: {step_result}\n")
            #计算token_ids的长度
            token_ids_len = len(output.token_ids)
            children_length += token_ids_len
            #step_result是text，parser_result是解析的结果
            #parser_result = { 
            #"action": "",
            #"action_input": "",
            #"final_answer": "",
            #}
            #print("before create_child")
            self.create_child(step_result, parser_result, node, prior_prob, idx, token_ids_len)
        #打印node的children
        #print(f"node.children: {node.children}\n")
        return children_length

    def create_child(
        self, 
        step_result: str, 
        parser_result: Dict[str, str], 
        node: Type[MCTSNode],
        prior_prob: float,
        idx: int,
        token_ids_len: int = 0,
    ) -> None:
        #print("This is create child")
        filename = "/workspace/Super_MARIO/mcts_math/agents/mcts_output.txt"
        if self.config.verbose:
            print(colored(f"{step_result}\n", SOLUTION_COLOR))
        #print到文件中
        #with open(filename, "a") as f:
        #    f.write(f"{step_result}\n")

        # initialize a new node
        new_node = self.create_node(parent=node)
        new_node.tag = f"{node.tag}.{idx}"
        new_node.depth = node.depth + 1
        new_node.prior = prior_prob
        new_node.token_ids_len = token_ids_len
        # update node state
        if parser_result is None:
            #print("This is parser_result is None")
            new_node.is_terminal = True
            new_node.state["text"] = step_result
            new_node.state["final_answer"] = NO_VALID_CHILD
            self.eval_final_answer(new_node)
        elif parser_result["final_answer"]:
            #print("This is final_answer")
            new_node.is_terminal = True
            new_node.state["text"] = step_result
            new_node.state["final_answer"] = parser_result["final_answer"]
            self.eval_final_answer(new_node)
        elif parser_result["action"]:
            #print("This is action")
            observation = code_execution(node, parser_result)
            observation = self.obs_wrap(observation)

            if self.config.verbose:
                print(colored(f"{observation}\n", OBSERVATION_COLOR))
                #print到文件中
                with open(filename, "a") as f:
                    f.write(f"{observation}\n")

            new_node.state["text"] = f"{step_result}{self.config.step_delim}{observation}"
            new_node.state["action"] = parser_result["action"]
            new_node.state["action_input"] = parser_result["action_input"]

            if "Error" in observation:
                new_node.consecutive_errors = node.consecutive_errors + 1
                if new_node.consecutive_errors > self.config.errors_threshold:
                    new_node.is_terminal = True
                    new_node.state["final_answer"] = TOO_MANY_CODE_ERRORS
                    self.eval_final_answer(new_node)
        else:
            if self.config.verbose:
                print(colored(f"WARNING: '{step_result}' Cannot resolve\n", WARNING_COLOR))
            new_node.state["text"] = step_result

        if not new_node.is_terminal and new_node.depth > self.config.max_depth:
            new_node.is_terminal = True
            new_node.state["final_answer"] = TOO_MANY_STEPS
            self.eval_final_answer(new_node)

        #print(f"new_node: {new_node}\n")
        node.children.append(new_node)
        #print("This is create child ok")

    def eval_final_answer(self, node: Type[MCTSNode]) -> None:
        if node.state["final_answer"] in [NO_VALID_CHILD, TOO_MANY_STEPS, TOO_MANY_CODE_ERRORS]:
            node.update_recursive(self.config.negative_reward, self.root)

            #添加，传入value
            #self.add_value_to_sglang(node)
            return 
        
        if self.ground_truth:
            final_answer = node.state["final_answer"]
            correct = is_equiv(self.ground_truth, final_answer)
            # backup
            node.update_recursive(self.config.positive_reward if correct else self.config.negative_reward, self.root)
            #添加，传入value
            #self.add_value_to_sglang(node)
        else:
            # for testset, no ground_truth, put this node in candidate_nodes, then it will be evaluated by value model and backup in select_next_step().
            self.candidate_nodes.append(node)

    def select_next_step(self, outputs: Optional[List[RequestOutput]] = None) -> None:
        """process output from vllm
        e.g.,
        prompts = tree.create_prompt(is_value_only=True)
        outputs = llm.generate(prompts, sampling_params)
        for output in outputs:
            step_generate(output)
        """
        self.current_nodes = []
        if outputs is not None:
            #print(f"outputs: {outputs}\n")
            for candidate_node, output in zip(self.candidate_nodes, outputs):
                # assert self.question in output.prompt
                # backup
                if candidate_node.is_terminal and self.ground_truth:
                    continue
                #修改
                #random_number = np.random.rand()#添加，用来凑数
                #value_estimate = output.value_estimate if output.value_estimate is not None else self.config.negative_reward
                value_estimate = 0
                #print(f"value_estimate: {value_estimate}\n")
                #value_estimate = 0
                #if output.value_estimate is None:
                #    candidate_node.is_terminal = True
                #    print("here is None\n")
                candidate_node.update_recursive(value_estimate, self.root)
                #添加，传入value
                #self.add_value_to_sglang(candidate_node)
                if self.__class__.is_valid_final_answer_node(candidate_node):
                    self.final_answer_nodes.append(candidate_node)
        selection_node = self.selection()
        #print(f"selection_node: {selection_node}\n")
        if selection_node is not None:
            self.current_nodes.append(selection_node)
    
    def generate_next_step(self, outputs: List[RequestOutput]) -> None:
        """process output from vllm
        e.g.,

        outputs = llm.generate(prompts, sampling_params)
        for output in outputs:
            step_generate(output)
        """
        #print("here is generate_next_step")
        self.candidate_nodes = []
        for current_node, output in zip(self.current_nodes, outputs):
            # assert self.question in output.prompt
            # current_step.value = output.value
            # expand n_generate_sample nodes
            #print(f"output: {output}")
            #解析metrics，得到所花费的时间
            #generate_time = 0
            #generate_time = output.metrics.finished_time - output.metrics.first_scheduled_time
            sequence_length = 0
            

            ##解析Output，得到prompt_token_ids的长度，并赋给root.token_ids_len
            if current_node == self.root:
                self.root.token_ids_len = len(output.prompt_token_ids)
            #print("here is OK\n")
            sequence_length += len(output.prompt_token_ids)
            #print("here is loop1")
            #print(f"is_none: {is_none}\n")
            #print(f"output_value_estimate: {output.value_estimate}")

            

            #value_estimate = output.value_estimate#这里到时候改回来
            #print(f"generate value_estimate: {value_estimate}\n")
            #用np的random函数生成一个0-1之间的随机数
            
            seed_value = current_node.token_ids_len
            random_generator = np.random.default_rng(seed_value)
            value_estimate = random_generator.random()
            #print(f"value_estimate: {value_estimate}\n")

            #打印value_estimate
            #print(f"value_estimate: {value_estimate}\n")
            #print("here is loop2")
            if value_estimate is not None:  # input exceeds 4096, output '' and None修改添加
            #if not is_none or current_node == self.root:
                #print("here is loop3")
                current_node.value = value_estimate
                
                #print(f"current_node: {current_node}\n")
                #print(f"curren_node.value: {current_node.value}\n")
                #print("begin to expand node\n")
                sequence_length += self.expand_node(output.outputs, current_node)
            else:
                #print("here is loop4")
                value_estimate = self.config.negative_reward
                current_node.is_terminal = True
            # self.expand_node(output.outputs, current_node)
            # self.candidate_nodes.extend(current_node.children)
            current_node.sequence_len = sequence_length
            # 计算current_node的MFU
            
            #mfu = calculate_mfu(sequence_length, generate_time)
            #current_node.mfu = mfu    
            #current_node.generate_time = generate_time
            # backup
            if self.config.update_leaf_value:
                # child node will be put into candidate_nodes, then all candidate_nodes will be evaluated by value model and backup in select_next_step().
                for value_node in current_node.children:
                    if value_node not in self.candidate_nodes and value_node.visit_count() < 1:
                        self.candidate_nodes.append(value_node)
            else:#如果self.config.update_leaf_value为False，说明不用value model？
                current_node.update_recursive(value_estimate, self.root)
                #添加，传入value
                #self.add_value_to_sglang(current_node)
                #递归更新，更新visit_count和q_value，不过这里为啥用的是value_estimate？

    def return_states(self, file_name) -> Dict[str, Union[Any, Dict[str, str]]]:
        output_image_path = file_name
        candidates = [self.root]
        states = {}
        dot = Digraph(comment='MCTS Tree')
        dot.attr(rankdir='500')
        while candidates:
            node = candidates.pop(0)
            states[node.tag] = node.state
            states[node.tag]["value"] = node.value
            states[node.tag]["q_value"] = node.q_value()
            states[node.tag]["prior"] = node.prior
            states[node.tag]["visit_count"] = node.visit_count()
            states[node.tag]['is_terminal'] = node.is_terminal
            states[node.tag]['token_ids_len'] = node.token_ids_len
            node_label = f"Tag: {node.tag}\nValue: {node.value}\nQ: {states[node.tag]['q_value']}\nP: {states[node.tag]['prior']}\nVisits: {states[node.tag]['visit_count']}\n Terminal: {states[node.tag]['is_terminal']}\nToken_ids_len: {states[node.tag]['token_ids_len']}\n"
            dot.node(node.tag, node_label)
            if node.has_children():
                for child in node.children:
                    dot.edge(node.tag, child.tag)
                candidates.extend(node.children)
        dot.render(output_image_path, format="png", cleanup=True)
        return states