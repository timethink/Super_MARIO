"""
author: lmp-decaderan
email: ldecaderan@gmail.com
"""
from __future__ import annotations

import os
import copy
import re
import random
import torch
import numpy as np
import json
import time
from datetime import datetime
from termcolor import colored
from functools import partial
from typing import Optional, Any, Dict, List, Callable, Type, Tuple
from tqdm import tqdm
from abc import abstractmethod
from pydantic import BaseModel, ConfigDict, field_validator
from omegaconf import DictConfig, OmegaConf

from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from .agents.tree import BaseTree
import matplotlib.pyplot as plt
from .llms.local_llms import local_generator, server_generator
from .llms.local_llm_engine import llm_engine
from .constants import TIMEOUT_SECONDS, ERROR_COLOR

def set_seed(seed: int = 1024) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    # logger.info(f"Random seed set as {seed}")

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
    flops_per_forward = 0
    # 估算每次前向传播的 FLOPS 数量
    for i in range(seq_length):
        flops_per_forward += num_hidden_layers * (24 * hidden_size ** 2 + 4 * (prefill_length + i) * hidden_size) + 2 * hidden_size * vocab_size
    return flops_per_forward


class Solver(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: Any

    stop: List[str] = None

    llm: Optional[Callable[[...], List[str]]] = None

    llm_model_id: Optional[str] = None 
    engine: Optional[LLM] = None
    generate_sampling_params: Optional[SamplingParams] = None
    value_sampling_params: Optional[SamplingParams] = None
    need_value_func: bool = False
    max_solver_steps: int = 1

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.llm_model_id = self.config.model_dir

        if self.config.stop:#规定停止条件,
            #如stop: ["\nObservation:", "Observation:", "<Solution complete>"]
            # omegaconf.listconfig.ListConfig -> list
            self.stop = OmegaConf.to_object(self.config.stop)

        self.llm = self.create_llm()#创建一个llm模型
        # create_llm()方法返回一个函数，该函数用于生成输出
        #这里函数是绑定了engine的local_generator
        self.need_value_func = self.config.need_value_func

        if self.config.mode == "sbs":
            self.max_solver_steps = self.config.max_fpdepth
        elif self.config.mode == "mcts":
            self.max_solver_steps = self.config.iterations
            self.config.step_beam_width = 1

    @field_validator("config")
    def validate_config(cls, cfg: Any):
        if issubclass(type(cfg), DictConfig):
            return cfg

        raise TypeError("Wrong type for `config`, must be subclass of BaseConfig")

    def create_llm(self) -> Callable[[...], List[str]]:
        if self.config.seed:
            set_seed(self.config.seed)
            print(f"Set seed as {self.config.seed}")
        engine, sampling_params = llm_engine(self.config)
        self.engine = engine
        self.generate_sampling_params = sampling_params
        self.value_sampling_params = copy.deepcopy(sampling_params)
        self.value_sampling_params.max_tokens = 1
        self.value_sampling_params.n = 1
        return partial(
            local_generator,#local_generator是产生输出的函数
            #参见local_llms.py
            #outputs = engine.generate(prompts, sampling_params=sampling_params)    # return List[RequestOutput]
            engine=self.engine,
        )
        
    @staticmethod
    def processor(solver: BaseTree, output: List[RequestOutput]) -> BaseTree:
        solver.generate_next_step(output)
        return solver
    
    @staticmethod
    def selector(solver: BaseTree, output: RequestOutput) -> BaseTree:
        solver.select_next_step(output)
        return solver

    def calculate_output_lengths(self, outputs: List[RequestOutput]):
        outputs_length = []
        for output in outputs:
            completion_outputs = output.outputs
            for completion_output in completion_outputs:
                token_ids = completion_output.token_ids
                outputs_length.append(len(token_ids))
        avg_length = sum(outputs_length) / len(outputs_length)
        return outputs_length, avg_length
    
        

    def generate_preprocess(self, solvers: List[BaseTree]) -> Tuple[List[str], List[int], List[BaseTree], List[BaseTree]]:
        #方法主要用于生成前的准备工作。它会检查每个 solver 对象，判断是否需要生成下一步，并根据判断结果将 solver 分类为有效或无效。
        prompts = []
        prompts_span = [0]
        valid_solvers = []
        invalid_solvers = []

        for solver in solvers:
            #这里的solver相当于一个agent，包含一个MCTS对象，包括question，ground_truth,current_nodes,candidate_nodes等信息
            if solver.should_generate_next():#判断是否需要生成下一步,在step_beam.py中实现
            #遍历current_nodes中的每个node，如果node不是终止节点且深度小于最大深度，则返回True
                solver_prompts = solver.create_prompt()
                prompts.extend(solver_prompts)
                prompts_span.append(prompts_span[-1] + len(solver_prompts))
                valid_solvers.append(solver)
            else:
                invalid_solvers.append(solver)
        return prompts, prompts_span, valid_solvers, invalid_solvers

    def generate_postprocess(
        self, 
        outputs: List[List[RequestOutput]], 
        valid_solvers: List[BaseTree],
        step: int
    ) -> List[BaseTree]:
        #print("here is generate_postprocess")
        #generate_postprocess 用于对模型的生成结果进行后处理。它会根据 outputs 更新每个 solver，并确保生成步骤在并行池中安全运行。
        post_solvers = []
        tmp_step = step
        #使用Pebble库的 ProcessPool 类来并行处理所有有效的 solver 对象。
        with ProcessPool(max_workers=min(len(valid_solvers), os.cpu_count())) as pool:
            future = pool.map(self.__class__.processor, valid_solvers, outputs, timeout=TIMEOUT_SECONDS)
            #processor方法在solver.py中定义，用于处理生成的输出
            #这里将valid_solvers和outputs传入processor方法中
            iterator = future.result()#返回一个迭代器，迭代器的每个元素是processor方法的返回值
        
        if len(valid_solvers) > 100:  
            progress_bar = tqdm(total=len(valid_solvers), desc="Execute")  
        else:  
            progress_bar = None 

        while True:
            try:
                result = next(iterator)
                post_solvers.append(result)
            except StopIteration:
                break
            except TimeoutError as error:
                post_solvers.append(None)
                if self.config.verbose:
                    print(colored(f"{error}\n", ERROR_COLOR))
            except SystemExit as error:
                post_solvers.append(None)
                if self.config.verbose:
                    print(colored(f"{error}\n", ERROR_COLOR))
            except Exception as error:
                if self.config.verbose:
                    print(colored(f"{error}\n", ERROR_COLOR))
                post_solvers.append(None)
            if progress_bar is not None:
                progress_bar.update(1) 
        
        if progress_bar is not None:
            progress_bar.close() 
        #打印post_solvers
        #print(post_solvers)
        #post_solvers_file = f"/workspace/MARIO_EVAL/data/step_{tmp_step}_post_generate_solvers.json"
        #with open(post_solvers_file, "w") as f:
        #    f.write(str(post_solvers))
        #    f.flush()
        # update solvers
        assert len(valid_solvers) == len(post_solvers), f"Data is not matched, {len(valid_solvers)} vs {len(post_solvers)}."
        updated_solvers = [
            post_solver if post_solver is not None else valid_solver
            for post_solver, valid_solver in zip(post_solvers, valid_solvers)
        ]
        return updated_solvers
    
    def value_preprocess(self, solvers: List[BaseTree]) -> Tuple[List[str], List[int]]:
        #value_preprocess 方法在价值评估阶段的预处理过程中生成提示信息，用于引导模型对生成的解答进行评估。
        prompts = []
        prompts_span = [0]

        for solver in solvers:
            solver_prompts = solver.create_prompt(is_value_only=True)
            prompts.extend(solver_prompts)
            prompts_span.append(prompts_span[-1] + len(solver_prompts))
        return prompts, prompts_span
    #返回值：
    #prompts：包含所有生成的提示信息。
    #prompts_span：记录各提示的起始和终止索引。
    
    def value_postprocess(
        self, 
        outputs: List[List[RequestOutput]], 
        valid_solvers: List[BaseTree],
    ) -> List[BaseTree]:
        #value_postprocess 方法用于处理价值评估阶段的输出结果，将生成的价值评估输出更新到每个 solver 中。
        for solver, output in zip(valid_solvers, outputs):
            if solver is not None:
                self.selector(solver, output)
        return valid_solvers
    
    def postprocess(
        self, 
        valid_solvers: List[BaseTree], 
        invalid_solvers: List[BaseTree],
    ) -> List[BaseTree]:

        # update solvers
        invalid_solvers.extend(valid_solvers)#将valid_solvers中的元素添加到invalid_solvers中，表示这些solver已经处理完毕
        return invalid_solvers


    def solve(self, solvers: List[BaseTree]):
        final_step = []
        final_seq_len = []
        final_prefill_len = []
        final_decode_len = []
        final_pre_mfu = []
        final_nopre_mfu = []
        final_time = []
        final_unfinished = []
        for step in tqdm(range(self.max_solver_steps), desc="Step Processing"):
            #filename1 = f"/workspace/MARIO_EVAL/data/step_{step}_pre_solvers.json"
            prompts, prompts_span, valid_solvers, invalid_solvers = self.generate_preprocess(solvers)
            #在mcts中，这里的solvers是agents，每个agent包含一个MCTS对象，包括question，ground_truth,current_nodes,candidate_nodes等信息
            #将prompts,prompts_span,valid_solvers,invalid_solvers保存到文件中
            #将处理前的tree打印出来
            """
            tree_filename1 = f"/workspace/MARIO_EVAL/data/runtime_tree/step_{step}_pre_tree.json"
            tree_pic_name =  f"/workspace/MARIO_EVAL/data/pic_tree/step_{step}_pre_tree"
            jsonlines1 = {}
            for i, valid_solver in enumerate(valid_solvers):         
                jsonlines1[valid_solver.question] = valid_solver.return_states(tree_pic_name)
            #格式化jsonlines
            jsonlines1 = json.dumps(jsonlines1, indent=4)
            with open(tree_filename1, "w") as f:
                f.write(str(jsonlines1))
            """
            

            if len(valid_solvers) < 1:
                break
            
            # llm run for step generation
            if step == 0:
                n = self.config.n_generate_sample * self.config.step_beam_width
            else:
                n = self.config.n_generate_sample
            self.generate_sampling_params.n = n
            self.generate_sampling_params.best_of = n
            
            #将prompts保存到文件中
            
            prompt_filename1 = f"/workspace/MARIO_EVAL/data/runtime_prompt/step_{step}_pre_generate_prompts.json"
            with open(prompt_filename1, "w") as f:
                f.write(str(prompts))
                f.write("\n")
            
          
          #计算mfu
            batch_size = self.config.batch_size
            beam_width = n

            start_time = time.time()
                
            outputs = self.llm(prompts, self.generate_sampling_params)

            end_time = time.time()
            #计算outputs中的prompt长度和以及CompletionOutput的token_ids长度和
            request_num = len(outputs)
            prompt_len = 0
            decode_len = 0
            prefill_len_sum = 0
            decode_len_sum = 0
            seq_len = 0
            pre_flops_sum = 0
            nopre_flops_sum = 0
            for output in outputs:
                prompt_len = len(output.prompt)
                seq_len += len(output.prompt)
                prefill_len_sum += prompt_len
                prefill_flops = calculate_prefill_flops(prompt_len)
                #prefill_time = output.metrics.first_token_time - output.metrics.first_scheduled_time
                pre_flops_sum += prefill_flops
                for completion_output in output.outputs:
                    decode_len =  len(completion_output.token_ids)
                    seq_len += decode_len
                    decode_len_sum += decode_len
                    decode_flops= calculate_decode_flops(prompt_len, decode_len)   
                    pre_flops_sum += decode_flops
                    nopre_flops_sum += decode_flops
                #decode_time = output.metrics.finished_time - output.metrics.first_token_time
                
            #计算mfu
            mfu_time = end_time - start_time
            A100_flops = 312 * 10 ** 12
            percentage = len(valid_solvers) / len(solvers)
            pre_mfu = pre_flops_sum / (A100_flops * mfu_time)
            nopre_mfu = nopre_flops_sum / (A100_flops * mfu_time)
            average_prefill_len = prefill_len_sum / request_num
            average_decode_len = decode_len_sum / request_num
            final_step.append(step)
            final_seq_len.append(seq_len)
            final_prefill_len.append(average_prefill_len)
            final_decode_len.append(average_decode_len)
            final_pre_mfu.append(pre_mfu)
            final_nopre_mfu.append(nopre_mfu)
            final_time.append(mfu_time)
            final_unfinished.append(percentage)
            #将batch_size,seq_len,time,mfu,percentage保存到文件中
            """
            mfu_filename = f"/workspace/MARIO_EVAL/data/runtime_mfu/step_{step}_mfu.json"
            with open(mfu_filename, "w") as f:
                f.write(str(batch_size))
                f.write("\n")
                f.write(str(seq_len))
                f.write("\n")
                f.write(str(mfu_time))
                f.write("\n")
                f.write(str(mfu))
                f.write("\n")  
                f.write(str(percentage))
                f.write("\n")
            """
            #将初始outputs保存到文件中
            filename2 = f"/workspace/MARIO_EVAL/data/runtime_output/step_{step}_generate_outputs.json"
            with open(filename2, "w") as f:
                f.write(str(outputs))
                f.write("\n")
            
            """CompletionOutput(index=0, text='<step>\n<p>\nFrom the result, we can see that the vertical asymptotes of the graph of $y=\\frac{2}{x^2+x-6}$ are at $x=-3$ and $x=2$.\n</p>\n<p>\nFinal Answer: $2$\n</p>\n', token_ids=[27, 9215, 29, 185, 27, 79, 29, 185, 4044, 254, 1230, 11, 395, 481, 1019, 344, 254, 10796, 16534, 5671, 280, 254, 4150, 280, 363, 88, 1928, 1122, 90, 17, 1061, 87, 61, 17, 10, 87, 12, 21, 759, 418, 430, 363, 87, 10196, 18, 3, 285, 363, 87, 28, 17, 1332, 185, 535, 79, 29, 185, 27, 79, 29, 185, 19275, 35829, 25, 363, 17, 3, 185, 535, 79, 29, 185, 535, 9215, 29], cumulative_logprob=-0.989820027285532, logprobs=None, finish_reason=stop), CompletionOutput"""
            
            # post-process outputs
            #将生成的模型输出 outputs 切分为多个子列表，以便与各个提示 prompts 对应。
            #每个子列表对应一个提示及其相关的生成。
            reconstructed_outputs = [outputs[bos_idx : eos_idx] for bos_idx, eos_idx in zip(prompts_span, prompts_span[1:])]
            #将reconstructed_outputs保存到文件中
            #filename3 = f"/workspace/MARIO_EVAL/data/step_{step}_reconstructed_outputs.json"
            #with open(filename3, "w") as f:
            #    f.write(str(reconstructed_outputs))
            #    f.write("\n")
            #print(reconstructed_outputs)
            #这里生成了题目的解答，接下来需要对解答进行处理
            # process output and run python interpreter
            valid_solvers = self.generate_postprocess(reconstructed_outputs, valid_solvers, step)
            #将valid_solvers保存到文件中
            #filename4 = f"/workspace/MARIO_EVAL/data/step_{step}_valid_solvers.json"
            #with open(filename4, "w") as f:
            #    f.write(str(valid_solvers))
            #    f.write("\n")
            #将处理后的tree打印出来
            """
            tree_filename2 = f"/workspace/MARIO_EVAL/data/runtime_tree/step_{step}_post_tree.json"
            tree_pic_name2 =  f"/workspace/MARIO_EVAL/data/pic_tree/step_{step}_post_tree"
            jsonlines2 = {}
            for i, valid_solver in enumerate(valid_solvers):
                jsonlines2[valid_solver.question] = valid_solver.return_states(tree_pic_name2)
            #格式化jsonlines
            jsonlines2 = json.dumps(jsonlines2, indent=4)

            with open(tree_filename2, "w") as f:
                f.write(str(jsonlines2))
            """

            # llm run for step evaluation
            prompts, prompts_span = self.value_preprocess(valid_solvers)
            #将prompts和prompts_span保存到文件中
            #奇怪，这里为什么prompts和prompts_span是空的？
            """
            filename5 = f"/workspace/MARIO_EVAL/data/step_{step}_prevalue_prompts.json"
            with open(filename5, "w") as f:
                f.write("prompts: \n")
                f.write(str(prompts))
                f.write("\n")
                f.write("prompts_span: \n")
                f.write(str(prompts_span))
                f.write("\n")
            """


            if self.need_value_func:#是否用到value head
                #将prompts保存到文件中
                """
                prompt_filename2 = f"/workspace/MARIO_EVAL/data/runtime_prompt/step_{step}_pre_value_prompts.json"
                with open(prompt_filename2, "w") as f:
                    f.write(str(prompts))
                    f.write("\n")
                """
                outputs = self.llm(prompts, self.value_sampling_params)
                #将初始outputs保存到文件中
                """
                filename6 = f"/workspace/MARIO_EVAL/data/runtime_output/step_{step}_value_outputs.json"
                with open(filename6, "w") as f:
                    f.write(str(outputs))
                    f.write("\n")
                """
                reconstructed_outputs = [outputs[bos_idx : eos_idx] for bos_idx, eos_idx in zip(prompts_span, prompts_span[1:])]
            else:
                reconstructed_outputs = [None] * (len(prompts_span) - 1)
            #将reconstructed_outputs保存到文件中
            valid_solvers = self.value_postprocess(reconstructed_outputs, valid_solvers)
            #将valid_solvers保存到文件中
            #filename5 = f"/workspace/MARIO_EVAL/data/step_{step}_after_valid_solvers.json"
            #with open(filename5, "w") as f:
            #    f.write(str(valid_solvers))
            #    f.write("\n")
            #将处理后的tree打印出来
            """
            tree_filename3 = f"/workspace/MARIO_EVAL/data/runtime_tree/step_{step}_postvalue_tree.json"
            tree_pic_name3 =  f"/workspace/MARIO_EVAL/data/pic_tree/step_{step}_postvalue_tree"
            jsonlines3 = {}
            for i, valid_solver in enumerate(valid_solvers):         
                jsonlines3[valid_solver.question] = valid_solver.return_states(tree_pic_name3)
            #格式化jsonlines
            jsonlines3 = json.dumps(jsonlines3, indent=4)
            with open(tree_filename3, "w") as f:
                f.write(str(jsonlines3))
            """

            solvers = self.postprocess(valid_solvers, invalid_solvers)
            
            #将solvers的return_states()保存到文件中
            """
            filename4 = f"/workspace/MARIO_EVAL/data/step_{step}_solvers_return_states.json"
            tree_pic_name4 =  f"/workspace/MARIO_EVAL/data/pic_tree/step_{step}_solvers_return_states"
            tmp_jsonlines = {}
            for i, solver in enumerate(solvers):         
                tmp_jsonlines[solver.question] = solver.return_states(tree_pic_name4)
            #格式化jsonlines
            tmp_jsonlines = json.dumps(tmp_jsonlines, indent=4)
            with open(filename4, "w") as f:
                f.write(str(tmp_jsonlines))
            """
            """
            #将处理后的tree打印出来
            tree_filename4 = f"/workspace/MARIO_EVAL/data/runtime_tree/step_{step}_finish_tree.json"
            tree_pic_name4 =  f"/workspace/MARIO_EVAL/data/pic_tree/step_{step}_finish_tree"
            jsonlines4 = {}
            for i, solver in enumerate(solvers):         
                jsonlines4[solver.question] = solver.return_states(tree_pic_name4)
            #格式化jsonlines
            jsonlines4 = json.dumps(jsonlines4, indent=4)
            with open(tree_filename4, "w") as f:
                f.write(str(jsonlines4))
            """
            """
              #记录时间
            mfu_filename = f"/workspace/MARIO_EVAL/data/runtime_mfu/final_mfu.json"
            with open(mfu_filename, "w") as f:
                f.write(str(final_seq_len))
                f.write("\n")
                f.write(str(final_mfu))
                f.write("\n")
         
            #单独画出final_mfu和final_seq_len随step的变化图
            mfu_step_pic_filename = f"/workspace/MARIO_EVAL/data/pic_mfu_step/final_mfu_step"
            plt.figure()
            plt.plot(final_step, final_mfu)
            plt.xlabel('step')
            plt.ylabel('mfu')
            plt.savefig(mfu_step_pic_filename)
            seq_len_pic_filename = f"/workspace/MARIO_EVAL/data/pic_mfu_step/final_seq_len_step"
            plt.figure()
            plt.plot(final_step, final_seq_len)
            plt.xlabel('step')
            plt.ylabel('seq_len')
            plt.savefig(seq_len_pic_filename)
            """

            foldername = f"/workspace/MARIO_EVAL/data/runtime_data/{self.config.batch_size}b_{self.config.n_generate_sample}sample_{self.config.iterations}iter_{self.config.question_range}_qaf_{self.config.num_few_shot}example"
            is_enable_prefix_caching = self.config.enable_prefix_caching
            if is_enable_prefix_caching:
                enable_number = 1
            else:
                enable_number = 0

            pre_mfu_pic_filename = f"{foldername}/final_pre_mfu{enable_number}"
            #画出final_seq_len和final_mfu随step的变化图，在同一个图中
            #pic_mfu是final_mfu的十万倍
            plt.figure()
            plt.plot(final_step, final_pre_mfu)
            plt.xlabel('step')
            plt.ylabel('mfu')
            plt.savefig(pre_mfu_pic_filename)
            plt.close()

            nopre_mfu_pic_filename = f"{foldername}/final_nopre_mfu{enable_number}"
            #画出final_seq_len和final_mfu随step的变化图，在同一个图中
            plt.figure()
            plt.plot(final_step, final_nopre_mfu)
            plt.xlabel('step')
            plt.ylabel('mfu')
            plt.savefig(nopre_mfu_pic_filename)
            plt.close()

            seq_len_pic_filename = f"{foldername}/final_seq_len{enable_number}"
            #画出final_seq_len随step的变化图
            plt.figure()
            plt.plot(final_step, final_seq_len)
            plt.xlabel('step')
            plt.ylabel('seq_len')
            plt.savefig(seq_len_pic_filename)
            plt.close()

            #画出final_prefill_len随step的变化图
            prefill_len_pic_filename = f"{foldername}/final_prefill_len{enable_number}"
            plt.figure()
            plt.plot(final_step, final_prefill_len)
            plt.xlabel('step')
            plt.ylabel('prefill_len')
            plt.savefig(prefill_len_pic_filename)
            plt.close()

            #画出final_decode_len随step的变化图
            decode_len_pic_filename = f"{foldername}/final_decode_len{enable_number}"
            plt.figure()
            plt.plot(final_step, final_decode_len)
            plt.xlabel('step')
            plt.ylabel('decode_len')
            plt.savefig(decode_len_pic_filename)
            plt.close()

            #画出time随step的变化图
            time_pic_filename = f"{foldername}/final_time{enable_number}"
            plt.figure()
            plt.plot(final_step, final_time)
            plt.xlabel('step')
            plt.ylabel('time')
            plt.savefig(time_pic_filename)
            plt.close()

            #画出final_unfinished随step的变化图
            unfinished_pic_filename = f"{foldername}/final_unfinished{enable_number}"
            plt.figure()
            plt.plot(final_step, final_unfinished)
            plt.xlabel('step')
            plt.ylabel('unfinished')
            plt.savefig(unfinished_pic_filename)
            plt.close()



            """
            #画出mfu随seq_len的变化图
            seq_mfu_filename = f"/workspace/MARIO_EVAL/data/runtime_mfu/final_seq_mfu"
            final_seq_len_sort = sorted(final_seq_len)
            pic_mfu_sort = [pic_mfu[final_seq_len.index(x)] for x in final_seq_len_sort]
            plt.figure()
            plt.plot(final_seq_len_sort, pic_mfu_sort)
            plt.xlabel('seq_len')
            plt.ylabel('mfu')
            plt.savefig(seq_mfu_filename)
            """

        #记录时间
        data = {
            "final_step": final_step,
            "final_seq_len": final_seq_len,
            "final_prefill_len": final_prefill_len,
            "final_decode_len": final_decode_len,
            "final_pre_mfu": final_pre_mfu,
            "final_nopre_mfu": final_nopre_mfu,
            "final_time": final_time,
            "final_unfinished": final_unfinished
        }
        #输出为json文件
        data_filename = f"{foldername}/final_data{enable_number}.json"
        #如果文件存在，则删除
        #if os.path.exists(data_filename):
        #    os.remove(data_filename)
        with open(data_filename, "w") as f:
            json.dump(data, f, indent=4)
        """
        mfu_pic_filename = f"/workspace/MARIO_EVAL/data/runtime_mfu/final_mfu"
        #画出final_seq_len和final_mfu随step的变化图，在同一个图中
        #pic_mfu是final_mfu的十万倍
        plt.figure()
        plt.plot(final_step, final_mfu, label='mfu')
        plt.xlabel('step')
        plt.ylabel('percent')
        plt.legend()
        plt.savefig(mfu_pic_filename)
        seq_mfu_filename = f"/workspace/MARIO_EVAL/data/runtime_mfu/final_seq_mfu"
        final_seq_len_sort = sorted(final_seq_len)
        pic_mfu_sort = [pic_mfu[final_seq_len.index(x)] for x in final_seq_len_sort]
        plt.figure()
        plt.plot(final_seq_len_sort, pic_mfu_sort)
        plt.xlabel('seq_len')
        plt.ylabel('mfu')
        plt.savefig(seq_mfu_filename)
        

        tree_pic_name5 =  f"/workspace/MARIO_EVAL/data/pic_tree/{datetime.now().strftime('%Y%m%d%H%M%S')}_final_tree"        

        """
        
    