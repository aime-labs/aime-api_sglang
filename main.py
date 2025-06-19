import asyncio
from argparse import ArgumentParser
import time
import inspect
from pathlib import Path
import re
import logging

import torch

import sglang as sgl
from sglang.srt.conversation import chat_templates
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.configs.model_config import ModelConfig
from sglang.test.test_utils import is_in_ci
from sglang.utils import trim_overlap
from sglang.lang.chat_template import get_chat_template_by_model_path
from sglang.srt.hf_transformers_utils import get_tokenizer

from aime_api_worker_interface import APIWorkerInterface


if is_in_ci():
    import patch
else:
    import nest_asyncio

    nest_asyncio.apply()


DEFAULT_WORKER_JOB_TYPE = "llama3"
DEFAULT_WORKER_AUTH_KEY = "5b07e305b50505ca2b3284b4ae5f65d1"
VERSION = 3

LLM_FAMILIES = [
    'Llama',
    'Mixtral',
    'Qwen',
    'DeepSeek'
]

QUANTIZATIONS = [
    'int1',
    'int4',
    'int8',
    'int16',
    'int32',
    'int64',
    'fp4',
    'fp8',
    'fp16',
    'fp32',
    'bfloat16',
    'uint4',
    'uint8',
    'uint16',
    'uint32',
    'uint64'
]

logger = logging.getLogger(__name__)

class SGLang():
    def __init__(self):
        self.args = self.load_flags()
        self.api_worker = APIWorkerInterface(
            self.args.api_server, 
            self.args.job_type, 
            self.args.api_auth_key, 
            self.args.gpu_id, 
            gpu_name=self.get_gpu_name(), 
            worker_version=VERSION,
            exit_callback=self.exit_callback,
            model_label=self.args.model_label,
            model_quantization=self.args.model_quantization, 
            model_size=self.args.model_size, 
            model_family=self.args.model_family, 
            model_type=self.args.model_type,
            model_repo_name=Path(self.args.model_path).name
        )
        
        self.progress_update_data = dict()
        self.last_progress_update = time.time()
        self.server_args = ServerArgs.from_cli_args(self.args)
        self.model_config = ModelConfig.from_server_args(self.server_args)
        self.llm_engine = sgl.Engine(server_args=self.server_args)

        asyncio.run(self.run_engine())


    def get_gpu_name(self):
        return f'{self.args.tensor_parallel_size}x{torch.cuda.get_device_name(0)}'


    async def generate(self, job_data):

        arrival_time = time.time()
        job_id = job_data.get('job_id')
        prompt_input_ids = self.get_prompt_input_ids(job_data)
        if prompt_input_ids:
            input_length = len(prompt_input_ids)
            if input_length <= self.model_config.context_len:
                generator = await self.llm_engine.async_generate(
                    input_ids=prompt_input_ids,
                    sampling_params=self.get_sampling_params(job_data),
                    stream=True
                )

                async for output in generator:
                    if not output.get('meta_info').get('finish_reason'):
                        self.progress_update_data[job_id] = self.get_result(output, arrival_time)
                        self.update_progress()
                    else:
                        self.api_worker.send_job_results(
                            self.get_result(output, arrival_time),
                            job_id=job_id,
                            wait_for_response=False,
                            error_callback=self.error_callback
                        )
                        self.progress_update_data.pop(job_id, None)
            else:
                error_msg = f'The context length {input_length} is exceeding the maximum context length {self.model_config.context_len}!'
                logging.warning(f'Job {job_id} failed: {error_msg}')
                output = {
                    'meta_info': {'prompt_tokens': input_length},
                    'error': error_msg
                }
                self.api_worker.send_job_results(
                    self.get_result(output, arrival_time),
                    job_id=job_id,
                    wait_for_response=False,
                    error_callback=self.error_callback
                )
            


    async def run_engine(self):
        job_request_generator = self.api_worker.job_request_generator(self.args.max_batch_size)
        for job_batch_data in job_request_generator:
            new_jobs = list()
            for job_data in job_batch_data:
                asyncio.ensure_future(self.generate(job_data))
                new_jobs.append('#' + job_data.get('job_id').split('#')[1])
            if new_jobs:
                logging.info(f'Job(s) {", ".join(new_jobs)} added.')
            await asyncio.sleep(0.5)



    def error_callback(self, response):
        logging.error(response)
        raise response


    def get_prompt_input_ids(self, job_data):
        prompt_input = job_data.get('prompt_input') or job_data.get('text')
        chat_context = job_data.get('chat_context')
        if chat_context:
            if not self.validate_chat_context(job_data.get('job_id'), chat_context):
                self.logger.warning('Wrong context shape')
                return
            if prompt_input:
                chat_context.append(
                    {
                        "role": "user", 
                        "content": prompt_input
                    }
                )
            chat_template = get_chat_template_by_model_path(self.args.model_path)
            prompt_input = chat_template.get_prompt(chat_context)

        tokenizer = get_tokenizer(
            self.server_args.tokenizer_path,
            tokenizer_mode=self.server_args.tokenizer_mode,
            trust_remote_code=self.server_args.trust_remote_code,
            revision=self.server_args.revision,
        )
        return tokenizer(prompt_input).get('input_ids')
      

    def validate_chat_context(self, job_id, chat_context):
        for item in chat_context:
            if not isinstance(item, dict) or not all(key in item for key in ("role", "content")):
                result = {
                    'error':  f'Dialog has invalid chat context format! Format should be [{{"role": "user/assistant/system", "content": "Message content"}}, ...] but is {chat_context}',
                }
                self.api_worker.send_job_results(result, job_id=job_id)
                return False
        return True


    def update_progress(self):
        now = time.time()
        if (now - self.last_progress_update) > (1.0 / self.args.progress_rate):
            self.last_progress_update = now
            progress_result_batch = list()
            job_id_batch = list()
            num_generated_tokens_batch = list()
            for job_id, progress_result in self.progress_update_data.items():
                progress_result_batch.append(progress_result)
                job_id_batch.append(job_id)
                num_generated_tokens_batch.append(progress_result.get('num_generated_tokens', 0))
            self.progress_update_data.clear()
            self.api_worker.send_batch_progress(
                num_generated_tokens_batch,
                progress_result_batch,
                job_batch_ids=job_id_batch,
                progress_error_callback=self.error_callback
            )


    def get_result(self, output, arrival_time):
        if output:
            meta_info = output.get('meta_info', {})
            input_length = meta_info.get('prompt_tokens', 0)
            num_generated_tokens = meta_info.get('completion_tokens', 0)

            result = {
                'num_generated_tokens': num_generated_tokens,
                'max_seq_len': self.model_config.context_len,
                'prompt_length': input_length,
                'arrival_time': arrival_time,
                'finished_time': time.time(),
                'current_context_length': input_length + num_generated_tokens,
                'pending_duration': 0,
                'metrics': {
                    'in_num_tokens': input_length,
                    'out_num_tokens': num_generated_tokens, 
                },
                'model_name': self.args.model_label
            }
            if output.get('text'):
                result['text'] = output['text']
            if output.get('error'):
                result['error'] = output['error']
            return result
        


    def get_sampling_params(self, job_data):
        
        sampling_params_keys = inspect.signature(SamplingParams).parameters.keys()
        sampling_params = {key: job_data[key] for key in sampling_params_keys if key in job_data}
        sampling_params['max_new_tokens'] = job_data.get('max_gen_tokens')
        return sampling_params #SamplingParams(**sampling_params)


    def load_flags(self):
        parser = ArgumentParser()
        parser.add_argument(
            "--max_batch_size", type=int, default=256,
            help="Maximum batch size"
        )
        parser.add_argument(
            "--job_type", type=str, default=DEFAULT_WORKER_JOB_TYPE,
            help="Worker job type for the API Server"
        )
        parser.add_argument(
            "--model_label", type=str,
            help="Model label to display in client. Default: Name of the directory given in --model"
        )
        parser.add_argument(
            "--model_size", type=str,
            help="Model size (number of parameters). E.g. '8B', '70B'"
        )
        parser.add_argument(
            "--model_family", type=str,
            help="Model family. E.g. 'Llama', 'Mixtral'"
        )
        parser.add_argument(
            "--model_quantization", choices=QUANTIZATIONS,
            help="Model quantization"
        )
        parser.add_argument(
            "--model_type", type=str, default="LLM",
            help="Model type e.g. 'LLM'."
        )
        parser.add_argument(
            "--api_server", type=str, required=True,
            help="Address of the API server"
        )
        parser.add_argument(
            "--gpu_id", type=int, default=0,
            help="ID of the GPU to be used"
        )
        parser.add_argument(
            "--api_auth_key", type=str , default=DEFAULT_WORKER_AUTH_KEY,
            help="API server worker auth key",
        )
        parser.add_argument(
            "--progress_rate", type=int , default=5,
            help="Progress updates per sec to the API Server.",
        )
        parser.add_argument(
            "--dev", action='store_true',
            help="Sets logger level to DEBUG",
        )
        ServerArgs.add_cli_args(parser)
        args = parser.parse_args()
        args.model_label = args.model_label or Path(args.model_path).name
        args.model_size = args.model_size or self.extract_model_size(args)
        args.model_quantization = args.model_quantization or self.extract_quantization(args) or 'fp16'
        args.model_family = args.model_family or self.extract_family(args)
        return args

    def extract_model_size(self, args):
        match = re.search(r'(\d+B)', Path(args.model_path).name.upper())  # Matches arbitrary number of digits followed by 'B'
        return match.group(1) if match else None
        

    def extract_quantization(self, args):
        for quantization in QUANTIZATIONS:
            if quantization in Path(args.model_path).name.lower():
                return quantization

    def extract_family(self, args):
        for family in LLM_FAMILIES:
            if family.lower() in Path(args.model_path).name.lower():
                return family

    def exit_callback(self):
        self.llm_engine.shutdown()
        del self.llm_engine
        torch.cuda.empty_cache()



if __name__ == "__main__":
    sg_lang = SGLang()
