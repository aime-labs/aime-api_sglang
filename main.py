import asyncio
import argparse
import time
import inspect
from pathlib import Path
import re
import logging

import torch

import sglang as sgl
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.configs.model_config import ModelConfig
from sglang.test.test_utils import is_in_ci

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


class SGLang():
    def __init__(self):
        self.args = self.load_flags()
        self.server_args = ServerArgs.from_cli_args(self.args)
        self.model_config = ModelConfig.from_server_args(self.server_args)
        self.api_worker = APIWorkerInterface(
            self.args.api_server, 
            self.args.job_type,
            self.args.api_auth_key, 
            self.args.gpu_id, 
            gpu_name=torch.cuda.get_device_name(0), 
            queue_name=self.args.queue_name,
            num_gpus=self.args.tensor_parallel_size,
            worker_version=VERSION,
            exit_callback=self.exit_callback,
            model_label=self.args.model_label,
            model_quantization=self.args.model_quantization, 
            model_size=self.args.model_size, 
            model_family=self.args.model_family, 
            model_type=self.args.model_type,
            model_repo_name=Path(self.args.model_path).name,
            model_path=self.args.model,
            framework='SGLang',
            framework_version=sgl.version.__version__,
            pytorch_version=torch.version.__version__,
            auto_tokenize_inputs=True,
            trust_remote_code=self.args.trust_remote_code,
            use_fast_tokenizer=self.args.use_fast_tokenizer,
            max_batch_size=self.args.max_batch_size,
            max_context_length=self.model_config.context_len,
            starts_with_think=self.args.starts_with_think
        )
        self.progress_update_data = dict()
        self.last_progress_update = time.time()
        self.llm_engine = sgl.Engine(server_args=self.server_args)
        try:
            asyncio.run(self.run_engine())
        except KeyboardInterrupt:
            logging.info('KeyboardInterrupt triggered. Initiating shutdown sequence...')
            self.api_worker.gracefully_exit()


    async def process_job_batch(self, job_batch_data):
        arrival_time = time.time()
        if job_batch_data:
            input_id_batch, sampling_params_batch, job_id_batch, image_data_batch, audio_data_batch, video_data_batch = self.get_parameter_batches(job_batch_data)
            generator = await self.llm_engine.async_generate(
                input_ids=input_id_batch,
                sampling_params=sampling_params_batch,
                stream=True,
                image_data=image_data_batch,
                #audio_data=audio_data_batch,
                #video_data=video_data_batch for later SGLang versions
            )
            logging.info(f'Job(s) {", ".join([self.format_job_id(job_id) for job_id in job_id_batch])} added.')
            start_time_processing = None

            async for output in generator:
                if output and not start_time_processing:
                    start_time_processing = time.time()
                job_id = job_id_batch[output.get('index')]
                if not output.get('meta_info').get('finish_reason'):
                    self.progress_update_data[job_id] = self.get_result(output, arrival_time, start_time_processing)
                    self.update_progress()
                else:
                    self.api_worker.send_job_results(
                        self.get_result(output, arrival_time, start_time_processing),
                        job_id=job_id,
                        wait_for_response=False,
                        error_callback=self.error_callback
                    )
                    self.progress_update_data.pop(job_id, None)
            if self.args.flush_cache:
                state = await self.llm_engine.tokenizer_manager.get_internal_state()
                if state[0].get('load') == 0:
                    await self.llm_engine.tokenizer_manager.flush_cache()


    async def run_engine(self):
        job_request_generator = self.api_worker.job_request_generator(self.args.max_batch_size)
        for job_batch_data in job_request_generator:
            if job_batch_data:
                asyncio.ensure_future(self.process_job_batch(job_batch_data))
            await asyncio.sleep(0.1)




    def error_callback(self, response):
        logging.error(response)


    def get_parameter_batches(self, job_batch_data):
        
        input_id_batch, sampling_params_batch, job_id_batch, image_data_batch, audio_data_batch, video_data_batch = zip(*[
            (
                job_data.get('chat_context') or job_data.get('text_context') or [self.api_worker.tokenizer.convert_tokens_to_ids(self.api_worker.tokenizer.bos_token)],
                self.get_sampling_params(job_data),
                job_data.get('job_id'),
                job_data.get('multimodal_data', {}).get('chat_context', {}).get('image', []),
                job_data.get('multimodal_data', {}).get('chat_context', {}).get('audio', []),
                job_data.get('multimodal_data', {}).get('chat_context', {}).get('video', [])
            )
            for job_data in job_batch_data
        ])
        return list(input_id_batch), list(sampling_params_batch), list(job_id_batch), list(image_data_batch), list(audio_data_batch), list(video_data_batch)


    def validate_chat_context(self, chat_context):
        for item in chat_context:
            if not isinstance(item, dict) or not all(key in item for key in ("role", "content")):
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
                job_batch_ids=job_id_batch
            )


    def get_result(self, output, arrival_time, start_time_processing):
        if output:
            meta_info = output.get('meta_info', {})
            input_length = meta_info.get('prompt_tokens', 0) or output.get('input_length')
            num_generated_tokens = meta_info.get('completion_tokens', 0)
            result = {
                'model_name': self.args.model_label,
                'num_generated_tokens': num_generated_tokens,
                'max_seq_len': self.model_config.context_len,
                'prompt_length': input_length,
                'arrival_time': arrival_time,
                'finished_time': time.time(),
                'current_context_length': input_length + num_generated_tokens,
                'preprocessing_duration': start_time_processing - arrival_time,
                'metrics': {
                    'in_num_tokens': input_length,
                    'out_num_tokens': num_generated_tokens, 
                }
            }
            if output.get('error'):
                result['error'] = output['error']
            elif output.get('output_ids'):
                result['text'] = output.get('output_ids')
            return result
        

    def get_sampling_params(self, job_data):
        input_length = len(job_data.get('chat_context') or job_data.get('text_context', []))
        sampling_params_keys = inspect.signature(SamplingParams).parameters.keys()
        sampling_params = {key: job_data[key] for key in sampling_params_keys if key in job_data}
        sampling_params['max_new_tokens'] = min(job_data.get('max_gen_tokens'), self.model_config.context_len - input_length -1)
        sampling_params['stop_token_ids'] = [self.api_worker.tokenizer.convert_tokens_to_ids(self.api_worker.tokenizer.eos_token)]
        return sampling_params #SamplingParams(**sampling_params)


    def load_flags(self):
        parser = argparse.ArgumentParser(conflict_handler='resolve')
        ServerArgs.add_cli_args(parser)
        parser.add_argument(
            "--max_batch_size", type=int, default=256,
            help="Maximum batch size"
        )
        parser.add_argument(
            "--job_type", type=str, default=DEFAULT_WORKER_JOB_TYPE,
            help="Worker job type for the API Server"
        )
        parser.add_argument(
            "--queue_name", type=str,
            help="Worker job type for the API Server"
        )        
        parser.add_argument(
            "--model", type=str, required=True,
            help="The path of the model weights. This can be a local folder or a Hugging Face repo ID. Translates to --model-path in SGLang."
        )
        parser.add_argument(
            "--model-path", type=str,
            help=argparse.SUPPRESS # To override original sglang cli arg to become optional
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
        parser.add_argument(
            "--flush_cache", action='store_true',
            help="Resets cache before every request",
        )
        parser.add_argument(
            "--use_fast_tokenizer", action='store_true',
            help="Use fast tokenizer in API Worker Interface",
        )
        parser.add_argument(
            "--starts_with_think", type=lambda x: x.lower() == 'true', nargs='?', const=True, default=None,
            help="Whether the model is a reasoning model and uses <think> / </think> tags. "
                 "If omitted, defaults to value from received from endpoint config."
         )
        args = parser.parse_args()
        args.model_path = args.model
        args.model_label = args.model_label or Path(args.model_path).name
        args.model_size = args.model_size or self.extract_model_size(args)
        args.model_quantization = args.model_quantization or self.extract_quantization(args) or 'fp16'
        args.model_family = args.model_family or self.extract_family(args)
        args.max_running_requests = args.max_batch_size
        args.skip_tokenizer_init = True
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
        if hasattr(self, 'llm_engine'):
            self.llm_engine.shutdown()
        torch.cuda.empty_cache()


    def format_job_id(self, job_id):
        uuid, seperator, counter = job_id.partition('#')
        return seperator + counter if counter else uuid


if __name__ == "__main__":
    sg_lang = SGLang()
