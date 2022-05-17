import gc
import logging
import os
import signal
import sys
import time
import kfserving
from huggingface_hub import snapshot_download
from accelerate import init_empty_weights, dispatch_model, infer_auto_device_map, load_checkpoint_and_dispatch
import os
from transformers import AutoConfig, AutoTokenizer, pipeline, AutoModelForCausalLM
import torch

SERVER_NUM_WORKERS = int(os.environ.get('SERVER_NUM_WORKERS', 1))
SERVER_PORT = int(os.environ.get('SERVER_PORT', 8080))
MODEL_DEVICE = 0
MODEL_SIZE = '125m'
MODEL_PATH = f'facebook/opt-{MODEL_SIZE}'
MODEL_NAME = 'opt'
MODEL_PRECISION = os.environ.get('MODEL_PRECISION', 'native').lower()
READY_FLAG = '/tmp/ready'
DEBUG_MODE = bool(os.environ.get('DEBUG_MODE', 0))

logging.basicConfig(level=kfserving.constants.KFSERVING_LOGLEVEL)

logger = logging.getLogger(MODEL_NAME)


def load_opt_model(checkpoint: str):
    weights_path = snapshot_download(checkpoint)

    files = os.listdir(weights_path)
    weights_path = os.path.join(weights_path, 'pytorch_model.bin') if 'pytorch_model.bin' in files else weights_path

    config = AutoConfig.from_pretrained(checkpoint)

    model = AutoModelForCausalLM.from_config(config)
    model.tie_weights()

    # Infer device map automatically
    device_map = infer_auto_device_map(model.model, no_split_module_classes=["OPTDecoderLayer"], dtype='float16')

    if any([k == 'disk' for k in device_map.values()]):
        offload_folder = 'offload_folder'
    else:
        offload_folder = None

    if '30b' in checkpoint:
        # Set a few layers to use the disk manually to ensure enough RAM for the 30B checkpoint.
        device_map['decoder.layers.23'] = 'disk'
        device_map['decoder.layers.24'] = 'disk'
        device_map['decoder.layers.25'] = 'disk'
        device_map['decoder.layers.26'] = 'disk'
        device_map['decoder.layers.27'] = 'disk'

    load_checkpoint_and_dispatch(
        model.model,
        weights_path,
        device_map=device_map,
        offload_folder=offload_folder,
        dtype='float16',
        offload_state_dict=True
    )
    model.tie_weights()

    return model


class KFServingHuggingFace(kfserving.KFModel):
    def __init__(self, name):
        super().__init__(name)
        self.name = MODEL_NAME
        self.ready = False
        self.config = None
        self.tokenizer = None
        self.model = None
        self.generator = None
        self.bad_words_ids = None

    def load_config(self):
        logger.info(f'Loading config from {MODEL_PATH}')
        self.config = AutoConfig.from_pretrained(MODEL_PATH, local_files_only=False)
        logger.info('Config loaded.')

    def load_tokenizer(self):
        logger.info(f'Loading tokenizer from {MODEL_PATH} ...')
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=False)
        logger.info('Tokenizer loaded.')

    def load_bad_word_ids(self):
        logger.info('loading bad word ids')

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            local_files_only=False,
            add_prefix_space=True
        )

        forbidden = [
            'nigger', 'nigga', 'negro', 'blacks',
            'rapist', 'rape', 'raping', 'niggas', 'raper',
            'niggers', 'rapers', 'niggas', 'NOOOOOOOO',
            'fag', 'faggot', 'fags', 'faggots']

        bad_words_ids = []
        for word in forbidden:
            bad_words_ids.append(tokenizer(word).input_ids)
        self.bad_words_ids = bad_words_ids

        logger.info('done loading bad word ids')

    def load(self):
        """
        Load from a pytorch saved pickle to reduce the time it takes
        to load the model.  To benefit from this it is important to
        have run pytorch save on the same machine / hardware.
        """

        gc.disable()
        start_time = time.time()

        self.load_config()
        self.load_tokenizer()
        self.load_bad_word_ids()

        self.model = load_opt_model(MODEL_PATH)
        logger.info('Model loaded.')

        logger.info('Creating generator for model ...')
        logger.info(f'Model is ready in {str(time.time() - start_time)} seconds.')

        gc.enable()
        self.ready = True
        self._set_ready_flag()

    def predict(self, request, parameters=None):
        # batching requires fixed parameters
        request_params = {
            'temperature': 0.72,
            'repetition_penalty': 1.13125,
            'max_new_tokens': 64,
            'top_p': 0.725,
            'top_k': 0,
            'do_sample': True,
            'eos_token_id': 198,
            'bad_words_ids': self.bad_words_ids
        }

        if parameters is not None:
            request_params.update(parameters)

        inputs = request['instances']

        input_ids = self.tokenizer(
            inputs,
            add_special_tokens=False,
            return_tensors="pt",
            return_attention_mask=True,
            padding=True).to(0)

        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids['input_ids'],
                attention_mask=input_ids['attention_mask'],
                **request_params)

        responses = []
        for ins, outs in zip(inputs, outputs):
            decoded = self.tokenizer.decode(
                outs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False)
            decoded = decoded[len(ins):]
            responses.append(decoded.rstrip())

        return {'predictions': responses}

    def _set_ready_flag(self):
        """Used by readiness probe. """
        with open(READY_FLAG, 'w') as fh:
            fh.write('1')


def terminate(signal, frame):
    """
    Kubernetes send SIGTERM to containers for them
    to stop so this must be handled by the process.
    """
    logger.info("Start Terminating")
    if os.path.exists(READY_FLAG):
        os.remove(READY_FLAG)
    time.sleep(5)
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, terminate)

    if DEBUG_MODE:
        import time

        time.sleep(3600 * 10)

    model = KFServingHuggingFace(MODEL_NAME)
    model.load()

    kfserving.KFServer(
        http_port=SERVER_PORT,
        workers=SERVER_NUM_WORKERS
    ).start([model])
