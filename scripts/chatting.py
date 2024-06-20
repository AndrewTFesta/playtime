"""
@title

@description

"""
import argparse
from collections import namedtuple
from enum import Enum
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import pipeline

from playtime import project_properties

chat_message = namedtuple('chat_message', ['role', 'content'])

# With my current setup, I cannot load and run inference on the 70B model with a single gpu.
# https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html
MODEL_NAMES = [
    'meta-llama/Meta-Llama-3-8B',
    'meta-llama/Meta-Llama-3-8B-Instruct',
    # 'meta-llama/Meta-Llama-3-70B'
    # 'meta-llama/Meta-Llama-3-70B-Instruct'
]
MODEL_TASK = 'text-generation'
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.1
SECRETS_PATH = Path(project_properties.secrets_file)
AUTH_TOKEN_NAME = 'huggingface_token_read'


class AgentType(Enum):
    AGENT = 'Agent'
    USER = 'User'
    SYSTEM = 'SYSTEM'


def auth_hf(secrets_path):
    with open(secrets_path, 'r') as secrets_file:
        secrets = yaml.safe_load(secrets_file)

    auth_token = secrets[AUTH_TOKEN_NAME]
    return auth_token


def print_response(response):
    print(response[0]['generated_text'][-1]['content'])
    return


def setup_pipeline():
    """
    Not all chat models support system messages, but when they do, they represent high-level directives about how the model should behave in the conversation.
    You can use this to guide the model - whether you want short or long responses, lighthearted or serious ones, and so on. If you want the model to do useful work
    instead of practicing its improv routine, you can either omit the system message or try a terse one such as “You are a helpful and intelligent AI assistant who
    responds to user queries.”

    Model considerations
    - The model’s size, which will determine if you can fit it in memory and how quickly it will run.
    - The quality of the model’s chat output.
    In general, these are correlated - bigger models tend to be more capable, but even so there’s a lot of variation at a given size point!

    Without quantization, you should expect to need about 2 bytes of memory per parameter.
    his means that an “8B” model with 8 billion parameters will need about 16GB of memory just to fit the parameters, plus a little extra for other overhead. It’s a good fit for
    a high-end consumer GPU with 24GB of memory, such as a 3090 or 4090.

    Some chat models are “Mixture of Experts” models. These may list their sizes in different ways, such as “8x7B” or “141B-A35B”.

    Even once you know the size of chat model you can run, there’s still a lot of choice out there. One way to sift through it all is to consult leaderboards.
    Two of the most popular leaderboards are the [OpenLLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) and the
    [LMSys Chatbot Arena Leaderboard](https://chat.lmsys.org/?leaderboard). Note that the LMSys leaderboard also includes proprietary models - look at the licence column to
    identify open-source ones that you can download, then search for them on the [Hugging Face Hub](https://huggingface.co/models?pipeline_tag=text-generation&sort=trending).

    :return:
    """
    with open(SECRETS_PATH, 'r') as secrets_file:
        secrets = yaml.safe_load(secrets_file)

    auth_token = secrets[AUTH_TOKEN_NAME]
    chat = [
        chat_message(role='system', content='You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986.')._asdict(),
        chat_message(role='user', content='Hey, can you tell me any fun things to do in New York?')._asdict()
    ]

    pipe = pipeline(MODEL_TASK, MODEL_NAMES[1], torch_dtype=torch.bfloat16, device_map="auto", token=auth_token)
    response = pipe(chat, max_new_tokens=MAX_NEW_TOKENS)
    print_response(response)

    chat = response[0]['generated_text']
    chat.append(chat_message(role='user', content='Wait, what\'s so wild about soup cans?')._asdict())
    response = pipe(chat, max_new_tokens=MAX_NEW_TOKENS)
    print_response(response)
    return


def main(main_args):
    with open(SECRETS_PATH, 'r') as secrets_file:
        secrets = yaml.safe_load(secrets_file)
    auth_token = secrets[AUTH_TOKEN_NAME]
    chat = [
        {'role': 'system', 'content': 'You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986.'},
        {'role': 'user', 'content': 'Hey, can you tell me any fun things to do in New York?'}
    ]

    # 1: Load the [model](https://huggingface.co/learn/nlp-course/en/chapter2/3) and [tokenizer](https://huggingface.co/learn/nlp-course/en/chapter2/4?fw=pt)
    # 2: Apply the [chat template](https://huggingface.co/docs/transformers/main/en/chat_templating)
    # 3: [Tokenize the chat](https://huggingface.co/learn/nlp-course/en/chapter2/4) (This can be combined with the previous step using tokenize=True)
    # Move the tokenized inputs to the same device the model is on (GPU/CPU)
    # 4: [Generate text](https://huggingface.co/docs/transformers/en/llm_tutorial) from the model
    # 5: Decode the output back to a string

    # model = AutoModelForCausalLM.from_pretrained(MODEL_NAMES[1], device_map='auto', torch_dtype=torch.bfloat16, token=auth_token)
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES[1], token=auth_token)

    # [Quantization](https://huggingface.co/docs/transformers/main/en/quantization/overview)
    # It is possible to go even lower than 16-bits using “quantization”, a method to lossily compress model weights. This allows each parameter to be squeezed down
    # to 8 bits, 4 bits or even less. Note that, especially at 4 bits, the model’s outputs may be negatively affected, but often this is a tradeoff worth making to
    # fit a larger and more capable chat model in memory.
    # Let's try loading the model in 8 bits. You can also try load_in_4bit.
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAMES[1], device_map="auto", token=auth_token, quantization_config=quantization_config)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES[1], token=auth_token, quantization_config=quantization_config)

    # Or using the pipeline
    # quantization_config = BitsAndBytesConfig(load_in_8bit=True)  # You can also try load_in_4bit
    # pipe = pipeline(MODEL_TASK, MODEL_NAMES[1], device_map="auto", model_kwargs={"quantization_config": quantization_config})

    formatted_chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_chat, return_tensors='pt', add_special_tokens=False)
    inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}
    outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE)
    decoded_output = tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)

    print(f'Formatted chat:\n{formatted_chat}')
    print(f'Tokenized inputs:\n{inputs}')
    print(f'Generated tokens:\n{outputs}')
    print(f'Decoded output:\n{decoded_output}')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
