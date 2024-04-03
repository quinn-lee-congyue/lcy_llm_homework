import os
import platform
from pathlib import Path
from typing import Union, TypeVar, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM


# 定义模型和分词器的类型变量
ModelType = TypeVar('ModelType', bound=AutoModelForCausalLM)
TokenizerType = TypeVar('TokenizerType', bound=AutoTokenizer)

# MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/chatglm3-6b')
# TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)

# tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
# model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map="auto").eval()
def _resolve_path(path: Union[str, Path]) -> Path:
    """
    将输入路径解析为绝对路径。

    Parameters:
        path (Union[str, Path]): 路径字符串或 Path 对象。

    Returns:
        Path: 绝对路径对象。
    """
    # 将路径字符串转换为 Path 对象，扩展用户路径并解析成绝对路径
    return Path(path).expanduser().resolve()

def load_model_and_tokenizer(
        model_dir: Union[str, Path], trust_remote_code: bool = True
) -> tuple[ModelType, TokenizerType]:
    model_dir = _resolve_path(model_dir)
    if (model_dir / 'adapter_config.json').exists():
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=trust_remote_code, device_map='auto'
        )
        tokenizer_dir = model.peft_config['default'].base_model_name_or_path
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=trust_remote_code, device_map='auto'
        )
        tokenizer_dir = model_dir
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, trust_remote_code=trust_remote_code
    )
    return model, tokenizer


model,tokenizer = load_model_and_tokenizer("/root/ChatGLM3/finetune_demo/output/checkpoint-1000")

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False

welcome_prompt = "欢迎使用 ChatGLM3-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"


def build_prompt(history):
    prompt = welcome_prompt
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM3-6B：{response}"
    return prompt

def main():
    past_key_values, history = None, []
    global stop_stream
    print(welcome_prompt)
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            past_key_values, history = None, []
            os.system(clear_command)
            print(welcome_prompt)
            continue
        print("\nChatGLM：", end="")
        current_length = 0
        for response, history, past_key_values in model.stream_chat(tokenizer, query, history=history, top_p=0.95,
                                                                    temperature=0.6,
                                                                    max_length=100,
                                                                    num_return_sequences=1,
                                                                    repetition_penalty=1.2,
                                                                    past_key_values=past_key_values,
                                                                    return_past_key_values=True):
            if stop_stream:
                stop_stream = False
                break
            else:
                print(response[current_length:], end="", flush=True)
                current_length = len(response)
        print("")


if __name__ == "__main__":
    main()
