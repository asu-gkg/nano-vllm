#!/usr/bin/env python3
"""
流式聊天程序 - 支持实时单词输出
"""

import os
import sys
import time
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


class StreamingChat:
    def __init__(self, model_path="/home/asu/qwen3-0.6b", typing_effect=True, system_prompt=None):
        """初始化聊天系统"""
        print("🤖 正在加载模型，请稍候...")
        
        self.model_path = os.path.expanduser(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.llm = LLM(self.model_path, enforce_eager=True, tensor_parallel_size=1)
        self.typing_effect = typing_effect
        
        # 设置system prompt
        if system_prompt is None:
            self.system_prompt = "你是一个有用、无害、诚实的AI助手。请用中文回答问题。"
        else:
            self.system_prompt = system_prompt
        
        # 设置采样参数
        self.sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=512,
            ignore_eos=False
        )
        
        effect_status = "开启" if typing_effect else "关闭"
        print(f"✅ 模型加载完成！打字效果: {effect_status}")
        print(f"🎯 System Prompt: {self.system_prompt[:50]}...")
        print()
    
    def format_prompt(self, user_input):
        """格式化用户输入为聊天模板"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input}
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
    
    def stream_generate(self, prompt):
        """流式生成回复"""
        # 添加请求到引擎
        self.llm.add_request(prompt, self.sampling_params)
        
        # 用于追踪已输出的内容
        last_decoded_length = 0
        sequence_id = None
        
        print("🤖 AI: ", end="", flush=True)
        
        # 循环生成直到完成
        while not self.llm.is_finished():
            # 执行一步生成
            outputs, _ = self.llm.step()
            
            # 如果有序列完成了，输出最终结果
            for seq_id, token_ids in outputs:
                if token_ids:
                    # 解码完整文本
                    full_text = self.tokenizer.decode(token_ids)
                    # 输出剩余未输出的部分
                    remaining_text = full_text[last_decoded_length:]
                    if remaining_text:
                        print(remaining_text, end="", flush=True)
                    return  # 序列完成，退出
            
            # 检查正在运行的序列，输出新生成的内容
            for seq in self.llm.scheduler.running:
                if sequence_id is None:
                    sequence_id = seq.seq_id
                
                if seq.seq_id == sequence_id and seq.num_completion_tokens > 0:
                    # 解码当前所有completion tokens
                    current_text = self.tokenizer.decode(seq.completion_token_ids)
                    
                                         # 只输出新增的部分
                    if len(current_text) > last_decoded_length:
                        new_text = current_text[last_decoded_length:]
                        
                        if self.typing_effect:
                            # 按字符逐个输出，营造打字效果
                            for char in new_text:
                                print(char, end="", flush=True)
                                # 添加不同延迟来模拟真实打字
                                if char in '，。！？；：,!?;:':
                                    time.sleep(0.2)  # 标点符号后稍微暂停长一点
                                elif char == ' ':
                                    time.sleep(0.1)  # 空格后暂停稍长
                                else:
                                    time.sleep(0.05)  # 普通字符
                        else:
                            # 直接输出新文本，只添加很短的延迟
                            print(new_text, end="", flush=True)
                            time.sleep(0.01)
                        
                        last_decoded_length = len(current_text)
                    break
        
        print("\n")  # 换行结束回复
    
    def chat_loop(self):
        """主聊天循环"""
        print("💬 欢迎使用流式聊天程序！")
        print("📝 输入你的问题，输入 'quit' 或 'exit' 退出程序")
        print("⚙️  输入 'toggle' 切换打字效果开关")
        print("-" * 50)
        
        while True:
            # 获取用户输入
            user_input = input("\n👤 你: ").strip()
            
            # 检查退出命令
            if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                print("👋 再见！")
                break
            
            # 检查切换打字效果命令
            if user_input.lower() in ['toggle', 'switch', '切换']:
                self.typing_effect = not self.typing_effect
                status = "开启" if self.typing_effect else "关闭"
                print(f"⚙️  打字效果已{status}")
                continue
            
            # 检查空输入
            if not user_input:
                print("❌ 请输入有效的问题")
                continue
            
            # 格式化prompt并生成回复
            formatted_prompt = self.format_prompt(user_input)
            self.stream_generate(formatted_prompt)


def main():
    """主函数"""
    # 检查命令行参数
    default_model_path = "/home/asu/qwen3-0.6b"
    model_path = default_model_path
    typing_effect = True
    
    # 解析命令行参数
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg in ['--no-typing', '--fast']:
            typing_effect = False
        elif arg in ['--typing', '--slow']:
            typing_effect = True
        elif arg in ['--help', '-h']:
            print("流式聊天程序")
            print("用法: python chat.py [模型路径] [选项]")
            print("选项:")
            print("  --typing, --slow    启用打字效果 (默认)")
            print("  --no-typing, --fast 禁用打字效果")
            print("  --help, -h          显示此帮助信息")
            print("示例:")
            print("  python chat.py                           # 使用默认模型和打字效果")
            print("  python chat.py --no-typing               # 使用默认模型，禁用打字效果")
            print("  python chat.py /path/to/model --fast     # 使用指定模型，禁用打字效果")
            sys.exit(0)
        elif not arg.startswith('--'):
            model_path = arg
        i += 1
    
    if not os.path.exists(os.path.expanduser(model_path)):
        print(f"❌ 错误: 找不到模型路径 {model_path}")
        print(f"💡 请确认模型路径正确，或者提供正确的路径作为参数:")
        print(f"   python chat.py /path/to/your/model")
        print(f"💡 使用 --help 查看所有选项")
        sys.exit(1)
    
    # 创建并启动聊天
    chat = StreamingChat(model_path, typing_effect)
    try:
        chat.chat_loop()
    except KeyboardInterrupt:
        print("\n\n👋 检测到中断信号，程序退出")


if __name__ == "__main__":
    main() 