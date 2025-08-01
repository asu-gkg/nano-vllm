
import os
import json

def check_mixtral_model(model_path):
    # 检查文件
    files = os.listdir(model_path)
    print(f"Files in {model_path}:")
    
    total_size = 0
    for f in files:
        if f.endswith(('.bin', '.safetensors')):
            size = os.path.getsize(os.path.join(model_path, f)) / 1024**3
            total_size += size
            print(f"  {f}: {size:.2f} GB")
    
    print(f"\nTotal size: {total_size:.2f} GB")
    
    # 检查索引
    for index_file in ['model.safetensors.index.json', 'pytorch_model.bin.index.json']:
        if index_file in files:
            with open(os.path.join(model_path, index_file)) as f:
                index = json.load(f)
                print(f"\nFound index: {index_file}")
                print(f"Total tensors: {len(index['weight_map'])}")
                
                # 统计专家权重
                expert_count = sum(1 for k in index['weight_map'] if 'experts' in k)
                print(f"Expert weights: {expert_count}")
                break
            
check_mixtral_model("/home/asu/Desktop/nano-vllm/Mixtral-8x7B-v0.1")

### 运行结果 ###

'''
uv run scripts/verify_mixtral_format.py 
Files in /home/asu/Desktop/nano-vllm/Mixtral-8x7B-v0.1:
  model-00010-of-00019.safetensors: 4.56 GB
  model-00009-of-00019.safetensors: 4.64 GB
  model-00005-of-00019.safetensors: 4.64 GB
  model-00004-of-00019.safetensors: 4.56 GB
  model-00014-of-00019.safetensors: 4.56 GB
  model-00018-of-00019.safetensors: 4.64 GB
  model-00015-of-00019.safetensors: 4.64 GB
  model-00002-of-00019.safetensors: 4.64 GB
  model-00003-of-00019.safetensors: 4.64 GB
  model-00008-of-00019.safetensors: 4.64 GB
  model-00017-of-00019.safetensors: 4.56 GB
  model-00006-of-00019.safetensors: 4.64 GB
  model-00012-of-00019.safetensors: 4.64 GB
  model-00001-of-00019.safetensors: 4.56 GB
  model-00007-of-00019.safetensors: 4.56 GB
  model-00013-of-00019.safetensors: 4.64 GB
  model-00019-of-00019.safetensors: 3.93 GB
  model-00011-of-00019.safetensors: 4.64 GB
  model-00016-of-00019.safetensors: 4.64 GB

Total size: 86.99 GB

Found index: model.safetensors.index.json
Total tensors: 995
Expert weights: 768
'''