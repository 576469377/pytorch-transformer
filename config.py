from pathlib import Path

def get_config():
    return {
        "batch_size": 24,         # 批次大小
        "num_epochs": 20,        # 训练轮数
        "lr": 1e-4,              # 学习率
        "seq_len": 500,          # 序列最大长度
        "d_model": 512,          # 模型维度
        "datasource": 'opus_books', # 数据源
        "lang_src": "en",        # 源语言（英语）
        "lang_tgt": "fr",        # 目标语言（法语）
        "model_folder": "weights", # 权重存储文件夹
        "model_basename": "tmodel_", # 模型文件名前缀
        "preload": "latest",     # 预加载模式（加载最新模型）
        "tokenizer_file": "tokenizer_{0}.json", # 分词器文件名格式
        "experiment_name": "runs/tmodel", # 实验名称（可能用于tensorboard）
        "cache_dir": "cache"     # 缓存目录
    }

# 获取权重文件路径
def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1]) 
