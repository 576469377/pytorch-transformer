import torch.backends
from model import build_transformer, Transformer
from dataset import BilingualDataset, causal_mask # defined in dataset.py
from config import get_config, get_weights_file_path, latest_weights_file_path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path
import gc

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import sacrebleu
from torchmetrics.text import CharErrorRate, WordErrorRate

from torch.utils.tensorboard import SummaryWriter

# 设置 CUDA 内存分配器配置以减少内存碎片
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5, 6, 7'

def greedy_decode(model: Transformer, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    # source 的形状是 (1 seq_len)
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    # decoder_input 的形状是 (1, 1) （只有 1 个 SOS token）
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    # 生成目标序列（next token prediction, 贪心解码）
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1]) # (1, vocab_size)
        _, next_word = torch.max(prob, dim=1) # 获取概率最大的 token
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0) # (1, seq_len) -> (seq_len)


def run_validation(model: Transformer, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval() # 设置模型为评估模式
    count = 0 # 统计验证集中的样本数量

    source_texts = [] # 存储源语言的句子
    expected = [] # 存储目标语言的句子
    predicted = [] # 存储预测的目标语言的句子

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds: # 每次迭代 1 个样本，validation 阶段使用的batch_size = 1
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device) # (seq_len)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy()) # (seq_len) -> (seq_len), 将模型输出转换为字符串

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    if writer:
        # 使用jiwer计算字符错误率
        cer = calculate_cer(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # 使用jiwer计算词错误率
        wer = calculate_wer(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # 使用NLTK计算BLEU分数
        try:
            bleu = calculate_bleu(predicted, expected)
            writer.add_scalar('validation BLEU', bleu, global_step)
            writer.flush()
        except:
            print("计算BLEU分数时出错")

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]")) # 使用 WordLevel 分词器，将未知的 token 设置为 "[UNK]"，每个单词一个 token
        tokenizer.pre_tokenizer = Whitespace() # 使用 Whitespace 分词器，将单词之间的空格作为分隔符
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2) # 一个词至少出现2次才能被保留
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer) # 从数据集中获取所有句子，并使用训练器训练分词器
        tokenizer.save(str(tokenizer_path)) # 保存分词器
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    # It only has the train split, so we divide it ourselves
    ds_raw = load_dataset(path=f"{config['datasource']}", name=f"{config['lang_src']}-{config['lang_tgt']}", split='train', cache_dir=f"{config['cache_dir']}")

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=0)


    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model

def train_model(config):
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    device = torch.device(device)

    # Make sure the weights folder exists
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    # Tensorboard
    writer = SummaryWriter(config['experiment_name']) # 创建一个 SummaryWriter 对象，用于记录训练过程中的指标

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    # initial_epoch 表示训练起始的 epoch 计数器（从第0轮开始）
    # 如果从检查点恢复训练，该值会被更新为上次中断的 epoch 数
    initial_epoch = 0
    # global_step 表示全局训练步数计数器（每处理一个 batch 递增一次）
    # 用于学习率调度、日志记录等需要精确到步的操作
    global_step = 0
    preload = config['preload']
    # 如果 preload 为 'latest'，则加载最新的模型
    # 如果 preload 为具体的 epoch 数，则加载该 epoch 的模型
    # 如果 preload 为 None，则不加载任何模型
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename) # 获取模型断点文件
        model.load_state_dict(state['model_state_dict']) # 加载模型参数
        initial_epoch = state['epoch'] + 1 # 更新初始 epoch 计数器
        optimizer.load_state_dict(state['optimizer_state_dict']) # 加载优化器参数
        global_step = state['global_step'] # 更新全局训练步数计数器
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        gc.collect()  # 添加 Python 的垃圾回收
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # Compute the loss using a simple cross entropy
            # nn.CrossEntropyLoss 期望的输入是：
            # 预测值：[N, C]，其中 N 是样本数，C 是类别数（词汇表大小）
            # 标签：[N]，一维的类别索引
            # proj_output.view(-1, tokenizer_tgt.get_vocab_size()) 的维度是 (B * seq_len, vocab_size)
            # label.view(-1) 的维度是 (B * seq_len)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            optimizer.step() # update the weights
            optimizer.zero_grad(set_to_none=True) # clear the gradients, set_to_none=True 表示如果梯度为 None，则不进行任何操作 (效率更高)

            global_step += 1

        # Run validation at the end of every epoch
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)
        torch.cuda.empty_cache()
        gc.collect()

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

# 辅助函数定义
# 三个指标：WER、CER 和 BLEU
def calculate_wer(predicted, expected):
    """使用torchmetrics计算WER"""
    wer_metric = WordErrorRate()
    if isinstance(predicted, list):
        predicted = " ".join(predicted)
    if isinstance(expected, list):
        expected = " ".join(expected)
    return wer_metric([predicted], [expected]).item()

def calculate_cer(predicted, expected):
    """使用torchmetrics计算CER"""
    cer_metric = CharErrorRate()
    if isinstance(predicted, list):
        predicted = " ".join(predicted)
    if isinstance(expected, list):
        expected = " ".join(expected)
    return cer_metric([predicted], [expected]).item()

def calculate_bleu(predicted, expected):
    """使用sacrebleu计算BLEU分数"""
    # 处理输入格式
    if isinstance(predicted, str):
        predicted = [predicted]
    if isinstance(expected, str):
        expected = [[expected]]
    else:
        expected = [[ref] for ref in expected]
    
    # 计算BLEU分数
    bleu = sacrebleu.corpus_bleu(predicted, expected)
    return bleu.score / 100.0  # sacrebleu返回0-100的分数，转换为0-1

if __name__ == '__main__':
    warnings.filterwarnings("ignore") # 忽略警告
    config = get_config() # 获取配置
    train_model(config) # 训练模型
