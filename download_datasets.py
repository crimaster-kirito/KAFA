#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集下载脚本
自动下载所有需要的数据集到本地
"""
import os
import urllib.request
import zipfile
import shutil
import sys

def download_file(url, dest_path, filename=None, show_progress=True):
    """下载文件"""
    if filename is None:
        filename = os.path.basename(url)
    
    file_path = os.path.join(dest_path, filename)
    
    if os.path.exists(file_path):
        print(f"✓ 文件已存在: {file_path}")
        return file_path
    
    print(f"下载: {filename}")
    print(f"  从: {url}")
    print(f"  到: {file_path}")
    
    try:
        os.makedirs(dest_path, exist_ok=True)
        
        def show_progress_hook(count, block_size, total_size):
            if show_progress and total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                sys.stdout.write(f"\r  进度: {percent}%")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, file_path, reporthook=show_progress_hook if show_progress else None)
        if show_progress:
            print()  # 换行
        
        print(f"✓ 下载完成: {file_path}")
        return file_path
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        raise

def download_llm_datasets():
    """下载LLM数据集"""
    print("=" * 60)
    print("下载LLM数据集到 data/ 目录")
    print("=" * 60)
    
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    
    datasets = {
        'gsm8k_train.jsonl': {
            'url': 'https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl',
            'temp_name': 'train.jsonl',  # 需要重命名
            'required': True  # 知识锚点必需
        },
        'alpaca_data.json': {
            'url': 'https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/761dc5bfbdeeffa89b8bff5d038781a4055f796a/alpaca_data.json',
            'required': False
        },
        'alpaca_data_cleaned.json': {
            'url': 'https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/a7d629079a95c2e4b7ec7dfe55087fbd18d9eba8/alpaca_data_cleaned.json',
            'required': False
        },
        'databricks-dolly-15k.jsonl': {
            'url': 'https://raw.githubusercontent.com/databrickslabs/dolly/d000e3030970379aabbf6d291f50ffdd3b715b64/data/databricks-dolly-15k.jsonl',
            'required': False
        },
        'rosetta_alpaca.json': {
            'url': 'https://raw.githubusercontent.com/sahil280114/codealpaca/d269da106a579a623a654529b3cb91b5dfa9c72f/data/rosetta_alpaca.json',
            'required': False
        },
    }
    
    for filename, info in datasets.items():
        required = info.get('required', False)
        marker = "[必需]" if required else "[可选]"
        print(f"\n{marker} {filename}")
        
        if filename == 'gsm8k_train.jsonl':
            # GSM8K需要重命名
            temp_file = download_file(info['url'], data_dir, info['temp_name'])
            if os.path.exists(temp_file):
                final_path = os.path.join(data_dir, filename)
                if os.path.exists(final_path):
                    os.remove(final_path)
                os.rename(temp_file, final_path)
                print(f"✓ 已重命名为: {final_path}")
        else:
            download_file(info['url'], data_dir, filename)
    
    print("\n" + "=" * 60)
    print("LLM数据集下载完成！")
    print("=" * 60)

def download_hetero_datasets():
    """下载Hetero NLP Tasks数据集"""
    print("\n" + "=" * 60)
    print("下载Hetero NLP Tasks数据集到 datasets/ 目录")
    print("=" * 60)
    print("注意: 这些数据集用于异构任务实验，知识锚点算法不需要")
    
    datasets_dir = 'datasets'
    base_url = 'https://federatedscope.oss-cn-beijing.aliyuncs.com'
    
    datasets = ['imdb', 'agnews', 'squad', 'newsqa', 'cnndm', 'msqg']
    
    for dataset in datasets:
        zip_url = f'{base_url}/{dataset}.zip'
        zip_path = os.path.join(datasets_dir, f'{dataset}.zip')
        dataset_dir = os.path.join(datasets_dir, dataset)
        
        # 检查是否已存在
        if os.path.exists(dataset_dir):
            print(f"\n✓ {dataset} 已存在: {dataset_dir}")
            continue
        
        print(f"\n下载 {dataset} 数据集...")
        try:
            download_file(zip_url, datasets_dir, f'{dataset}.zip')
            
            # 解压
            print(f"解压 {dataset}...")
            extract_dir = os.path.join(datasets_dir, dataset + '_raw')
            os.makedirs(extract_dir, exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # 移动文件
            extracted_dataset = os.path.join(extract_dir, dataset)
            if os.path.exists(extracted_dataset):
                shutil.move(extracted_dataset, datasets_dir)
            
            # 清理
            if os.path.exists(zip_path):
                os.remove(zip_path)
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
            
            print(f"✓ {dataset} 完成")
        except Exception as e:
            print(f"✗ {dataset} 下载失败: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("Hetero NLP Tasks数据集下载完成！")
    print("=" * 60)

def main():
    """主函数"""
    print("数据集下载脚本")
    print("=" * 60)
    print("此脚本将下载所有数据集到本地")
    print("=" * 60)
    
    # 询问用户
    print("\n请选择要下载的数据集类型：")
    print("1. LLM数据集（知识锚点算法需要）")
    print("2. Hetero NLP Tasks数据集（异构任务实验）")
    print("3. 全部下载")
    print("4. 仅下载必需数据集（gsm8k）")
    
    choice = input("\n请输入选择 (1/2/3/4，默认3): ").strip() or "3"
    
    if choice == "1":
        download_llm_datasets()
    elif choice == "2":
        download_hetero_datasets()
    elif choice == "3":
        download_llm_datasets()
        download_hetero_datasets()
    elif choice == "4":
        # 仅下载gsm8k
        data_dir = 'data'
        os.makedirs(data_dir, exist_ok=True)
        url = 'https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl'
        temp_file = download_file(url, data_dir, 'train.jsonl')
        if os.path.exists(temp_file):
            final_path = os.path.join(data_dir, 'gsm8k_train.jsonl')
            if os.path.exists(final_path):
                os.remove(final_path)
            os.rename(temp_file, final_path)
            print(f"✓ 已重命名为: {final_path}")
    else:
        print("无效选择，退出")
        return
    
    print("\n" + "=" * 60)
    print("✓ 下载任务完成！")
    print("=" * 60)
    print("\n数据集位置：")
    print("- LLM数据集: data/")
    print("- Hetero NLP Tasks数据集: datasets/")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n下载已取消")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n错误: {e}")
        sys.exit(1)

