#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版测试脚本：测试数据集自动下载功能
不依赖完整的FederatedScope环境
"""
import os
import sys
import logging
import tempfile
import shutil
import urllib.request
import zipfile

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_url(url, dest_path, filename=None):
    """下载文件"""
    if filename is None:
        filename = os.path.basename(url)
    
    file_path = os.path.join(dest_path, filename)
    
    if os.path.exists(file_path):
        logger.info(f"文件已存在: {file_path}")
        return file_path
    
    logger.info(f"开始下载: {url}")
    logger.info(f"保存到: {file_path}")
    
    try:
        os.makedirs(dest_path, exist_ok=True)
        urllib.request.urlretrieve(url, file_path)
        logger.info(f"✓ 下载完成: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"下载失败: {e}")
        raise


def test_hetero_nlp_dataset_download():
    """测试hetero_nlp_tasks数据集的自动下载"""
    logger.info("=" * 60)
    logger.info("测试: hetero_nlp_tasks数据集自动下载")
    logger.info("=" * 60)
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix='test_hetero_')
    logger.info(f"使用临时目录: {temp_dir}")
    
    try:
        # 测试数据集
        test_dataset = 'imdb'
        dataset_dir = os.path.join(temp_dir, test_dataset)
        
        # 检查数据集是否存在
        if os.path.exists(dataset_dir):
            logger.info(f"数据集已存在: {dataset_dir}")
        else:
            logger.info(f"数据集不存在，将自动下载: {dataset_dir}")
            
            # 下载URL（与FederatedScope中的一致）
            url = f'https://federatedscope.oss-cn-beijing.aliyuncs.com/{test_dataset}.zip'
            
            # 下载
            zip_path = download_url(url, temp_dir, f'{test_dataset}.zip')
            
            # 解压
            logger.info(f"解压文件: {zip_path}")
            raw_dir = os.path.join(temp_dir, test_dataset + '_raw')
            extract_dir = os.path.join(temp_dir, test_dataset)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(raw_dir)
            
            # 移动文件
            if os.path.exists(os.path.join(raw_dir, test_dataset)):
                import shutil
                shutil.move(os.path.join(raw_dir, test_dataset), temp_dir)
            
            # 清理
            if os.path.exists(zip_path):
                os.remove(zip_path)
            if os.path.exists(raw_dir):
                shutil.rmtree(raw_dir)
            
            logger.info(f"✓ 数据集下载并解压完成: {dataset_dir}")
        
        # 检查目录结构
        if os.path.exists(dataset_dir):
            logger.info(f"数据集目录内容: {os.listdir(dataset_dir)}")
            
            # 检查是否有train和test目录
            train_dir = os.path.join(dataset_dir, 'train')
            test_dir = os.path.join(dataset_dir, 'test')
            
            if os.path.exists(train_dir):
                logger.info(f"✓ 训练数据目录存在: {train_dir}")
                logger.info(f"  内容: {os.listdir(train_dir)}")
            
            if os.path.exists(test_dir):
                logger.info(f"✓ 测试数据目录存在: {test_dir}")
                logger.info(f"  内容: {os.listdir(test_dir)}")
        
        logger.info("✓ 测试通过！")
        return True
        
    except Exception as e:
        logger.error(f"测试失败: {e}", exc_info=True)
        return False
        
    finally:
        # 询问是否保留临时目录
        logger.info(f"\n临时目录: {temp_dir}")
        logger.info("测试完成后可以手动删除，或保留用于后续测试")
        # 不自动删除，让用户决定
        # shutil.rmtree(temp_dir)


def test_llm_dataset_info():
    """测试LLM数据集信息"""
    logger.info("=" * 60)
    logger.info("测试: LLM数据集信息")
    logger.info("=" * 60)
    
    data_root = 'data/'
    
    # 检查常见LLM数据集
    datasets = {
        'gsm8k': 'gsm8k_train.jsonl',
        'alpaca': 'alpaca_data.json',
        'dolly-15k': 'databricks-dolly-15k.jsonl'
    }
    
    logger.info(f"数据根目录: {os.path.abspath(data_root)}")
    
    for dataset_name, filename in datasets.items():
        file_path = os.path.join(data_root, filename)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            logger.info(f"✓ {dataset_name}: {file_path} ({size:.2f} MB)")
        else:
            logger.info(f"✗ {dataset_name}: {file_path} (不存在)")
            logger.info(f"  提示: 运行完整训练时会自动下载")
    
    return True


def main():
    """运行所有测试"""
    logger.info("开始测试数据集下载功能...")
    logger.info("")
    
    results = []
    
    # 测试1: hetero_nlp_tasks数据集下载
    results.append(("hetero_nlp_tasks下载", test_hetero_nlp_dataset_download()))
    logger.info("")
    
    # 测试2: LLM数据集信息
    results.append(("LLM数据集信息", test_llm_dataset_info()))
    logger.info("")
    
    # 汇总结果
    logger.info("=" * 60)
    logger.info("测试结果汇总")
    logger.info("=" * 60)
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        logger.info("")
        logger.info("✓ 所有测试通过！")
    else:
        logger.info("")
        logger.warning("✗ 部分测试失败，请检查错误信息")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

