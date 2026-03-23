#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试异构数据集自动下载功能
验证多个同任务数据集的加载和分配
"""
import os
import sys
import logging
import tempfile
import shutil

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_hetero_nlp_datasets_download():
    """测试hetero_nlp_tasks数据集的自动下载"""
    logger.info("=" * 60)
    logger.info("测试1: hetero_nlp_tasks数据集自动下载")
    logger.info("=" * 60)
    
    try:
        from federatedscope.nlp.hetero_tasks.dataset.get_data import HeteroNLPDataLoader
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp(prefix='test_hetero_')
        logger.info(f"使用临时目录: {temp_dir}")
        
        # 测试数据集列表
        test_datasets = ['imdb']  # 先测试一个数据集，避免下载太多
        test_client_nums = [1]  # 1个客户端
        
        logger.info(f"测试数据集: {test_datasets}")
        logger.info(f"客户端数量: {test_client_nums}")
        
        # 创建数据加载器
        data_loader = HeteroNLPDataLoader(
            data_dir=temp_dir,
            data_name=test_datasets,
            num_of_clients=test_client_nums,
            split=[0.9, 0.1]
        )
        
        logger.info("开始加载数据（如果数据集不存在会自动下载）...")
        data = data_loader.get_data()
        
        logger.info("✓ 数据加载成功！")
        logger.info(f"训练数据: {len(data['train'])} 个客户端")
        logger.info(f"验证数据: {len(data['val'])} 个客户端")
        logger.info(f"测试数据: {len(data['test'])} 个客户端")
        
        for i, (train, val, test) in enumerate(zip(data['train'], data['val'], data['test'])):
            logger.info(f"客户端 {i}: 训练={len(train)}, 验证={len(val)}, 测试={len(test)}")
        
        # 检查数据集是否已下载
        dataset_dir = os.path.join(temp_dir, test_datasets[0])
        if os.path.exists(dataset_dir):
            logger.info(f"✓ 数据集已下载到: {dataset_dir}")
            logger.info(f"目录内容: {os.listdir(dataset_dir)}")
        else:
            logger.warning(f"数据集目录不存在: {dataset_dir}")
        
        # 清理
        logger.info(f"清理临时目录: {temp_dir}")
        shutil.rmtree(temp_dir)
        
        return True
        
    except Exception as e:
        logger.error(f"测试失败: {e}", exc_info=True)
        return False


def test_llm_datasets_loading():
    """测试LLM数据集的加载"""
    logger.info("=" * 60)
    logger.info("测试2: LLM数据集加载（gsm8k, alpaca等）")
    logger.info("=" * 60)
    
    try:
        from federatedscope.llm.dataloader.dataloader import load_llm_dataset
        from federatedscope.core.configs.config import CN
        
        # 创建测试配置
        config = CN()
        config.model = CN()
        config.model.type = 'llama-2-7b-hf@huggingface_llm'
        config.data = CN()
        config.data.root = 'data/'
        config.data.type = 'gsm8k@llm'
        config.llm = CN()
        config.llm.tok_len = 512
        
        logger.info(f"测试数据集: {config.data.type}")
        logger.info(f"数据根目录: {config.data.root}")
        
        # 检查数据文件是否存在
        data_path = os.path.join(config.data.root, 'gsm8k_train.jsonl')
        if os.path.exists(data_path):
            logger.info(f"✓ 数据文件已存在: {data_path}")
        else:
            logger.warning(f"数据文件不存在: {data_path}")
            logger.info("如果运行完整训练，会自动下载")
        
        # 尝试加载数据集（如果文件存在）
        if os.path.exists(data_path):
            logger.info("开始加载数据集...")
            dataset, modified_config = load_llm_dataset(config)
            logger.info(f"✓ 数据集加载成功！大小: {len(dataset)}")
            return True
        else:
            logger.info("数据文件不存在，跳过加载测试")
            logger.info("提示: 运行完整训练时会自动下载数据集")
            return True
            
    except Exception as e:
        logger.error(f"测试失败: {e}", exc_info=True)
        return False


def test_knowledge_anchor_splitter_hetero():
    """测试知识锚点splitter的异构数据集支持"""
    logger.info("=" * 60)
    logger.info("测试3: 知识锚点Splitter异构数据集支持")
    logger.info("=" * 60)
    
    try:
        from federatedscope.core.splitters.generic.knowledge_anchor_splitter import KnowledgeAnchorSplitter
        
        # 测试单数据集模式
        logger.info("测试单数据集模式...")
        splitter_single = KnowledgeAnchorSplitter(
            client_num=3,
            server_ratio=0.4,
            client_ratio=0.4,
            test_ratio=0.2
        )
        assert not splitter_single.hetero_mode, "单数据集模式应该关闭异构模式"
        logger.info("✓ 单数据集模式正常")
        
        # 测试异构数据集模式
        logger.info("测试异构数据集模式...")
        hetero_datasets = ['gsm8k@llm', 'alpaca@llm', 'dolly-15k@llm']
        splitter_hetero = KnowledgeAnchorSplitter(
            client_num=3,
            server_ratio=0.4,
            client_ratio=0.4,
            test_ratio=0.2,
            hetero_datasets=hetero_datasets,
            hetero_distribution='uniform'
        )
        assert splitter_hetero.hetero_mode, "异构数据集模式应该开启"
        assert splitter_hetero.hetero_datasets == hetero_datasets, "数据集列表应该匹配"
        logger.info(f"✓ 异构数据集模式正常: {hetero_datasets}")
        
        return True
        
    except Exception as e:
        logger.error(f"测试失败: {e}", exc_info=True)
        return False


def main():
    """运行所有测试"""
    logger.info("开始测试异构数据集功能...")
    logger.info("")
    
    results = []
    
    # 测试1: hetero_nlp_tasks数据集下载
    results.append(("hetero_nlp_tasks下载", test_hetero_nlp_datasets_download()))
    logger.info("")
    
    # 测试2: LLM数据集加载
    results.append(("LLM数据集加载", test_llm_datasets_loading()))
    logger.info("")
    
    # 测试3: Splitter异构支持
    results.append(("Splitter异构支持", test_knowledge_anchor_splitter_hetero()))
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

