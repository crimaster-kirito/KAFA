# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.configs.config import global_cfg
from federatedscope.core.auxiliaries.runner_builder import get_runner
from federatedscope.core.auxiliaries.worker_builder import get_server_cls, get_client_cls


class FedOptTest(unittest.TestCase):
    """
    测试联邦优化算法（FedOpt）在 standalone 模式下的运行情况。
    
    包含一个测试用例，用于验证 FedOpt 方法是否能正常训练并在测试集上达到预期性能。
    """

    def setUp(self):
        """
        在每个测试方法执行前打印当前正在运行的测试名称。
        """
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def set_config_fedopt(self, cfg):
        """
        配置 FedOpt 算法所需的参数。

        参数:
            cfg: 全局配置对象，将被修改以适应 FedOpt 的设置。

        返回:
            备份的原始配置对象，以便后续恢复。
        """
        backup_cfg = cfg.clone()

        import torch
        # 根据是否有 GPU 设置 use_gpu 参数
        cfg.use_gpu = torch.cuda.is_available()
        cfg.eval.freq = 10
        cfg.eval.metrics = ['acc']

        # 设置联邦学习为单机模式，所有客户端在本地运行
        cfg.federate.mode = 'standalone'
        # 设置每个客户端本地更新步数为5步
        cfg.train.local_update_steps = 5
        # 设置联邦学习总轮数为20轮
        cfg.federate.total_round_num = 20
        # 设置每轮联邦学习采样的客户端数量为5个
        cfg.federate.sample_client_num = 5
        # 设置参与联邦学习的总客户端数量为10个
        cfg.federate.client_num = 10
        # 设置联邦学习方法为FedOpt算法
        cfg.federate.method = 'FedOpt'

        # 设置数据根目录路径
        cfg.data.root = 'test_data/'
        # 设置数据类型为femnist（联邦EMNIST手写数字数据集）
        cfg.data.type = 'femnist'
        # 设置数据集划分比例：训练集60%，验证集20%，测试集20%
        cfg.data.splits = [0.6, 0.2, 0.2]
        # 设置数据批次大小为10
        cfg.data.batch_size = 10
        # 设置数据子采样比例为1%
        cfg.data.subsample = 0.01
        
        # 数据预处理变换：转为张量并归一化
        # ToTensor将PIL Image或numpy.ndarray转换为tensor，并将像素值从[0, 255]缩放到[0, 1]
        # Normalize对tensor进行标准化处理，mean和std分别是均值和标准差
        cfg.data.transform = [['ToTensor'],
                              [
                                  'Normalize', {
                                      'mean': [0.9637],
                                      'std': [0.1592]
                                  }
                              ]]

        # 设置模型类型为convnet2，隐藏层大小为2048，输出通道数为62（对应FEMNIST数据集的62个类别）
        cfg.model.type = 'convnet2'
        cfg.model.hidden = 2048
        cfg.model.out_channels = 62

        # 设置训练优化器的学习率为0.001，权重衰减为0.0（即不使用正则化）
        cfg.train.optimizer.lr = 0.001
        cfg.train.optimizer.weight_decay = 0.0

        # 设置损失函数为交叉熵损失函数，训练器类型为计算机视觉训练器
        cfg.criterion.type = 'CrossEntropyLoss'
        cfg.trainer.type = 'cvtrainer'
        # 设置随机种子为123以保证实验结果的可复现性
        cfg.seed = 123

        cfg.fedopt.use = True
        cfg.fedopt.optimizer.lr = 1.

        return backup_cfg

    def test_fedopt_standalone(self):
        """
        测试 FedOpt 在 standalone 模式下能否成功运行，并检查最终损失是否低于阈值。
        
        此测试会初始化配置、加载数据、构建服务器与客户端、启动训练流程，
        并断言测试损失小于设定的上限（600），确保模型收敛性。
        """
        # 初始化配置并设置 FedOpt 相关参数
        init_cfg = global_cfg.clone()
        backup_cfg = self.set_config_fedopt(init_cfg)
        # 设置随机种子以确保实验可重现
        setup_seed(init_cfg.seed)
        # 更新日志配置
        update_logger(init_cfg, True)

        # 加载数据并更新配置
        data, modified_cfg = get_data(init_cfg.clone())
        init_cfg.merge_from_other_cfg(modified_cfg)
        self.assertIsNotNone(data)

        # 构建联邦学习运行器
        Fed_runner = get_runner(data=data,
                                server_class=get_server_cls(init_cfg),
                                client_class=get_client_cls(init_cfg),
                                config=init_cfg.clone())
        self.assertIsNotNone(Fed_runner)
        test_results = Fed_runner.run()
        init_cfg.merge_from_other_cfg(backup_cfg)

        # 断言测试损失应低于 600
        self.assertLess(
            test_results['client_summarized_weighted_avg']['test_loss'], 600)


if __name__ == '__main__':
    unittest.main()