# Copyright (c) Alibaba, Inc. and its affiliates.

import argparse
import os
import sys
import unittest

# 获取当前脚本文件的父目录路径，用于构建项目根目录路径
file_dir = os.path.join(os.path.dirname(__file__), '..')
# 将项目根目录添加到Python模块搜索路径中，以便能够导入项目中的模块
sys.path.append(file_dir)

# 创建命令行参数解析器，用于解析测试运行器的相关参数
parser = argparse.ArgumentParser('test runner')
# 添加--list_tests参数，用于控制是否列出所有测试用例
parser.add_argument('--list_tests', action='store_true', help='list all tests')
# 添加--pattern参数，用于指定测试文件的匹配模式，默认为'test_*.py'
parser.add_argument('--pattern', default='test_*.py', help='test file pattern')
# 添加--test_dir参数，用于指定测试目录，默认为'tests'
parser.add_argument('--test_dir',
                    default='tests',
                    help='directory to be tested')
# 解析命令行参数
args = parser.parse_args()

def gather_test_cases(test_dir, pattern, list_tests):
    """
    收集测试用例并构建测试套件
    
    参数:
        test_dir (str): 测试文件所在的目录路径
        pattern (str): 匹配测试文件的模式，如 'test_*.py'
        list_tests (bool): 是否打印发现的测试用例列表
    
    返回:
        unittest.TestSuite: 包含所有发现测试用例的测试套件
    """
    test_suite = unittest.TestSuite()
    # 使用unittest默认加载器发现测试用例
    discover = unittest.defaultTestLoader.discover(test_dir,
                                                   pattern=pattern,
                                                   top_level_dir=None)
    # 遍历所有发现的测试套件和测试用例
    for suite_discovered in discover:

        for test_case in suite_discovered:
            test_suite.addTest(test_case)
            # 处理可迭代的测试用例（可能包含子用例）
            if hasattr(test_case, '__iter__'):
                for subcase in test_case:
                    if list_tests:
                        print(subcase)
            else:
                # 打印单个测试用例
                if list_tests:
                    print(test_case)
    return test_suite


def main():
    """
    主函数，负责执行测试套件
    该函数创建测试运行器，收集测试用例，并执行测试。如果测试失败则退出程序。
    """
    # 创建文本测试运行器
    runner = unittest.TextTestRunner()
    
    # 收集测试用例，构建测试套件
    test_suite = gather_test_cases(os.path.abspath(args.test_dir),
                                   args.pattern, args.list_tests)
    
    # 如果不是仅列出测试用例，则执行测试
    if not args.list_tests:
        res = runner.run(test_suite)
        # 如果测试不成功，则退出程序返回错误码1
        if not res.wasSuccessful():
            exit(1)


if __name__ == '__main__':
    main()
