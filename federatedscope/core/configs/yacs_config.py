# An extended configuration system based on [yacs],
# modified by Alibaba Group
#
#
# Yacs' license is as follows:
#
# Copyright (c) 2018-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
"""
    An extended configuration system based on [yacs](
        https://github.com/rbgirshick/yacs).

    We enhance yacs with more functionalities to support FederatedScope
    configureation, such as more flexible type conversion, help info,
    allowing required attributes, etc.,

"""

import copy
import io
import logging
import os
import sys
from ast import literal_eval

import yaml

logger = logging.getLogger(__name__)

# Flag for py2 and py3 compatibility to use when separate code paths are
# necessary. When _PY2 is False, we assume Python 3 is in use
_PY2 = sys.version_info.major == 2

# Filename extensions for loading configs from files
_YAML_EXTS = {"", ".yaml", ".yml"}
_PY_EXTS = {".py"}
_FILE_TYPES = (io.IOBase, )

# py2 and py3 compatibility for checking file object type
# We simply use this to infer py2 vs py3

# Utilities for importing modules from file paths
if _PY2:
    # imp is available in both py2 and py3 for now, but is deprecated in py3
    import imp
else:
    import importlib.util

logger = logging.getLogger(__name__)


class Argument:
    """
    参数配置项类，用于定义配置参数的属性
    
    该类封装了配置项的各种属性，包括默认值、描述信息、类型和是否必需等信息。
    """
    
    def __init__(self,
                 default_value=None,
                 description='',
                 required_type=None,
                 required=False):
        """
        初始化Argument实例
        
        Args:
            default_value: 配置项的默认值
            description (str): 配置项的描述信息
            required_type (type): 配置项要求的数据类型
            required (bool): 配置项是否为必需项
        """
        self.value = default_value
        self.description = description
        self.required = required
        if required_type is not None:
            assert required_type in _VALID_TYPES, ""
            self.type = required_type
        else:
            assert type(default_value) in _VALID_TYPES, ""
            self.type = type(default_value)

    def __str__(self):
        return 'Required({})'.format(self.type.__name__) \
            if self.required else str(self.value)

    def __repr__(self):
        return 'Required({})'.format(self.type.__name__) \
            if self.required else str(self.value)


# CfgNodes can only contain a limited set of valid types
# 定义配置节点中允许包含的有效数据类型集合
_VALID_TYPES = {tuple, list, str, int, float, bool, type(None), Argument, dict}
# py2 allow for str and unicode
if _PY2:
    _VALID_TYPES = _VALID_TYPES.union({unicode})  # noqa: F821

# allow int <-> float conversation
# 定义允许进行类型转换的数据类型对
casts = [(tuple, list), (list, tuple), (int, float), (float, int)]


class CfgNode(dict):
    """
    CfgNode represents an internal node in the configuration tree. It's a \
    simple dict-like container that allows for attribute-based access to keys.
    """

    # 标记配置节点是否为不可变（冻结）状态的键名
    IMMUTABLE = "__immutable__"
    # 存储已弃用配置键的键名
    DEPRECATED_KEYS = "__deprecated_keys__"
    # 存储重命名配置键映射关系的键名
    RENAMED_KEYS = "__renamed_keys__"
    # 标记是否允许添加新键的键名
    NEW_ALLOWED = "__new_allowed__"

    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        """
        初始化CfgNode实例
        
        Args:
            init_dict (dict): 用于初始化CfgNode的可能嵌套的字典
            key_list (list[str]): 从根节点索引到此CfgNode的名称列表，主要用于日志记录
            new_allowed (bool): 在与其他配置合并时是否允许添加新键
        """
        # 递归地将init_dict中的嵌套字典转换为CfgNode
        init_dict = {} if init_dict is None else init_dict
        key_list = [] if key_list is None else key_list
        init_dict = self._create_config_tree_from_dict(init_dict, key_list)
        super(CfgNode, self).__init__(init_dict)
        # 管理CfgNode是否处于冻结状态
        self.__dict__[CfgNode.IMMUTABLE] = False
        # 已弃用的配置选项
        # 如果从代码中删除了某个选项，但又不想破坏现有的yaml配置文件，
        # 可以将完整的配置键作为字符串添加到下面的集合中
        self.__dict__[CfgNode.DEPRECATED_KEYS] = set()
        # 重命名的选项
        # 如果重命名了配置选项，请在下面的字典中记录从旧名称到新名称的映射。
        # 可选地，如果类型也发生了更改，可以使值成为一个元组，
        # 该元组首先指定重命名的键，然后指定如何编辑配置文件的说明
        self.__dict__[CfgNode.RENAMED_KEYS] = {
            # 'EXAMPLE.OLD.KEY': 'EXAMPLE.NEW.KEY',  # Dummy example to follow
            # 'EXAMPLE.OLD.KEY': (
            #                               # A more complex example to follow
            #     'EXAMPLE.NEW.KEY',
            #     "Also convert to a tuple, e.g., 'foo' -> ('foo',) or "
            #     + "'foo:bar' -> ('foo', 'bar')"
            # ),
        }

        # 初始化后允许添加新属性
        self.__dict__[CfgNode.NEW_ALLOWED] = new_allowed

    @classmethod
    def _create_config_tree_from_dict(cls, dic, key_list):
        """
        Create a configuration tree using the given dict.
        Any dict-like objects inside dict will be treated as a new CfgNode.

        Args:
            dic (dict): ``dict`` to be converted
            key_list (list[str]): a list of names which index this CfgNode \
            from the root. Currently only used for logging purposes.
        """
        dic = copy.deepcopy(dic)
        for k, v in dic.items():
            if isinstance(v, dict):
                # Convert dict to CfgNode
                dic[k] = cls(v, key_list=key_list + [k])
            else:
                # Check for valid leaf type or nested CfgNode
                _assert_with_logging(
                    _valid_type(v, allow_cfg_node=False),
                    "Key {} with value {} is not a valid type; valid types: {}"
                    .format(".".join(key_list + [str(k)]), type(v),
                            _VALID_TYPES),
                )
        return dic

    def __getattr__(self, name):
        """
        获取配置项的值
        
        Args:
            name (str): 配置项名称
            
        Returns:
            配置项对应的值
            
        Raises:
            AttributeError: 当配置项不存在时抛出异常
        """
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        """
        设置配置项的值
        
        Args:
            name (str): 配置项名称
            value: 配置项的值
            
        Raises:
            AttributeError: 当CfgNode处于冻结状态或尝试修改内部状态时抛出异常
        """
        if self.is_frozen():
            raise AttributeError(
                "Attempted to set {} to {}, but CfgNode is immutable".format(
                    name, value))

        _assert_with_logging(
            name not in self.__dict__,
            "Invalid attempt to modify internal CfgNode state: {}".format(
                name),
        )
        _assert_with_logging(
            _valid_type(value, allow_cfg_node=True),
            "Invalid type {} for key {}; valid types = {}".format(
                type(value), name, _VALID_TYPES),
        )

        self[name] = value

    def __str__(self):
        """
        将CfgNode转换为字符串表示
        
        Returns:
            str: CfgNode的字符串表示形式
        """
        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        r = ""
        s = []
        for k, v in sorted(self.items()):
            seperator = "\n" if isinstance(v, CfgNode) else " "
            attr_str = "{}:{}{}".format(str(k), seperator, str(v))
            attr_str = _indent(attr_str, 2)
            s.append(attr_str)
        r += "\n".join(s)
        return r

    def __repr__(self):
        """
        返回CfgNode的官方字符串表示
        
        Returns:
            str: CfgNode的官方字符串表示
        """
        return "{}({})".format(self.__class__.__name__,
                               super(CfgNode, self).__repr__())

    def dump(self, **kwargs):
        """
        将配置转储为字符串
        
        Returns:
            str: YAML格式的配置字符串
        """
        def convert_to_dict(cfg_node, key_list):
            """
            递归地将CfgNode转换为普通字典
            
            Args:
                cfg_node: 要转换的CfgNode或值
                key_list (list): 键列表，用于错误信息
                
            Returns:
                转换后的字典或值
            """
            if not isinstance(cfg_node, CfgNode):
                _assert_with_logging(
                    _valid_type(cfg_node),
                    "Key {} with value {} is not a valid type; valid types: {}"
                    .format(".".join(key_list), type(cfg_node), _VALID_TYPES),
                )
                return cfg_node
            else:
                cfg_dict = dict(cfg_node)
                for k, v in cfg_dict.items():
                    cfg_dict[k] = convert_to_dict(v, key_list + [k])
                return cfg_dict

        self_as_dict = convert_to_dict(self, [])
        return yaml.safe_dump(self_as_dict, **kwargs)

    def merge_from_file(self, cfg_filename):
        """
        从YAML配置文件加载配置并合并到当前CfgNode
        
        Args:
            cfg_filename (str): 配置文件路径
        """
        with open(cfg_filename, "r") as f:
            cfg = self.load_cfg(f)
        self.merge_from_other_cfg(cfg)

    def merge_from_other_cfg(self, cfg_other):
        """
        将另一个CfgNode合并到当前CfgNode
        
        Args:
            cfg_other (CfgNode): 要合并的配置对象
        """
        _merge_a_into_b(cfg_other, self, self, [])

    def merge_from_list(self, cfg_list):
        """
        从列表（例如来自命令行）合并配置（键、值）到当前CfgNode。
        例如，cfg_list = ['FOO.BAR', 0.5]。
        
        Args:
            cfg_list (list): 配置键值对列表，格式为[key1, value1, key2, value2, ...]
        """
        _assert_with_logging(
            len(cfg_list) % 2 == 0,
            "Override list has odd length: {}; it must be a list of pairs".
            format(cfg_list),
        )
        root = self
        for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
            if root.key_is_deprecated(full_key):
                continue
            if root.key_is_renamed(full_key):
                root.raise_key_rename_error(full_key)
            key_list = full_key.split(".")
            d = self
            for subkey in key_list[:-1]:
                _assert_with_logging(subkey in d,
                                     "Non-existent key: {}".format(full_key))
                d = d[subkey]
            subkey = key_list[-1]
            _assert_with_logging(subkey in d or d.is_new_allowed(),
                                 "Non-existent key: {}".format(full_key))
            value = self._decode_cfg_value(v)
            value = _check_and_coerce_cfg_value_type(value, d[subkey], subkey,
                                                     full_key)
            d[subkey] = value

    def freeze(self):
        """
        使当前CfgNode及其所有子节点变为不可变（冻结状态）
        冻结后的配置节点不能被修改
        """
        self._immutable(True)

    def defrost(self):
        """
        使当前CfgNode及其所有子节点变为可变（解冻状态）
        解冻后的配置节点可以被修改
        """
        self._immutable(False)

    def is_frozen(self):
        """
        检查当前CfgNode是否处于冻结状态
        
        Returns:
            bool: 如果处于冻结状态返回True，否则返回False
        """
        return self.__dict__[CfgNode.IMMUTABLE]

    def _immutable(self, is_immutable):
        """
        设置当前CfgNode及其所有嵌套CfgNode的可变性状态
        
        Args:
            is_immutable (bool): True表示设置为不可变（冻结），False表示设置为可变（解冻）
        """
        self.__dict__[CfgNode.IMMUTABLE] = is_immutable
        # 递归地设置不可变状态
        for v in self.__dict__.values():
            if isinstance(v, CfgNode):
                v._immutable(is_immutable)
        for v in self.values():
            if isinstance(v, CfgNode):
                v._immutable(is_immutable)

    def clone(self):
        """
        递归地复制当前CfgNode
        
        Returns:
            CfgNode: 当前CfgNode的深拷贝副本
        """
        return copy.deepcopy(self)

    def register_deprecated_key(self, key):
        """
        将指定键注册为已弃用选项。当合并已弃用的键时会生成警告并忽略该键。
        
        Args:
            key (str): 要标记为已弃用的配置键（如'FOO.BAR'）
        """
        _assert_with_logging(
            key not in self.__dict__[CfgNode.DEPRECATED_KEYS],
            "key {} is already registered as a deprecated key".format(key),
        )
        self.__dict__[CfgNode.DEPRECATED_KEYS].add(key)

    def register_renamed_key(self, old_name, new_name, message=None):
        """
        将键注册为从old_name重命名为new_name。当合并重命名的键时，
        会抛出异常提醒用户该键已被重命名。
        
        Args:
            old_name (str): 旧的配置键名
            new_name (str): 新的配置键名
            message (str, optional): 可选的附加信息，说明如何编辑配置文件
        """
        _assert_with_logging(
            old_name not in self.__dict__[CfgNode.RENAMED_KEYS],
            "key {} is already registered as a renamed cfg key".format(
                old_name),
        )
        value = new_name
        if message:
            value = (new_name, message)
        self.__dict__[CfgNode.RENAMED_KEYS][old_name] = value

    def key_is_deprecated(self, full_key):
        """
        检查指定键是否已弃用
        
        Args:
            full_key (str): 完整的配置键路径
            
        Returns:
            bool: 如果键已弃用返回True，否则返回False
        """
        if full_key in self.__dict__[CfgNode.DEPRECATED_KEYS]:
            logger.warning(
                "Deprecated config key (ignoring): {}".format(full_key))
            return True
        return False

    def key_is_renamed(self, full_key):
        """
        检查指定键是否已重命名
        
        Args:
            full_key (str): 完整的配置键路径
            
        Returns:
            bool: 如果键已重命名返回True，否则返回False
        """
        return full_key in self.__dict__[CfgNode.RENAMED_KEYS]

    def raise_key_rename_error(self, full_key):
        """
        抛出键重命名错误异常
        
        Args:
            full_key (str): 完整的配置键路径
            
        Raises:
            KeyError: 包含重命名信息的键错误异常
        """
        new_key = self.__dict__[CfgNode.RENAMED_KEYS][full_key]
        if isinstance(new_key, tuple):
            msg = " Note: " + new_key[1]
            new_key = new_key[0]
        else:
            msg = ""
        raise KeyError(
            "Key {} was renamed to {}; please update your config.{}".format(
                full_key, new_key, msg))

    def is_new_allowed(self):
        """
        检查是否允许添加新键
        
        Returns:
            bool: 如果允许添加新键返回True，否则返回False
        """
        return self.__dict__[CfgNode.NEW_ALLOWED]

    def set_new_allowed(self, is_new_allowed):
        """
        设置当前配置（及其递归子配置）是否允许从其他配置合并新键
        
        Args:
            is_new_allowed (bool): True表示允许添加新键，False表示不允许
        """
        self.__dict__[CfgNode.NEW_ALLOWED] = is_new_allowed
        # 递归地设置new_allowed状态
        for v in self.__dict__.values():
            if isinstance(v, CfgNode):
                v.set_new_allowed(is_new_allowed)
        for v in self.values():
            if isinstance(v, CfgNode):
                v.set_new_allowed(is_new_allowed)

    @classmethod
    def load_cfg(cls, cfg_file_obj_or_str):
        """
        加载配置对象
        
        支持从多种来源加载配置，包括YAML文件、Python源文件或YAML格式的字符串。
        该方法是加载配置的主要入口点，会根据输入类型自动选择合适的加载方式。
        
        Args:
            cfg_file_obj_or_str (str or file): 支持从以下方式加载配置:
                - 由YAML文件支持的文件对象
                - 由导出"cfg"属性（为dict或CfgNode类型）的Python源文件支持的文件对象
                - 可解析为有效YAML的字符串
                
        Returns:
            CfgNode: 加载并解析后的配置对象
            
        Raises:
            NotImplementedError: 当输入类型不支持时抛出异常（理论上不应到达此处）
        """
        _assert_with_logging(
            isinstance(cfg_file_obj_or_str, _FILE_TYPES + (str, )),
            "Expected first argument to be of type {} or {}, but it was {}".
            format(_FILE_TYPES, str, type(cfg_file_obj_or_str)),
        )
        if isinstance(cfg_file_obj_or_str, str):
            return cls._load_cfg_from_yaml_str(cfg_file_obj_or_str)
        elif isinstance(cfg_file_obj_or_str, _FILE_TYPES):
            return cls._load_cfg_from_file(cfg_file_obj_or_str)
        else:
            raise NotImplementedError(
                "Impossible to reach here (unless there's a bug)")

    @classmethod
    def _load_cfg_from_file(cls, file_obj):
        """
        从YAML文件或Python源文件加载配置
        
        根据文件扩展名判断文件类型，并调用相应的加载方法。支持.yaml、.yml和.py扩展名。
        对于YAML文件，读取内容后作为字符串传递给_yaml_str加载器；
        对于Python文件，使用模块导入机制加载并验证其包含正确的cfg属性。
        
        Args:
            file_obj: 文件对象，必须具有name属性以获取文件名和扩展名
            
        Returns:
            CfgNode: 加载的配置对象
            
        Raises:
            Exception: 当文件类型不支持时抛出异常，提示用户仅支持特定文件类型
        """
        _, file_extension = os.path.splitext(file_obj.name)
        if file_extension in _YAML_EXTS:
            return cls._load_cfg_from_yaml_str(file_obj.read())
        elif file_extension in _PY_EXTS:
            return cls._load_cfg_py_source(file_obj.name)
        else:
            raise Exception(
                "Attempt to load from an unsupported file type {}; "
                "only {} are supported".format(file_obj,
                                               _YAML_EXTS.union(_PY_EXTS)))

    @classmethod
    def _load_cfg_from_yaml_str(cls, str_obj):
        """
        从YAML字符串编码加载配置
        
        将YAML格式的字符串解析为Python字典，然后转换为CfgNode对象。
        这是处理YAML内容的核心方法，被文件加载器和其他组件广泛使用。
        
        Args:
            str_obj (str): YAML格式的字符串，包含有效的YAML语法
            
        Returns:
            CfgNode: 由YAML字符串解析得到的配置对象
        """
        cfg_as_dict = yaml.safe_load(str_obj)
        return cls(cfg_as_dict)

    @classmethod
    def _load_cfg_py_source(cls, filename):
        """
        从Python源文件加载配置
        
        动态导入指定的Python文件作为模块，并从中提取cfg属性。
        要求目标Python文件必须定义名为'cfg'的全局变量，且该变量必须是字典或CfgNode类型。
        这种方式允许使用Python代码的全部功能来构建复杂的配置逻辑。
        
        Args:
            filename (str): Python源文件的路径
            
        Returns:
            CfgNode: 从Python文件中加载的配置对象
            
        Raises:
            AssertionError: 当文件中没有'cfg'属性，或'cfg'属性类型不正确时抛出
        """
        module = _load_module_from_file("yacs.config.override", filename)
        _assert_with_logging(
            hasattr(module, "cfg"),
            "Python module from file {} must have 'cfg' attr".format(filename),
        )
        VALID_ATTR_TYPES = {dict, CfgNode}
        _assert_with_logging(
            type(module.cfg) in VALID_ATTR_TYPES,
            "Imported module 'cfg' attr must be in {} but is {} instead".
            format(VALID_ATTR_TYPES, type(module.cfg)),
        )
        return cls(module.cfg)

    @classmethod
    def _decode_cfg_value(cls, value):
        """
        将原始配置值解码为Python对象
        
        处理来自不同来源（如YAML文件或命令行参数）的原始配置值，将其转换为适当的Python数据类型。
        这是配置系统类型处理的核心部分，确保所有配置值都具有正确的类型。
        
        解码规则:
        (1) 如果值是字典，它将被递归地转换为新的CfgNode实例
        (2) 如果值是字符串，尝试使用literal_eval解析为Python字面量
        (3) 其他类型的值按原样返回
        
        特别注意对字符串的处理：使用literal_eval可以安全地将字符串表示的数字、
        布尔值、列表、元组、字典等转换为对应的Python对象，同时保留纯文本字符串不变。
        
        Args:
            value: 原始配置值，可以是任何类型
            
        Returns:
            解码后的Python对象，可能是CfgNode、基本数据类型或其组合
        """
        # Configs parsed from raw yaml will contain dictionary keys that need
        # to be converted to CfgNode objects
        if isinstance(value, dict):
            return cls(value)
        # All remaining processing is only applied to strings
        if not isinstance(value, str):
            return value
        # Try to interpret `value` as a:
        #   string, number, tuple, list, dict, boolean, or None
        try:
            value = literal_eval(value)
        # The following two excepts allow v to pass through when it represents
        # a string.
        #
        # Longer explanation:
        # The type of v is always a string (before calling literal_eval), but
        # sometimes it *represents* a string and other times a data structure,
        # like a list. In the case that v represents a string, what we got
        # back from the yaml parser is 'foo' *without quotes* (so,
        # not '"foo"'). literal_eval is ok with '"foo"', but will raise a
        # ValueError if given 'foo'. In other cases, like paths (v =
        # 'foo/bar' and not v = '"foo/bar"'), literal_eval
        # will raise a SyntaxError.
        except ValueError:
            pass
        except SyntaxError:
            pass
        return value


def _valid_type(value, allow_cfg_node=False):
    """
    检查值是否为有效类型
    
    Args:
        value: 要检查的值
        allow_cfg_node (bool): 是否允许CfgNode类型
        
    Returns:
        bool: 如果值是有效类型返回True，否则返回False
    """
    return (type(value) in _VALID_TYPES) or (allow_cfg_node
                                             and isinstance(value, CfgNode))


def _merge_a_into_b(a, b, root, key_list):
    """
    [从yacs修改而来，允许int <-> float转换]
    
    将配置字典a合并到配置字典b中，当a中也指定了b中的选项时会覆盖b中的选项
    
    Args:
        a (CfgNode): 源配置节点
        b (CfgNode): 目标配置节点
        root (CfgNode): 根配置节点
        key_list (list): 键列表，用于构建完整键路径
    """
    _assert_with_logging(
        isinstance(a, CfgNode),
        "`a` (cur type {}) must be an instance of {}".format(type(a), CfgNode),
    )
    _assert_with_logging(
        isinstance(b, CfgNode),
        "`b` (cur type {}) must be an instance of {}".format(type(b), CfgNode),
    )

    for k, v_ in a.items():
        full_key = ".".join(key_list + [k])

        v = copy.deepcopy(v_)
        v = b._decode_cfg_value(v)

        if k in ['help_info', "__help_info__"] and k in b:
            for name, info in v_.items():
                b[k][name] = info
        elif k in b:
            v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)
            # 递归合并字典
            if isinstance(v, CfgNode):
                try:
                    _merge_a_into_b(v, b[k], root, key_list + [k])
                except BaseException:
                    raise
            else:
                b[k] = v
        elif b.is_new_allowed():
            b[k] = v
        else:
            if root.key_is_deprecated(full_key):
                continue
            elif k in [
                    '__cfg_check_funcs__', '__help_info__', 'is_ready_for_run'
            ]:
                continue
            elif root.key_is_renamed(full_key):
                root.raise_key_rename_error(full_key)
            else:
                raise KeyError("Non-existent config key: {}".format(full_key))


def _check_and_coerce_cfg_value_type(replacement, original, key, full_key):
    """
    [从yacs修改而来，允许int <-> float转换]
    
    检查replacement（用于替换original）是否为正确类型。
    类型正确是指它完全匹配或是在几种可以轻松强制转换的类型情况之一。
    
    Args:
        replacement: 新值
        original: 原始值
        key (str): 键名
        full_key (str): 完整键路径
        
    Returns:
        转换后的值
        
    Raises:
        ValueError: 当类型不匹配且无法转换时抛出异常
    """
    original_type = type(original)
    replacement_type = type(replacement)
    if original_type is Argument:
        original_type = original.type
        original = original.value
    if replacement_type is Argument:
        replacement_type = replacement.type
        replacement = replacement.value

    # 类型必须匹配（有一些例外）
    if replacement_type == original_type:
        return replacement

    # 如果其中一个是None，允许类型转换为有效类型之一
    if (replacement_type is None and original_type in _VALID_TYPES) or (
            original_type is None and replacement_type in _VALID_TYPES):
        return replacement

    # 如果replacement和original类型分别匹配from_type和to_type，则将replacement从from_type转换为to_type
    def conditional_cast(from_type, to_type):
        if replacement_type == from_type and original_type == to_type:
            return True, to_type(replacement)
        else:
            return False, None

    # int <-> list, 强制替换
    # 允许单个 <-> 多个情况
    # 用例: 默认情况下，我们有cfg.attack.attacker_id = -1来表示在模拟的FL过程中不存在攻击者。
    # 我们可以轻松地将其更改为cfg.attack.attacker_id = 1来进行一个攻击者客户端#1的实验。
    # 但是，当我们想要通过设置cfg.attack.attacker_id = [1,2,3]来实现多个攻击者时，
    # yacs会引发错误，因为默认值(-1, int类型)无法替换为[1,2,3] (list类型)。
    # 这个错误促使我添加这样的强制替换规则来支持单个和多个攻击者。
    # TODO: 更好的解决方案？
    if replacement_type == int and original_type == list:
        return replacement
    if replacement_type == list and original_type == int:
        return replacement

    # 条件类型转换
    # list <-> tuple
    # 对于py2: 允许从str (bytes)转换为unicode字符串
    try:
        casts.append((str, unicode))  # noqa: F821
    except Exception:
        pass

    for (from_type, to_type) in casts:
        converted, converted_value = conditional_cast(from_type, to_type)
        if converted:
            return converted_value

    raise ValueError(
        "Type mismatch ({} vs. {}) with values ({} vs. {}) for config "
        "key: {}".format(original_type, replacement_type, original,
                         replacement, full_key))


def _assert_with_logging(cond, msg):
    """
    带日志记录的断言检查
    
    Args:
        cond (bool): 条件表达式
        msg (str): 断言失败时的消息
    """
    if not cond:
        logger.debug(msg)
    assert cond, msg


def _load_module_from_file(name, filename):
    """
    从文件加载Python模块
    
    Args:
        name (str): 模块名称
        filename (str): 文件路径
        
    Returns:
        module: 加载的模块对象
    """
    if _PY2:
        module = imp.load_source(name, filename)
    else:
        spec = importlib.util.spec_from_file_location(name, filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module
