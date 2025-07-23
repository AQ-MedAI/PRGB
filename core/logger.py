import logging
import sys
from typing import Optional

# 全局日志级别控制
LOG_LEVEL = logging.INFO  # 可以改为 logging.DEBUG 来显示详细日志

# 全局日志格式
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 全局logger实例
_global_logger = None

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    获取全局配置的logger实例
    
    Args:
        name: logger名称，如果为None则使用调用模块的名称
        
    Returns:
        配置好的logger实例
    """
    global _global_logger
    
    if _global_logger is None:
        # 初始化全局logger配置
        _setup_global_logger()
    
    if name is None:
        # 获取调用模块的名称
        import inspect
        frame = inspect.currentframe().f_back
        if frame:
            module_name = frame.f_globals.get('__name__', 'unknown')
        else:
            module_name = 'unknown'
        return logging.getLogger(module_name)
    else:
        return logging.getLogger(name)

def _setup_global_logger():
    """设置全局logger配置"""
    global _global_logger
    
    # 配置根logger
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)
    
    # 清除现有的handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 添加控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 设置第三方库的日志级别
    _setup_third_party_loggers()
    
    _global_logger = root_logger

def _setup_third_party_loggers():
    """设置第三方库的日志级别"""
    # 设置httpx的日志级别为WARNING，避免输出HTTP请求日志
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.WARNING)
    
    # 设置requests的日志级别为WARNING
    requests_logger = logging.getLogger("requests")
    requests_logger.setLevel(logging.WARNING)
    
    # 设置urllib3的日志级别为WARNING
    urllib3_logger = logging.getLogger("urllib3")
    urllib3_logger.setLevel(logging.WARNING)
    
    # 设置openai的日志级别为WARNING
    openai_logger = logging.getLogger("openai")
    openai_logger.setLevel(logging.WARNING)

def set_log_level(level: int):
    """
    设置全局日志级别
    
    Args:
        level: 日志级别 (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR)
    """
    global LOG_LEVEL
    LOG_LEVEL = level
    
    # 重新设置全局logger
    _global_logger = None
    _setup_global_logger()

def set_verbose(verbose: bool = True):
    """
    设置详细日志模式
    
    Args:
        verbose: 是否启用详细日志
    """
    if verbose:
        set_log_level(logging.DEBUG)
    else:
        set_log_level(logging.INFO)
