import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Dict, List, Optional

import requests
from tqdm import tqdm

# 使用统一的日志管理器
from ..logger import get_logger

logger = get_logger()

# OpenAI import - only imported when this module is used
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI is not installed. Please install it with: pip install openai")

# 移除模块级别的API_KEY读取，改为在需要时读取
# API_KEY = os.getenv("API_KEY")

def get_api_key():
    """获取API密钥，支持运行时读取环境变量"""
    return os.getenv("API_KEY")

def transfer_dict_conv(
    inputs: List[str], system: Optional[str] = None
) -> List[Dict[str, str]]:
    assert len(inputs) % 2 == 1, "number of rounds must be odd"
    output_chat_dict = (
        [{"role": "system", "content": system}] if system else []
    )

    for i in range(len(inputs)):
        if i % 2 == 0:
            output_chat_dict.append({"role": "user", "content": inputs[i]})
        elif i % 2 == 1:
            output_chat_dict.append(
                {"role": "assistant", "content": inputs[i]}
            )
    return output_chat_dict

class APIInferenceBase:
    def __init__(
        self,
        url="https://api.openai.com/v1/chat/completions",  # 修改为正确的chat completions端点
        api_key=None,
        model="gpt-3.5-turbo",
        inference_mode=False,
        max_retries=3,
        retry_delay=1.0,
        retry_backoff=2.0,
    ):
        if api_key is None:
            api_key = get_api_key()
            if api_key is None:
                raise ValueError("API key is not set. You can set it by `export API_KEY=<your_api_key>` or pass it to the constructor.")
        self.url = url
        self.model = model
        self.api_key = api_key
        self.inference_mode = inference_mode
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_backoff = retry_backoff

    def _should_retry(self, exception, response=None):
        """
        判断是否应该重试
        """
        # 网络相关错误
        if isinstance(exception, (requests.exceptions.ConnectionError, 
                                requests.exceptions.Timeout,
                                requests.exceptions.RequestException)):
            return True
        
        # HTTP状态码错误
        if response is not None:
            # 5xx服务器错误
            if response.status_code != 200:
                return True
        
        return False

    def _retry_with_backoff(self, func, *args, **kwargs):
        """
        带指数退避的重试机制
        """
        last_exception = None
        delay = self.retry_delay
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt == 0:
                    # 第一次尝试，直接调用
                    return func(*args, **kwargs)
                else:
                    # 重试前等待
                    time.sleep(delay)
                    # 增加延迟时间（指数退避）
                    delay *= self.retry_backoff
                    # 添加一些随机性避免惊群效应
                    delay += random.uniform(0, 0.1 * delay)
                    
            except Exception as e:
                last_exception = e
                
                # 检查是否应该重试
                should_retry = self._should_retry(e)
                
                if not should_retry or attempt == self.max_retries:
                    # 不重试或已达到最大重试次数
                    raise last_exception
                
                logger.warning(f"API调用失败，第{attempt + 1}次重试: {str(e)}")
        
        # 如果所有重试都失败了
        raise last_exception

    def batch_generate(
        self, data, temperature=0.0, top_p=0.8, batch_size=10
    ):
        """
        Batch generate responses with QPS control and threading
        
        Args:
            data: List of input messages to process
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            batch_size: Batch size for processing
        """
        self.queue = Queue()
        self.stats = {
            'total': len(data),
            'success': 0,
            'fail': 0,
            'total_time': 0.0,
            'lock': threading.Lock()
        }
        self.session = requests.Session()
        # 使用字典来保持输入和输出的对应关系
        self.results = {}
        self.results_lock = threading.Lock()
        self.pbar = tqdm(total=len(data), desc="Processing API Requests")
        self.stop_event = threading.Event()
        
        # QPS控制相关 - 使用令牌桶算法
        self.qps = batch_size
        self.tokens = batch_size  # 初始令牌数
        self.last_refill = time.time()
        self.token_lock = threading.Lock()
        
        # 将查询请求放入队列，同时保存索引
        for i, item in enumerate(data):
            self.queue.put((i, item))
            
        # 运行批量处理
        return self.run_batch(temperature, top_p, batch_size)  

    def acquire_token(self):
        """
        使用令牌桶算法控制QPS
        """
        while True:
            with self.token_lock:
                now = time.time()
                # 补充令牌
                time_passed = now - self.last_refill
                new_tokens = time_passed * self.qps
                self.tokens = min(self.qps, self.tokens + new_tokens)
                self.last_refill = now
                
                if self.tokens >= 1:
                    self.tokens -= 1
                    return  # 获得令牌，可以执行请求
                else:
                    # 没有令牌，等待
                    wait_time = (1 - self.tokens) / self.qps
                    time.sleep(wait_time)

    def worker(self, temperature, top_p):
        """
        工作线程，从队列中获取请求并处理
        """
        thread_id = threading.current_thread().ident
        logger.debug(f"Worker线程 {thread_id} 启动")
        
        while not self.stop_event.is_set():
            try:
                item = self.queue.get(timeout=1)
                if item is None:
                    break  # 收到结束信号则退出
                
                index, messages = item
                logger.debug(f"线程 {thread_id} 开始处理请求 {index}")
                
                # 获取令牌以控制QPS
                self.acquire_token()
                
                start_time = time.time()
                success = False
                try:
                    result = self.generate(messages, temperature, top_p)
                    with self.results_lock:
                        self.results[index] = result
                    success = True
                    logger.debug(f"线程 {thread_id} 请求 {index} 成功，耗时: {time.time() - start_time:.2f}s")
                except Exception as e:
                    with self.results_lock:
                        self.results[index] = f"Error: {str(e)}"
                    logger.error(f"线程 {thread_id} 请求 {index} 失败: {messages} - {str(e)}")
                
                response_time = time.time() - start_time

                with self.stats['lock']:
                    if success:
                        self.stats['success'] += 1
                        self.stats['total_time'] += response_time
                    else:
                        self.stats['fail'] += 1
                    
                    self.pbar.update(1)  # 更新进度条

                self.queue.task_done()
            except Exception as e:
                if not self.queue.empty():
                    logger.error(f"Worker线程 {thread_id} 异常: {str(e)}")
                break
        
        logger.debug(f"Worker线程 {thread_id} 退出")

    def run_batch(self, temperature=0.0, top_p=0.8, qps=10):
        """
        启动批量请求运行，包含多线程逻辑和 QPS 控制
        """
        start_time = time.time()
        
        # 计算线程数量 - 修复计算逻辑
        data_size = self.stats['total']
        # 根据数据量和QPS计算合适的线程数
        max_workers = min(data_size, max(1, min(10, qps * 2)))  # 最多10个线程，最少1个
        
        logger.debug(f"启动批量处理: 数据量={data_size}, QPS={qps}, 线程数={max_workers}")

        # 创建线程池
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 启动工作线程
            workers = [executor.submit(self.worker, temperature, top_p) for _ in range(max_workers)]

            # 等待所有任务完成
            self.queue.join()

            # 发送结束信号
            for _ in range(max_workers):
                self.queue.put(None)

            self.stop_event.set()  # 通知所有工作线程退出

        self.pbar.close()  # 关闭进度条
        
        # 按原始顺序返回结果
        results = []
        for i in range(data_size):
            if i in self.results:
                results.append(self.results[i])
            else:
                results.append("Error: Result not found")
        
        return results

class APIModel(APIInferenceBase):
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature=0.7,
        top_p=1,
    ):
        def _make_request():
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            query = {
                "model": self.model,
                "temperature": temperature,
                "top_p": top_p,
                "messages": messages,
                "stream": False,
            }
            response = requests.post(self.url, headers=headers, json=query)
            
            # 检查HTTP状态码
            if response.status_code != 200:
                raise requests.exceptions.HTTPError(f"HTTP {response.status_code}: {response.text}")
            
            response_json = response.json()
            if "choices" not in response_json:
                logger.error(f"Unexpected response format: {messages}")
                logger.error(f"Response: {response_json}")
                raise ValueError("Invalid response format: 'choices' not found")
            
            return response_json["choices"][0]["message"]["content"]
        
        return self._retry_with_backoff(_make_request)


class OpenAIModel(APIInferenceBase):
    def __init__(
        self,
        url="https://api.openai.com/v1/completions",
        api_key=None,
        model="gpt-3.5-turbo",
        inference_mode=False,
        max_retries=3,
        retry_delay=1.0,
        retry_backoff=2.0,
    ):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI is not installed. Please install it with: pip install openai")
        super().__init__(url, api_key, model, inference_mode, max_retries, retry_delay, retry_backoff)
        self.client = openai.Client(
            api_key=self.api_key,
            base_url=self.url
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature=0.7,
        top_p=1,
    ):
        def _make_request():
            completion = self.client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                top_p=top_p,
                messages=messages,
                stream=False,
            )
            return completion.choices[0].message.content
        
        return self._retry_with_backoff(_make_request) 