# Local-RAG-sys
西农大模型结课作业

## 项目结构
```plaintext
Local-RAG/
├── configs/
│   └── settings.yaml     # 配置文件
├── src/                  # 核心逻辑模块
│   ├── config_manager.py # 配置注入器
│   ├── loader.py         # PDF解析与降噪
│   ├── vector_store.py   # ChromaDB持久化
│   ├── reranker.py       # 精排模块
│   └── log_utils.py      # 日志系统与性能计时器
├── logs/                 # 运行时日志（用于实验分析）
├── app.py                # 平台前端
└── requirements.txt      # 环境依赖目录
```

## 快速开始
1. 确保已安装Ollama，并获取模型
   ```
   ollama pull qwen3-vl:30b
   ```
2. 依赖安装
   ```
   pip install -r requirements.txt
   ```
3. 配置文件修改
   
   根据实际情况，编辑`configs/settings.yaml`
5. 启动平台
   ```
   streamlit run app.py
   ```
