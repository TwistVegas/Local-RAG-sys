# Local-RAG-sys
西农大模型结课作业

## 项目结构
```plaintext
Local-RAG/
├── chroma_db/            # 运行时数据
├── configs/
│   └── settings.yaml     # 配置文件
├── src/                  
│   ├── config_manager.py # 配置注入器
│   ├── loader.py         # PDF解析与降噪
│   ├── vector_store.py   # ChromaDB持久化
│   ├── reranker.py       # 精排模块
│   └── log_utils.py      # 日志系统与性能计时器
├── logs/                 # 运行时日志
├── app.py                # 平台前端
└── requirements.txt      # 环境依赖目录
```

## 快速开始
1. 安装`Python 3.9`
2. 确保已安装Ollama，并获取模型
   ```
   ollama pull qwen3-vl:30b
   ```
3. 依赖安装
   ```
   pip install -r requirements.txt
   ```
4. 配置文件修改
   
   根据实际情况，编辑`configs/settings.yaml`
5. 启动平台
   ```
   streamlit run app.py
   ```
