import yaml
import os


class ConfigManager:

    def __init__(self, config_path="configs/settings.yaml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"未找到配置文件: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            self.configs = yaml.safe_load(f)

    def get(self, key, default=None):
        return self.configs.get(key, default)

    def get_rag_config(self):
        return self.configs.get('rag', {})

    def get_model_config(self):
        return self.configs.get('models', {})