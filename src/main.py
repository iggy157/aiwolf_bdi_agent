"""設定に応じたエージェントを起動するスクリプト."""

import logging
import multiprocessing
from pathlib import Path

import yaml

import starter

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
console_handler.setFormatter(formatter)

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    config_path = "/home/bi23056/aiwolf-nlp-agent-llm/src/config/config.yml"
    with Path.open(Path(config_path)) as f:
        config = yaml.safe_load(f)
        logger.info("設定ファイルを読み込みました")

    agent_num = int(config["agent"]["num"])
    logger.info("エージェント数を %d に設定しました", agent_num)
    if agent_num == 1:
        starter.connect(config)
    else:
        threads = []
        for i in range(agent_num):
            thread = multiprocessing.Process(
                target=starter.connect,
                args=(config, i + 1),
            )
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
