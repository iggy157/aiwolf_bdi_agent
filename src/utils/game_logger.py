"""ゲームのログを出力するクラスを定義するモジュール."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ulid import ULID

if TYPE_CHECKING:
    from aiwolf_nlp_common.packet import Request


class GameLogger:
    """ゲームのログを出力するクラス."""

    def __init__(
        self,
        config: dict,
        game_id: str,
    ) -> None:
        """ゲームのログを初期化する."""
        self.config = config
        self.logger = logging.getLogger("game")
        self.logger.setLevel(logging.INFO)
        
        if bool(self.config["game_logger"]["enable"]):
            ulid: ULID = ULID.from_str(game_id)
            tz = datetime.now(UTC).astimezone().tzinfo
            timestamp = datetime.fromtimestamp(ulid.timestamp, tz=tz).strftime("%Y%m%d%H%M%S%f")
            
            output_dir = Path(str(self.config["game_logger"]["output_dir"]))
            output_dir.mkdir(parents=True, exist_ok=True)
            
            filename = self.config["game_logger"]["filename"].format(
                timestamp=timestamp,
                teams=self.config["agent"]["team"]
            )
            
            handler = logging.FileHandler(
                output_dir / f"{filename}.log",
                mode="w",
                encoding="utf-8",
            )
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log(self, message: str) -> None:
        """ログを出力する."""
        self.logger.info(message) 