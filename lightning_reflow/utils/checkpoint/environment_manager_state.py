"""Environment variable manager state for checkpoint persistence."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EnvironmentManagerState:
    """Manager state for environment variable tracking."""

    def __init__(self):
        self.env_vars: Dict[str, str] = {}
        self.config_sources: List[str] = []

    @property
    def manager_name(self) -> str:
        return "environment_manager"

    def set_environment_variables(
        self, env_vars: Dict[str, str], config_sources: Optional[List[str]] = None
    ) -> None:
        """Update tracked env vars and apply them to os.environ."""
        self.env_vars.update(env_vars)
        self.config_sources = config_sources or []
        for key, value in env_vars.items():
            os.environ[key] = str(value)

    def capture_state(self) -> Dict[str, Any]:
        return {
            'env_vars': self.env_vars.copy(),
            'config_sources': self.config_sources.copy(),
        }

    def restore_state(self, state: Dict[str, Any]) -> bool:
        try:
            self.set_environment_variables(
                state.get('env_vars', {}),
                state.get('config_sources', []),
            )
            return True
        except Exception as e:
            logger.warning("Failed to restore environment state: %s", e)
            return False

    def validate_state(self, state: Dict[str, Any]) -> bool:
        return isinstance(state, dict) and 'env_vars' in state

    def get_captured_variables(self) -> Dict[str, str]:
        return self.env_vars.copy()
