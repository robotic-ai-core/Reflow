"""
Manager state utilities for checkpoint persistence.

This module provides a minimal implementation of the manager state system
for tracking component states during checkpoint operations.
"""

from typing import Dict, Any, List, Optional, Protocol
import logging

logger = logging.getLogger(__name__)


class ManagerState(Protocol):
    """Protocol for manager state objects."""
    
    @property
    def manager_name(self) -> str:
        """Unique name for this manager."""
        ...
    
    def capture_state(self) -> Dict[str, Any]:
        """Capture current state for persistence."""
        ...
    
    def restore_state(self, state: Dict[str, Any]) -> bool:
        """Restore state from persistence. Returns True if successful."""
        ...
    
    def validate_state(self, state: Dict[str, Any]) -> bool:
        """Validate that the state is compatible. Returns True if valid."""
        ...


class EnvironmentManagerState:
    """Manager state for environment variable tracking."""
    
    def __init__(self):
        self.env_vars: Dict[str, str] = {}
        self.config_sources: List[str] = []
    
    @property
    def manager_name(self) -> str:
        return "environment_manager"
    
    def set_environment_variables(self, env_vars: Dict[str, str], config_sources: List[str] = None) -> None:
        """Set environment variables and track their sources."""
        import os
        self.env_vars.update(env_vars)
        self.config_sources = config_sources or []
        
        # Actually set the environment variables
        for key, value in env_vars.items():
            os.environ[key] = str(value)
    
    def capture_state(self) -> Dict[str, Any]:
        """Capture environment state for persistence."""
        return {
            'env_vars': self.env_vars.copy(),
            'config_sources': self.config_sources.copy()
        }
    
    def restore_state(self, state: Dict[str, Any]) -> bool:
        """Restore environment state from persistence."""
        try:
            env_vars = state.get('env_vars', {})
            config_sources = state.get('config_sources', [])
            self.set_environment_variables(env_vars, config_sources)
            return True
        except Exception as e:
            logger.warning(f"Failed to restore environment state: {e}")
            return False
    
    def validate_state(self, state: Dict[str, Any]) -> bool:
        """Validate environment state."""
        return isinstance(state, dict) and 'env_vars' in state


class ManagerStateRegistry:
    """Registry for managing all manager states."""
    
    def __init__(self):
        self.managers: Dict[str, ManagerState] = {}
    
    def register(self, manager: ManagerState) -> None:
        """Register a manager for state persistence."""
        self.managers[manager.manager_name] = manager
        logger.debug(f"Registered manager: {manager.manager_name}")
    
    def unregister(self, manager_name: str) -> None:
        """Unregister a manager."""
        if manager_name in self.managers:
            del self.managers[manager_name]
            logger.debug(f"Unregistered manager: {manager_name}")
    
    def capture_all_states(self) -> Dict[str, Any]:
        """Capture all registered manager states."""
        states = {}
        for name, manager in self.managers.items():
            try:
                states[name] = manager.capture_state()
            except Exception as e:
                logger.warning(f"Failed to capture state for {name}: {e}")
        return states
    
    def restore_all_states(self, states: Dict[str, Any]) -> Dict[str, bool]:
        """Restore all manager states. Returns dict of success status per manager."""
        results = {}
        for name, state in states.items():
            if name in self.managers:
                try:
                    if self.managers[name].validate_state(state):
                        results[name] = self.managers[name].restore_state(state)
                    else:
                        logger.warning(f"Invalid state for manager {name}")
                        results[name] = False
                except Exception as e:
                    logger.warning(f"Failed to restore state for {name}: {e}")
                    results[name] = False
            else:
                logger.warning(f"No manager registered for {name}")
                results[name] = False
        return results


# Global registry instance
_global_registry: Optional[ManagerStateRegistry] = None


def get_global_registry() -> ManagerStateRegistry:
    """Get the global manager state registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ManagerStateRegistry()
    return _global_registry


def register_manager(manager: ManagerState) -> None:
    """Register a manager with the global registry."""
    registry = get_global_registry()
    registry.register(manager)


def unregister_manager(manager_name: str) -> None:
    """Unregister a manager from the global registry."""
    registry = get_global_registry()
    registry.unregister(manager_name)


def capture_all_manager_states() -> Dict[str, Any]:
    """Capture all registered manager states."""
    registry = get_global_registry()
    return registry.capture_all_states()


def restore_all_manager_states(states: Dict[str, Any]) -> Dict[str, bool]:
    """Restore all manager states."""
    registry = get_global_registry()
    return registry.restore_all_states(states)


# Re-export for compatibility
__all__ = [
    'ManagerState',
    'EnvironmentManagerState', 
    'ManagerStateRegistry',
    'get_global_registry',
    'register_manager',
    'unregister_manager',
    'capture_all_manager_states',
    'restore_all_manager_states'
] 