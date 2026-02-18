"""Multi-agent pool for running multiple sandboxed agents in parallel."""

import asyncio
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.bus.queue import MessageBus
from nanobot.config.schema import Config, AgentDefinition, AgentDefaults, TelegramConfig
from nanobot.session.manager import SessionManager
from nanobot.utils.helpers import ensure_dir


class ConfigError(Exception):
    """Configuration validation error."""
    pass


class AgentInstance:
    """A single sandboxed agent with isolated resources."""

    def __init__(self, agent_id: str, definition: AgentDefinition, config: Config):
        self.agent_id = agent_id
        self.definition = definition
        self.config = config

        # Contextual logger for this agent
        self.logger = logger.bind(agent=agent_id)

        # Resolve workspace
        self.workspace = self._resolve_workspace()

        # Isolated resources (initialized in async context)
        self.bus: MessageBus | None = None
        self.provider: Any = None  # LLMProvider
        self.agent_loop: Any = None  # AgentLoop
        self.channels: Any = None  # ChannelManager
        self.session_manager: SessionManager | None = None

        self._running = False

    def _resolve_workspace(self) -> Path:
        """Resolve and validate workspace path."""
        ws = self.definition.workspace or self.config.agents.defaults.workspace
        path = Path(ws).expanduser().resolve()
        return path

    def _resolve_model_config(self) -> dict[str, Any]:
        """Resolve model config with defaults + overrides."""
        defaults = self.config.agents.defaults
        return {
            "model": self.definition.model or defaults.model,
            "temperature": self.definition.temperature if self.definition.temperature is not None else defaults.temperature,
            "max_tokens": self.definition.max_tokens or defaults.max_tokens,
            "max_iterations": self.definition.max_tool_iterations or defaults.max_tool_iterations,
            "memory_window": self.definition.memory_window or defaults.memory_window,
        }

    def _create_provider(self):
        """Create LLM provider for this agent."""
        from nanobot.cli.commands import _make_provider

        # Create a modified config with the agent's model
        model_config = self._resolve_model_config()

        # Temporarily override the default model for provider creation
        original_model = self.config.agents.defaults.model
        self.config.agents.defaults.model = model_config["model"]

        try:
            provider = _make_provider(self.config)
        finally:
            self.config.agents.defaults.model = original_model

        return provider

    def _build_channel_config(self) -> Config:
        """Build a config with agent-specific channels."""
        # Create a shallow copy of the config
        # Override telegram if agent has its own
        if self.definition.telegram:
            # Use agent's telegram config
            self.config.channels.telegram = self.definition.telegram

        return self.config

    async def _ensure_workspace(self) -> None:
        """Ensure workspace directory exists with required structure."""
        ensure_dir(self.workspace)
        ensure_dir(self.workspace / "memory")
        ensure_dir(self.workspace / "sessions")
        ensure_dir(self.workspace / "skills")

        # Create default files if not exist
        memory_file = self.workspace / "memory" / "MEMORY.md"
        if not memory_file.exists():
            memory_file.write_text("# Long-term Memory\n\n")

        history_file = self.workspace / "memory" / "HISTORY.md"
        if not history_file.exists():
            history_file.write_text("")

        self.logger.debug(f"Workspace ready: {self.workspace}")

    async def start(self) -> None:
        """Initialize and start this agent."""
        self.logger.info(f"Starting agent '{self.agent_id}'...")

        try:
            # 1. Ensure workspace exists
            await self._ensure_workspace()

            # 2. Create isolated message bus
            self.bus = MessageBus()
            self.logger.debug("Message bus created")

            # 3. Create provider
            self.provider = self._create_provider()
            self.logger.debug("Provider created")

            # 4. Create session manager
            self.session_manager = SessionManager(self.workspace)
            self.logger.debug(f"Session manager initialized")

            # 5. Create agent loop
            from nanobot.agent.loop import AgentLoop

            model_config = self._resolve_model_config()
            self.agent_loop = AgentLoop(
                bus=self.bus,
                provider=self.provider,
                workspace=self.workspace,
                model=model_config["model"],
                temperature=model_config["temperature"],
                max_tokens=model_config["max_tokens"],
                max_iterations=model_config["max_iterations"],
                memory_window=model_config["memory_window"],
                brave_api_key=self.config.tools.web.search.api_key or None,
                exec_config=self.config.tools.exec,
                restrict_to_workspace=self.config.tools.restrict_to_workspace,
                session_manager=self.session_manager,
                mcp_servers=self.config.tools.mcp_servers,
            )
            self.logger.debug("Agent loop created")

            # 6. Create channels with agent-specific config
            from nanobot.channels.manager import ChannelManager

            channel_config = self._build_channel_config()
            self.channels = ChannelManager(channel_config, self.bus)
            self.logger.info(f"Channels: {self.channels.enabled_channels or 'none'}")

            # 7. Start everything
            self._running = True

            if self.channels.enabled_channels:
                await asyncio.gather(
                    self.agent_loop.run(),
                    self.channels.start_all(),
                )
            else:
                # No channels, just run agent loop (for CLI/testing)
                await self.agent_loop.run()

        except Exception as e:
            self.logger.error(f"Failed to start agent '{self.agent_id}': {e}")
            self._running = False
            raise

    async def stop(self, timeout: float = 60.0) -> None:
        """Stop this agent with timeout protection."""
        if not self._running:
            return

        self.logger.info(f"Stopping agent '{self.agent_id}'...")

        try:
            # Stop agent loop
            if self.agent_loop:
                self.agent_loop.stop()
                await self.agent_loop.close_mcp()

            # Stop channels
            if self.channels:
                await asyncio.wait_for(
                    self.channels.stop_all(),
                    timeout=timeout
                )

            self._running = False
            self.logger.info(f"Agent '{self.agent_id}' stopped")

        except asyncio.TimeoutError:
            self.logger.error(f"Agent '{self.agent_id}' shutdown timeout after {timeout}s")
            self._running = False
        except Exception as e:
            self.logger.error(f"Agent '{self.agent_id}' shutdown error: {e}")
            self._running = False

    @property
    def is_running(self) -> bool:
        return self._running


class AgentPool:
    """Manages multiple sandboxed agents running in parallel."""

    SHUTDOWN_TIMEOUT = 60.0  # 1 minute timeout

    def __init__(self, config: Config):
        self.config = config
        self.logger = logger.bind(component="agent_pool")
        self.agents: dict[str, AgentInstance] = {}
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate agent pool configuration for common issues."""
        pool = self.config.agents.pool
        if not pool:
            return

        # Check workspace uniqueness
        workspaces: dict[str, str] = {}
        for agent_id, agent_def in pool.items():
            if not agent_def.enabled:
                continue

            ws = agent_def.workspace or self.config.agents.defaults.workspace
            ws_path = str(Path(ws).expanduser().resolve())

            if ws_path in workspaces:
                raise ConfigError(
                    f"Duplicate workspace: agents '{workspaces[ws_path]}' and '{agent_id}' "
                    f"both use {ws_path}"
                )
            workspaces[ws_path] = agent_id

        self.logger.info(f"Validated {len(workspaces)} agent workspaces")

    async def start_all(self) -> None:
        """Start all enabled agents in parallel with timeout protection."""
        pool = self.config.agents.pool
        if not pool:
            self.logger.warning("No agents in pool")
            return

        enabled = {k: v for k, v in pool.items() if v.enabled}
        if not enabled:
            self.logger.warning("No enabled agents in pool")
            return

        self.logger.info(f"Starting {len(enabled)} agents: {list(enabled.keys())}")

        # Start all agents in parallel, catching individual failures
        async def start_with_timeout(agent_id: str, definition: AgentDefinition) -> None:
            try:
                agent = AgentInstance(agent_id, definition, self.config)
                self.agents[agent_id] = agent

                # 1 minute timeout per agent startup
                await asyncio.wait_for(
                    agent.start(),
                    timeout=60.0
                )
            except asyncio.TimeoutError:
                self.logger.error(f"Agent '{agent_id}' startup timed out after 60s")

                # Notify user via bus if available
                if agent.bus:
                    from nanobot.bus.events import OutboundMessage
                    await agent.bus.publish_outbound(OutboundMessage(
                        channel="system",
                        chat_id="admin",
                        content=f"⚠️ Agent '{agent_id}' startup timed out. Please check the logs."
                    ))
            except Exception as e:
                self.logger.error(f"Agent '{agent_id}' failed to start: {e}")

        tasks = [start_with_timeout(aid, defn) for aid, defn in enabled.items()]
        await asyncio.gather(*tasks)

        running = sum(1 for a in self.agents.values() if a.is_running)
        self.logger.info(f"Agent pool started: {running}/{len(enabled)} running")

    async def stop_all(self) -> None:
        """Stop all agents with timeout protection."""
        self.logger.info(f"Stopping {len(self.agents)} agents...")

        async def stop_one(agent_id: str, agent: AgentInstance) -> None:
            try:
                await asyncio.wait_for(
                    agent.stop(timeout=self.SHUTDOWN_TIMEOUT),
                    timeout=self.SHUTDOWN_TIMEOUT + 5
                )
            except asyncio.TimeoutError:
                self.logger.error(f"Agent '{agent_id}' shutdown timed out")

        await asyncio.gather(*[
            stop_one(aid, agent)
            for aid, agent in self.agents.items()
        ])

        self.logger.info("All agents stopped")

    def get_status(self) -> dict[str, Any]:
        """Get status of all agents for debugging."""
        return {
            "total": len(self.agents),
            "running": sum(1 for a in self.agents.values() if a.is_running),
            "agents": {
                agent_id: {
                    "running": agent.is_running,
                    "workspace": str(agent.workspace),
                    "channels": agent.channels.enabled_channels if agent.channels else [],
                }
                for agent_id, agent in self.agents.items()
            }
        }

    @property
    def running_count(self) -> int:
        """Count of running agents."""
        return sum(1 for a in self.agents.values() if a.is_running)
