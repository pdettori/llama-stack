#! /usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from llama_stack.apis.agents import AgentConfig
from typing import AsyncGenerator, List
from llama_stack.apis.inference import (
    ToolResponseMessage,
    UserMessage,
)

class MultiFrameworkAgent:
    """
    Abstract base class for running agents.
    """
    def __init__(self, agent_config: AgentConfig) -> None:
        """
        Initializes the MultiFrameworkAgent with the given agent AgentConfig.
        """

        agent_metadata = self.extract_agent_metadata(agent_config)

        self.name = agent_metadata["name"]
        self.description = agent_metadata["description"]
        self.framework = agent_metadata["framework"]
        self.impl_class = agent_metadata["impl_class"]
        self.instructions = agent_config.instructions
        self.model = agent_config.model
        
    @staticmethod
    def extract_agent_metadata(agent_config):
        for tool in agent_config.client_tools:
            if tool.name == "AgentMetadata":
                return tool.metadata
        return None

    @abstractmethod
    def run(self, messages: List[UserMessage | ToolResponseMessage]) -> str:
        """
        Runs the agent with the input messages.
        """

    @abstractmethod
    async def run_streaming(self, messages: List[UserMessage | ToolResponseMessage]) -> AsyncGenerator:
        """
        Runs the agent in streaming mode with the given messages.
        """
