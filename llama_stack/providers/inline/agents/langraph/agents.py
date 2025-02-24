# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import asyncio
from llama_stack.providers.inline.agents.meta_reference import MetaReferenceAgentsImpl
from llama_stack.providers.inline.agents.meta_reference import (
    MetaReferenceAgentsImplConfig,
)
from llama_stack.distribution.request_headers import NeedsRequestProviderData
from llama_stack.apis.agents import (
    AgentToolGroup,
    AgentTurnCreateRequest,
    AgentTurnResponseStreamChunk,
    Document,
    AgentTurnResponseEventType,
    AgentTurnResponseEvent,
    AgentTurnResponseTurnCompletePayload,
    Turn,
    AgentConfig,
    AgentCreateResponse,
)
from typing import AsyncGenerator, List, Optional, Union
from llama_stack.apis.inference import (
    ToolConfig,
    ToolResponseMessage,
    UserMessage,
)
import asyncio
import uuid
import logging
import redis.asyncio as redis
from llama_stack.apis.safety import Safety
from llama_stack.apis.tools import ToolGroups, ToolRuntime
from llama_stack.apis.vector_io import VectorIO
from llama_stack.apis.inference import Inference
import hashlib
from pydantic import BaseModel, ValidationError
from datetime import datetime
from typing import List
import json

EventType = AgentTurnResponseEventType

log = logging.getLogger(__name__)


# Dispatches jobs for agent turns using a Queue-Worker Pattern
class LangGraphAgentImpl(MetaReferenceAgentsImpl, NeedsRequestProviderData):
    def __init__(
        self,
        config: MetaReferenceAgentsImplConfig,
        inference_api: Inference,
        vector_io_api: VectorIO,
        safety_api: Safety,
        tool_runtime_api: ToolRuntime,
        tool_groups_api: ToolGroups,
    ):
        super().__init__(
            config,
            inference_api,
            vector_io_api,
            safety_api,
            tool_runtime_api,
            tool_groups_api,
        )
       
       
    async def initialize(self):
        await super().initialize()

    async def create_agent(
        self,
        agent_config: AgentConfig,
    ) -> AgentCreateResponse:
        # example of getting provider_data
        # provider_data = self.get_request_provider_data()
        # the Web-Queue-Worker pattern requires the enable_session_persistence
        # always True
        agent_config.enable_session_persistence = True
        return await super().create_agent(agent_config)

    async def create_agent_turn(
        self,
        agent_id: str,
        session_id: str,
        messages: List[
            Union[
                UserMessage,
                ToolResponseMessage,
            ]
        ],
        toolgroups: Optional[List[AgentToolGroup]] = None,
        documents: Optional[List[Document]] = None,
        stream: Optional[bool] = False,
        tool_config: Optional[ToolConfig] = None,
    ) -> AsyncGenerator:
        log.info(
            f"LangGraphAgentImpl.create_agent_turn: {agent_id} and session {session_id}"
        )
        request = AgentTurnCreateRequest(
            agent_id=agent_id,
            session_id=session_id,
            messages=messages,
            stream=True,
            toolgroups=toolgroups,
            documents=documents,
            tool_config=tool_config,
        )

        

    