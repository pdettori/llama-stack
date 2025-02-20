# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from bullmq import Queue
import asyncio
from llama_stack.providers.inline.agents.meta_reference import MetaReferenceAgentsImpl
from llama_stack.providers.inline.agents.meta_reference import (
    MetaReferenceAgentsImplConfig,
)
from llama_stack.apis.agents import (
    AgentToolGroup,
    AgentTurnCreateRequest,
    AgentTurnResponseStreamChunk,
    Document,
    AgentTurnResponseEventType
)
from typing import AsyncGenerator, List, Optional, Union
from llama_stack.apis.inference import (
    ToolConfig,
    ToolResponseMessage,
    UserMessage,
)
import asyncio
import logging
import json
import redis.asyncio as redis
from llama_stack.apis.safety import Safety
from llama_stack.apis.tools import ToolGroups, ToolRuntime
from llama_stack.apis.vector_io import VectorIO
from llama_stack.apis.inference import Inference
from llama_stack.providers.inline.agents.meta_reference_dispatcher import (
    TurnJobsList,
    gen_turn_job_id_list_key,
)

log = logging.getLogger(__name__)

EventType = AgentTurnResponseEventType


# Dispatches jobs for agent turns using a Queue-Worker Pattern
class MetaReferenceAgentsWorkerImpl(MetaReferenceAgentsImpl):
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
        self.publisher = RedisPublisher()

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
            f"MetaReferenceAgentsWorkerImpl.create_agent_turn: {agent_id} and session {session_id}"
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
        if stream:
            return self._create_agent_turn_streaming(request)
        else:
            raise NotImplementedError("Non-streaming agent turns not yet implemented")

    async def _create_agent_turn_streaming(
        self,
        request: AgentTurnCreateRequest,
    ) -> AsyncGenerator:
        await self.publisher.connect()
        turn_job_id = await self.retrieve_turn_job_id(
            request.agent_id, request.session_id
        )
        # on the consumer side, turn_job_id is used as the channel to receive events
        publisher_channel = turn_job_id

        try:
            agent = await super().get_agent(request.agent_id)
            async for event in agent.create_and_execute_turn(request):
                await self.publisher.publish(publisher_channel, event)
                yield event    
        except asyncio.CancelledError as e:
            log.info(e)
        except ValueError as e:
            log.exception("error publishing event: %s - make sure you enable session persistence in your agent config", e)    
        except Exception as e:
            log.exception("unexpected error publishing event: %s", e)

    async def retrieve_turn_job_id(self, agent_id, session_id) -> str:
        """retrieves the turn_job_id associated with this turn from the persistent store"""
        key = gen_turn_job_id_list_key(agent_id, session_id)
        list_raw = await self.persistence_store.get(key)

        if list_raw:
            list_json = json.loads(list_raw)
        else:
            log.error(f"No data found for key {key}.")
            return None

        try:
            list = TurnJobsList(**list_json)
        except ValueError as e:
            log.error(f"Failed to validate JSON for key {key}: {e}")
            return None

        item = list.get_last_item()

        if not item:
            log.error(f"No items found in the list for key {key}.")
            return None

        return item.turn_job_id


class RedisPublisher:
    def __init__(self):
        self.redis = None

    async def connect(self):
        self.redis = await redis.from_url("redis://localhost")

    async def publish(self, channel, event: AgentTurnResponseStreamChunk):
        if not self.redis:
            raise Exception("Not connected to Redis. Call connect() first.")
            
        await self.redis.publish(
            channel, AgentTurnResponseStreamChunk.model_dump_json(event)
        )


    async def disconnect(self):
        if self.pubsub:
            await self.pubsub.unsubscribe(self.channel_name)
            await self.pubsub.close()
        if self.redis:
            await self.redis.close()
