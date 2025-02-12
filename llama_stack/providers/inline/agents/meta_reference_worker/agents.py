from bullmq import Queue
import asyncio
from llama_stack.providers.inline.agents.meta_reference import MetaReferenceAgentsImpl
from llama_stack.providers.inline.agents.meta_reference import MetaReferenceAgentsImplConfig
from llama_stack.apis.agents import (
    AgentToolGroup,
    AgentTurnCreateRequest,
    AgentTurnResponseStreamChunk,
    Document,
)
from typing import AsyncGenerator, List, Optional, Union
from llama_stack.apis.inference import (
    ToolConfig,
    ToolResponseMessage,
    UserMessage,
)
import asyncio
import logging
import redis.asyncio as redis
from llama_stack.apis.safety import Safety
from llama_stack.apis.tools import ToolGroups, ToolRuntime
from llama_stack.apis.vector_io import VectorIO
from llama_stack.apis.inference import Inference

log = logging.getLogger(__name__)

LS_JOBS_QUEUE = "ls_jobs" # this might come from env
DEBUG_RUN_ID = "run_123456" # used for debug/test

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
        super().__init__(config, inference_api, vector_io_api, safety_api, tool_runtime_api, tool_groups_api)
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
        print(f"MetaReferenceAgentsWorkerImpl.create_agent_turn: {agent_id} and session {session_id}")
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
        import asyncio
        try:
            agent = await super().get_agent(request.agent_id)
            async for event in agent.create_and_execute_turn(request):
                await self.publisher.publish(DEBUG_RUN_ID, event)
                yield event
        except asyncio.CancelledError as e:
            print(e)
            # Handle cleanup or logging before exiting.
            raise
        except Exception as e:
            log.exception("error publishing event: %s", e)
            raise e       

class RedisPublisher:
    def __init__(self):
        self.redis = None

    async def connect(self):
        self.redis = await redis.from_url("redis://localhost")
    
    async def publish(self, channel, event: AgentTurnResponseStreamChunk):
        if not self.redis:
            raise Exception("Not connected to Redis. Call connect() first.")
        await self.redis.publish(channel, AgentTurnResponseStreamChunk.model_dump_json(event))

    async def disconnect(self):
        if self.pubsub:
            await self.pubsub.unsubscribe(self.channel_name)
            await self.pubsub.close()
        if self.redis:
            await self.redis.close()
