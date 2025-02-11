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
DEBUG_RUN_ID = "run_123456"

class MetaReferenceAgentsQueuedImpl(MetaReferenceAgentsImpl):
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
        self.jobs_queue = Queue(LS_JOBS_QUEUE)

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
        request = AgentTurnCreateRequest(
            agent_id=agent_id,
            session_id=session_id,
            messages=messages,
            stream=True,
            toolgroups=toolgroups,
            documents=documents,
            tool_config=tool_config,
        )
        log.info(f"Hello Agent {agent_id} and session {session_id}")

        # TODO:
        # 1. generate run_id
        # 2. store agent_id, session_id, messages, toolgroups, documents, stream, tool_config with key run_id

        await self.add_job_to_queue(DEBUG_RUN_ID)

        # wait for events from redis pub-sub
        subscriber = RedisSubscriber(channel_name=DEBUG_RUN_ID)
        await subscriber.connect()
        return subscriber

    async def add_job_to_queue(self, run_id):
        # Add a job to the queue and wait for it asynchronously
        job = await self.jobs_queue.add(run_id, {'run_id': run_id})
        
        log.info(f"Job for run_id {run_id} added with ID: {job.id}")


class RedisSubscriber:
    def __init__(self, channel_name):
        self.channel_name = channel_name
        self.redis = None
        self.pubsub = None

    async def connect(self):
        self.redis = await redis.from_url("redis://localhost")
        self.pubsub = self.redis.pubsub()
        await self.pubsub.subscribe(self.channel_name)
        log.info(f"Subscribed to {self.channel_name}")

    async def __anext__(self):
        if not self.pubsub:
            raise RuntimeError("You must call connect() before iterating")

        while True:
            try:
                message = await self.pubsub.get_message(ignore_subscribe_messages=True, timeout=1)
                if message and message['type'] == 'message':
                    event = message['data'].decode('utf-8')
                    return AgentTurnResponseStreamChunk.model_validate_json(event)
            except asyncio.CancelledError as e:
                log.info(">>>> Generator cancelled")
                raise e  # Re-raise the CancelledError to ensure proper cleanup
            except Exception as e:
                log.exception("Unexpected error: %s", e)
                raise e

    async def disconnect(self):
        if self.pubsub:
            await self.pubsub.unsubscribe(self.channel_name)
            await self.pubsub.close()
        if self.redis:
            await self.redis.close()

    async def aclose(self):
        await self.disconnect()        

    def __aiter__(self):
        return self    