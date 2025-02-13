from bullmq import Queue
import asyncio
from llama_stack.providers.inline.agents.meta_reference import MetaReferenceAgentsImpl
from llama_stack.providers.inline.agents.meta_reference import MetaReferenceAgentsImplConfig
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
import redis.asyncio as redis
from llama_stack.apis.safety import Safety
from llama_stack.apis.tools import ToolGroups, ToolRuntime
from llama_stack.apis.vector_io import VectorIO
from llama_stack.apis.inference import Inference

EventType = AgentTurnResponseEventType

log = logging.getLogger(__name__)

LS_JOBS_QUEUE = "ls_jobs" # this might come from env
DEBUG_RUN_ID = "run_123456" # used for debug/test

# Dispatches jobs for agent turns using a Queue-Worker Pattern
class MetaReferenceAgentsDispatcherImpl(MetaReferenceAgentsImpl):
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
        print(f"MetaReferenceAgentsDispatcherImpl.create_agent_turn: {agent_id} and session {session_id}")
        request = AgentTurnCreateRequest(
            agent_id=agent_id,
            session_id=session_id,
            messages=messages,
            stream=True,
            toolgroups=toolgroups,
            documents=documents,
            tool_config=tool_config,
        )
        # TODO:
        # 1. generate run_id
        # 2. store agent_id, session_id, messages, toolgroups, documents, stream, tool_config with key run_id
        import json

        # TODO - temp hack to pass session_id along
        def write_dict_to_file(file_path, dictionary):
            """Writes a flat dictionary to a JSON file."""
            try:
                with open(file_path, 'w') as file:
                    json.dump(dictionary, file)
                print(f"Dictionary successfully written to {file_path}.")
            except Exception as e:
                print(f"An error occurred while writing to the file: {e}")

        dict = {"session_id":session_id, "agent_id": agent_id}
        write_dict_to_file("/tmp/turn_info.json", dict)      

        await self.add_job_to_queue(DEBUG_RUN_ID)

        # wait for events from redis pub-sub
        subscriber = RedisSubscriber(channel_name=DEBUG_RUN_ID)
        await subscriber.connect()
        return subscriber.wait_for_events()

    async def add_job_to_queue(self, run_id):
        # Add a job to the queue and wait for it asynchronously
        job = await self.jobs_queue.add(run_id, {'run_id': run_id})
        print(f"Job for run_id {run_id} added with ID: {job.id}")

class RedisSubscriber:
    def __init__(self, channel_name):
        self.channel_name = channel_name
        self.redis = None
        self.pubsub = None
        self.end_turn = False

    async def connect(self):
        self.redis = await redis.from_url("redis://localhost")
        self.pubsub = self.redis.pubsub()
        await self.pubsub.subscribe(self.channel_name)
        log.info(f"Subscribed to {self.channel_name}")
          
    async def wait_for_events(self) -> AsyncGenerator:
        if not self.pubsub:
            raise RuntimeError("You must call connect() before iterating")

        while True:
            try:
                message = await self.pubsub.get_message(ignore_subscribe_messages=True, timeout=1)
                if message and message['type'] == 'message':
                    try:
                        json_event = message['data'].decode('utf-8')
                    except UnicodeDecodeError:
                        log.error("Failed to decode message data")
                    
                    chunk = AgentTurnResponseStreamChunk.model_validate_json(json_event)

                    yield chunk   
            except asyncio.CancelledError as e:
                raise e  
            except Exception as e:
                log.exception("Unexpected error: %s", e)
                raise e        
            
    async def disconnect(self):
        if self.pubsub:
            await self.pubsub.unsubscribe(self.channel_name)
            await self.pubsub.close()
        if self.redis:
            await self.redis.close()
