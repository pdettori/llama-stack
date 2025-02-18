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
    AgentTurnResponseEventType,
    AgentTurnResponseEvent,
    AgentTurnResponseTurnCompletePayload,
    Turn
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

TURNS_JOB_QUEUE = "turns"  # this might come from env or config


def gen_turn_job_id_list_key(agent_id, session_id) -> str:
    """Generates a key used to persist a list of turn_job_ids associated with a specific session"""
    return f"turn_job_list:{agent_id}:{session_id}"


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
        super().__init__(
            config,
            inference_api,
            vector_io_api,
            safety_api,
            tool_runtime_api,
            tool_groups_api,
        )
        self.jobs_queue = Queue(TURNS_JOB_QUEUE)

    async def initialize(self):
        await super().initialize()

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
            f"MetaReferenceAgentsDispatcherImpl.create_agent_turn: {agent_id} and session {session_id}"
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

        turn_job_id = self.gen_turn_job_id(agent_id)
        await self.persist_turn_job_id_by_session_and_agent_id(
            agent_id, session_id, turn_job_id
        )
        await self.persist_turn_job_data(turn_job_id, request)

        await self.add_job_to_queue(turn_job_id)

        # wait for events from redis pub-sub
        subscriber = RedisSubscriber(channel_name=turn_job_id)
        await subscriber.connect()
        log.info(f"waiting for events on channel {turn_job_id}")
        return subscriber.wait_for_events()

    async def persist_turn_job_data(self, turn_job_id, request) -> str:
        turn_job_data = request.model_dump_json()
        await self.persistence_store.set(turn_job_id, turn_job_data)

    def gen_turn_job_id(self, agent_id) -> str:
        return f"turn_job:{agent_id}:{str(uuid.uuid4())}"

    async def add_job_to_queue(self, turn_job_id):
        job = await self.jobs_queue.add(turn_job_id, {"turn_job_id": turn_job_id})
        log.info(f"Job for turn_job_id {turn_job_id} added with ID: {job.id}")

    async def persist_turn_job_id_by_session_and_agent_id(
        self, agent_id, session_id, turn_job_id
    ):
        """Stores the list of turn_job_id associated with a session"""
        key = gen_turn_job_id_list_key(agent_id, session_id)

        list_raw = await self.persistence_store.get(key)
        try:
            if list_raw:
                list_json = json.loads(list_raw)
                turn_jobs_list = TurnJobsList.model_validate_json(list_json)
            else:
                turn_jobs_list = TurnJobsList()
        except ValidationError as e:
            log.error(f"Failed to validate JSON due to: {e}")
            turn_jobs_list = TurnJobsList()

        item = TurnJobItem.create(turn_job_id=turn_job_id)
        turn_jobs_list.append_item(item)

        updated_json = turn_jobs_list.model_dump_json()
        await self.persistence_store.set(key, updated_json)


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
                message = await self.pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1
                )
                if message and message["type"] == "message":
                    try:
                        json_event = message["data"].decode("utf-8")
                    except UnicodeDecodeError:
                        log.error("Failed to decode message data")

                    chunk = AgentTurnResponseStreamChunk.model_validate_json(json_event)
                   
                    # closing the SSE connection after the last event requires return without data
                    # this will cause the server to send a content length of 0 and no data after that
                    # normally there will be a content lenght line followed by a content line
                    # e.g., 
                    # ac <-content length in hex
                    # data: {"event":{"payload":{"event_type":"step_progress","step_type":"inference",
                    # "step_id":"b5d7edca-44a3-4dc7-9426-1b82e05c2b6c",
                    # "delta":{"type":"text","text":" more"}}}} <- data
                    if chunk is not None and chunk.event.payload.event_type == EventType.turn_complete.value:
                        yield chunk
                        return
                    else:
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


class TurnJobItem(BaseModel):
    """TurnJobItem represents info required to associate a list of turns with an agent and session ids"""

    turn_job_id: str
    creation_time: datetime

    def __post_init__(self):
        log.info(f"Creating an TurnJobItem object with id: {self.turn_job_id}")

    @classmethod
    def create(cls, turn_job_id: str, creation_time: datetime = None):
        """Convenience method to initialize with default time."""
        if not creation_time:
            creation_time = datetime.now()
        return cls(turn_job_id=turn_job_id, creation_time=creation_time)


class TurnJobsList(BaseModel):
    items: List[TurnJobItem] = []

    def append_item(self, item: TurnJobItem) -> None:
        """Append an item to the list."""
        self.items.append(item)

    def get_last_item(self) -> Optional[TurnJobItem]:
        """Retrieve the last item appended to the list."""
        if self.items:
            return self.items[-1]
        return None
