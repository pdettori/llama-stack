# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging

from bullmq import Job
from opentelemetry import trace

# from database import create_postgres_client
import json
import sys
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client.types.agents.turn_create_params import Document
from llama_stack.apis.agents import AgentTurnCreateRequest
import time
import asyncio

tracer = trace.get_tracer("job-trace")
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file

logger = logging.getLogger()


class JobHandler:
    def __init__(self, ls_client, persistence_store, redis_store):
        self.ls_client = ls_client
        self.persistence_store = persistence_store
        self.redis_store = redis_store

    async def handleTurn(self, job: Job, *args, **kwargs):
        with tracer.start_as_current_span("job") as span:
            print(f'processing job {job.id}')
            turn_job_id = job.data.get("turn_job_id")
            if turn_job_id is None:
                raise RuntimeError("turn_job_id not found")

            # retrieve the turn info from the DB to start the create turn
            turn_data = await self.persistence_store.get(turn_job_id)
            request = AgentTurnCreateRequest.model_validate_json(turn_data)

            await self.create_turn(request)

    async def create_turn(
        self, request: AgentTurnCreateRequest, disable_safety: bool = False
    ):
        client = self.ls_client
   
        response = client.agents.turn.create(
            messages = request.messages,
            documents = request.documents,
            session_id = request.session_id,
            agent_id = request.agent_id,
            stream = request.stream,
        )

        # need to iterate and give time for events to be emitted by the MetaReferenceAgentsWorkerImpl
        for log in EventLogger().log(response):
            await asyncio.sleep(0.005)

        # this is required to allow time for deallocating mem.  
        time.sleep(1)    
