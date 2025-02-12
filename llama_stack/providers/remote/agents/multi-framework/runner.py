# Copyright 2024 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

from bullmq import Job
from opentelemetry import trace

from workers import create_worker
from database import create_postgres_client
from workers import redis_client
import json
from config import config
import sys
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client.types.agents.turn_create_params import Document
import time

tracer = trace.get_tracer("job-trace")
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv()) # read local .env file

logger = logging.getLogger()
DEBUG_RUN_ID = "run_123456"

class JobHandler:
    def __init__(self, ls_client):
        self.ls_client = ls_client
        
    async def handleRun(self, job: Job, *args, **kwargs):
        with tracer.start_as_current_span("job") as span:
            data = job.data
            print(data)
            run_id = data.get('run_id')
            if run_id is None:
                raise RuntimeError("run_id not found")
            
            # need to retrieve the run info from the DB to start the create turn
            
            await self.create_turn()
            

    async def create_turn(self, disable_safety: bool = False):
        client = self.ls_client
        urls = [
            "README.md",
        ]

        attachments = [
            Document(
                content=f"https://raw.githubusercontent.com/kubestellar/kubeflex/refs/heads/main/{url}",
                mime_type="text/plain",
            )
            for i, url in enumerate(urls)
        ]

        available_shields = [shield.identifier for shield in client.shields.list()]
        print(available_shields)
        if not available_shields:
            print("No available shields. Disabling safety.", "yellow")
        else:
            print(f"Available shields found: {available_shields}")
        available_models = [
            model.identifier for model in client.models.list() if model.model_type == "llm"
        ]
        if not available_models:
            print("No available models. Exiting.", "red")
            return

        selected_model = available_models[0]
        print(f"Using model: {selected_model}")

        # this should be retrieved from the DB
        agent_config = AgentConfig(
            model=selected_model,
            instructions="You are a helpful assistant",
            sampling_params={
                "strategy": {"type": "top_p", "temperature": 1.0, "top_p": 0.9},
            },
            toolgroups=["builtin::rag"],
            tool_choice="auto",
            tool_prompt_format="json",
            input_shields=available_shields if available_shields else [],
            output_shields=available_shields if available_shields else [],
            enable_session_persistence=False,
        )

        agent = Agent(client, agent_config)
        session_id = agent.create_session("test-session")
        print(f"Created session_id={session_id} for Agent({agent.agent_id})")

        user_prompts = [
            (
                "What is KubeFlex? Give a short summary.",
                attachments,
            ),
        ]

        for prompt in user_prompts:
            response = agent.create_turn(
                messages=[
                    {
                        "role": "user",
                        "content": prompt[0],
                    }
                ],
                documents=prompt[1],
                session_id=session_id,
            )

            turnId = "db989fce-4dcf-4128-a960-8908028670f4"
            channel = DEBUG_RUN_ID
    

            for log in EventLogger().log(response):
                logger.info(log.content)
                #step_progress = {"event":{"payload":{"event_type":"step_progress","step_type":"inference","step_id":"step_id","delta":{"type":"text","text": log.content}}}}
                #await redis_client.publish(f'{channel}', json.dumps(step_progress))
