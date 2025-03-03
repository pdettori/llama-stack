# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import fire
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client.types.tool_def_param import ToolDefParam
from termcolor import colored


def run_main(host: str, port: int, disable_safety: bool = False):

    client = LlamaStackClient(
        base_url=f"http://{host}:{port}",
    )

    agent_config = AgentConfig(
        model="llama3.2:3b-instruct-fp16",
        instructions="You are a helpful assistant tasked with writing performing arithmetic on a set of inputs.",
        client_tools=[
            ToolDefParam(
                name="AgentMetadata",
                metadata={
                    "name": "math_agent",
                    "description": "does math",
                    "framework": "langgraph",
                    "impl_class": "llama_stack.providers.inline.agents.multi_framework.MathAgent.getGraph",
                },
            )
        ],
        toolgroups=[],
    )

    agent = Agent(client, agent_config)
    session_id = agent.create_session("test-session")
    print(f"Created session_id={session_id} for Agent({agent.agent_id})")

    user_prompts = [
        ("What is the sum of 10 and 20?"),
        ("What is the product of 10 and 20?"),
    ]

    for prompt in user_prompts:
        response = agent.create_turn(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            session_id=session_id,
        )

        for log in EventLogger().log(response):
            log.print()


def main(host: str, port: int):
    run_main(host, port)


if __name__ == "__main__":
    fire.Fire(main)
