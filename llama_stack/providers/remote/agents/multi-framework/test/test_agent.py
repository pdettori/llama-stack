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
from llama_stack_client.types.agents.turn_create_params import Document
from termcolor import colored

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv()) # read local .env file

HOST = "localhost"
PORT = 8321


def run_main(disable_safety: bool = False):
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

    client = LlamaStackClient(
        base_url=f"http://{HOST}:{PORT}",
    )
   
    available_shields = [shield.identifier for shield in client.shields.list()]
    if not available_shields:
        print(colored("No available shields. Disabling safety.", "yellow"))
    else:
        print(f"Available shields found: {available_shields}")
    available_models = [
        model.identifier for model in client.models.list() if model.model_type == "llm"
    ]
    if not available_models:
        print(colored("No available models. Exiting.", "red"))
        return

    selected_model = available_models[0]
    print(f"Using model: {selected_model}")

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
        enable_session_persistence=True,
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

        for log in EventLogger().log(response):
            log.print()


def main():
    run_main()


if __name__ == "__main__":
    fire.Fire(main)

