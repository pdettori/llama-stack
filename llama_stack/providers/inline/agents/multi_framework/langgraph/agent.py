import importlib

from ..agent import MultiFrameworkAgent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage # type: ignore
from typing import AsyncGenerator, List, Optional, Union
from llama_stack.apis.agents import (
    AgentConfig,
    AgentTurnResponseTurnCompletePayload,
)
from llama_stack.apis.inference import (
    ToolResponseMessage,
    UserMessage,
)
from .converters import convert_messages, EventProcessor


class LangGraphAgent(MultiFrameworkAgent):
    """
    LangGraphAgent extends the MultiFrameworkAgent class to load and run a specific LangGraph agent.
    """

    def __init__(self, agent_config: AgentConfig) -> str:
        """
        Initializes the workflow for the specified agent.
        The executable code must be within $PYTHONPATH.
        Args:
            agent_name (dict): Agent Configuration
        Raises:
            Exception: If the agent cannot be loaded, an exception is raised with an error message.
        """

        super().__init__(agent_config)

        # class naming: <directory>.<filename>.<class>.<method> ie
        #   test.lg_test.MathAssistant.invoke

        try:
            partial_impl_class, method_name = self.impl_class.rsplit(".", 1)
            module_name, class_name = partial_impl_class.rsplit(".", 1)
            my_module = importlib.import_module(module_name)
            # Get the class object
            self.langgraph_agent_class = getattr(my_module, class_name)
            # Instantiate the class
            self.instance = self.langgraph_agent_class()
            self.method_name = method_name
        except Exception as e:
            print(f"Failed to load agent {self.name}: {e}")
            raise (e)

    def run(
        self, session_id: str, messages: List[UserMessage | ToolResponseMessage]
    ) -> str:
        """
        Executes the LangGraph agent with the given prompt. The agent's `invoke` method is called with the input.

        Args:
            prompt (str): The input to be processed by the agent.
        Returns:
            Any: The output from the agent's `invoke` method.
        Raises:
            Exception: If there is an error in retrieving or executing the agent's method.
        """
        print(f"Running LangGraph agent: {self.impl_class}")

        try:
            method = getattr(self.instance, self.method_name)
            config = {"configurable": {"thread_id": session_id}, "model": self.model}
            sys_msg = SystemMessage(content=self.instructions)
            lg_messages = [sys_msg] + convert_messages(messages)
            response = method().invoke({"messages": lg_messages}, config)
            last_ai_message_content = None
            for message in response["messages"]:
                if isinstance(message, AIMessage):
                    last_ai_message_content = message.content
            return last_ai_message_content
        except Exception as e:
            print(f"Failed to kickoff langgraph agent: {self.name}: {e}")
            raise (e)

    async def run_streaming(
        self, session_id: str, messages: List[UserMessage | ToolResponseMessage]
    ) -> AsyncGenerator:
        """
        Streams the execution of the LangGraph agent with the given prompt.
        Args:
            prompt (str): The input prompt to be processed by the LangGraph agent.
        """
        print(
            f"Running LangGraph agent (streaming): {self.impl_class}"
        )

        print(self.instance)
        print(self.method_name)
        method = getattr(self.instance, self.method_name)
        config = {"configurable": {"thread_id": session_id}, "model": self.model}
        sys_msg = SystemMessage(content=self.instructions)
        lg_messages = [sys_msg] + convert_messages(messages)
        processor = EventProcessor()
        async for event in method().astream_events(
            {"messages": lg_messages}, config, version="v2"
        ):
            chunk = processor.process_event(event)
            if chunk is not None:
                yield chunk
                if isinstance(
                    chunk.event.payload, AgentTurnResponseTurnCompletePayload
                ):
                    return
