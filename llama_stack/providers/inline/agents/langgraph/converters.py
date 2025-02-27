
from llama_stack.apis.agents import (
    AgentTurnResponseStreamChunk,
    AgentTurnResponseEvent,
    AgentTurnResponseStepProgressPayload,
    StepType,
)
from typing import  List
from llama_stack.apis.inference import (
    UserMessage,
)
from llama_stack.apis.common.content_types import (
    TextDelta,
)
import logging
from langchain_core.messages import SystemMessage, HumanMessage

class EventProcessor:
    """Processes events streamed from the langgraph graph and converts
    them in the format used by llama stack"""

    def __init__(self):
        self.start_run_id = None

    def process_event(self, event: dict):
        handlers = {
            "on_chain_start": self.handle_chain_start,
            "on_chain_stream": self.on_chain_stream,
            "on_chain_end": self.handle_chain_end,
            "on_chat_model_start": self.on_chat_model_start,
            "on_chat_model_stream": self.handle_chat_model_stream,
            "on_chat_model_end": self.on_chat_model_end,
            "on_tool_start": self.on_tool_start,
            "on_tool_end": self.on_tool_end,
        }
        event_type = event.get("event")
        handler = handlers.get(event_type, self.handle_unknown_event)
        result = handler(event)

        # handle end of run - we use a boolean to mark it
        if result == False:
            return False
        # Check if the handler is a generator and yield results if any
        if result is not None:
            return result

    def handle_chat_model_stream(self, event):
        try:
            chunk_content = event.get("data", {}).get("chunk", {}).content
            if chunk_content:
                return AgentTurnResponseStreamChunk(
                    event=AgentTurnResponseEvent(
                        payload=AgentTurnResponseStepProgressPayload(
                            step_type=StepType.inference.value,
                            step_id="step_id",
                            delta=TextDelta(text=f"{chunk_content}"),
                        )
                    )
                )
        except Exception as e:
            log.error(f"Error in handle_chat_model_stream: {e}")

    def handle_chain_end(self, event):
        run_id = event.get("run_id")
        if run_id == self.start_run_id:
            print("end streaming")
            return False

    def handle_chain_start(self, event):
        if self.start_run_id is None:
            self.start_run_id = event.get("run_id")

    def on_chain_stream(self, event):
        pass

    def on_chat_model_start(self, event):
        pass

    def on_chat_model_end(self, event):
        pass

    def on_tool_start(self, event):
        pass

    def on_tool_end(self, event):
        pass

    def handle_unknown_event(self, event):
        print(f"Unknown event type: {event.get('event')}")


def convert_messages(ls_messages: List[UserMessage]) -> List[HumanMessage]:
        """
        Converts messages from the format used in lllama-stack to the format used in langgraph.

        Parameters:
            ls_messages (List[UserMessage]): A list of user messages in llama-stack format.

        Returns:
            List[HumanMessage]: A list of human messages in langgraph format.
        """
    
        if not ls_messages:
            return []
        
        lg_messages = [HumanMessage(content=message.content) for message in ls_messages]
        
        return lg_messages