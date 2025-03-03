from llama_stack.apis.agents import (
    AgentTurnResponseStreamChunk,
    AgentTurnResponseEvent,
    AgentTurnResponseStepProgressPayload,
    AgentTurnResponseTurnStartPayload,
    AgentTurnResponseTurnCompletePayload,
    InferenceStep,
    AgentTurnResponseStepStartPayload,
    AgentTurnResponseStepCompletePayload,
    StepType,
    Turn,
    ToolExecutionStep,
    ToolCall,
    ToolResponse
)
from typing import List
from llama_stack.apis.inference import (
    UserMessage,CompletionMessage
)
from llama_stack.apis.common.content_types import (
    TextDelta,
)
from llama_models.datatypes import (
    RawContent,
    RawMediaItem,
    RawMessage,
    RawTextItem,
    StopReason,
)

import logging
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from datetime import datetime
import json
import pprint

log = logging.getLogger(__name__)


def extract_messages(messages) ->(List[UserMessage] | CompletionMessage):
    user_messages = []
    completion_message = None

    for message in messages:
        if isinstance(message, HumanMessage):
            user_messages = user_messages + [UserMessage(content=message.content)]
        if isinstance(message, AIMessage):
            completion_message = CompletionMessage(content=message.content, stop_reason=StopReason.end_of_turn)

    return user_messages, completion_message


class EventProcessor:
    """Processes events streamed from the langgraph graph and converts
    them in the format used by llama stack"""

    def __init__(self):
        self.start_run_id = None
        self.turn_start_time = datetime.now()
        self.tool_start_times = {}

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

        # pp = pprint.PrettyPrinter(indent=2)
        # pp.pprint(event)

        handler = handlers.get(event_type, self.handle_unknown_event)
        return handler(event)

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

        if run_id != self.start_run_id:
            return

        print("end streaming")

        try:
            messages = event.get("data", {}).get("output", {}).get("messages", [])

            user_messages, completion_message = extract_messages(messages)

            return AgentTurnResponseStreamChunk(
                event=AgentTurnResponseEvent(
                    payload=AgentTurnResponseTurnCompletePayload(
                        turn=Turn(
                            turn_id=run_id,
                            session_id=event.get("metadata", {}).get("thread_id", {}),
                            input_messages=user_messages,
                            output_message=completion_message,
                            started_at=self.turn_start_time,
                            completed_at=datetime.now(),
                            steps=[],
                        )
                    )
                )
            )

        except Exception as e:
            log.error(f"Error in handle_chain_end: {e}")

    def handle_chain_start(self, event):
        if self.start_run_id is None:
            self.start_run_id = event.get("run_id")
            return AgentTurnResponseStreamChunk(
                event=AgentTurnResponseEvent(
                    payload=AgentTurnResponseTurnStartPayload(
                        turn_id=self.start_run_id,
                    )
                )
            )

    def on_chain_stream(self, event):
        pass

    def on_chat_model_start(self, event):
        pass

    def on_chat_model_end(self, event):
        pass

    def on_tool_start(self, event):
        step_id = event.get("run_id")
        self.tool_start_times[step_id] = datetime.now()
        return AgentTurnResponseStreamChunk(
                    event=AgentTurnResponseEvent(
                        payload=AgentTurnResponseStepStartPayload(
                            step_type=StepType.tool_execution.value,
                            step_id=step_id,
                        )
                    )
                )

    def on_tool_end(self, event):
        step_id = event.get("run_id")
        tool_output = event.get("data", {}).get("output", {})
        tool_call = ToolCall(
            call_id=tool_output.tool_call_id,
            tool_name=tool_output.name,
            arguments=event.get("data", {}).get("input", {})
        )
        tool_execution_step = ToolExecutionStep(
                step_id=event.get("run_id"),
                turn_id=self.start_run_id,
                tool_calls=[tool_call],
                tool_responses=[
                    ToolResponse(
                        call_id=tool_output.tool_call_id,
                        tool_name=tool_output.name,
                        content=tool_output.content,
                    )
                ],
                completed_at=datetime.now(),
                started_at=self.tool_start_times[step_id],
            )
        return AgentTurnResponseStreamChunk(
                event=AgentTurnResponseEvent(
                    payload=AgentTurnResponseStepCompletePayload(
                        step_type=StepType.tool_execution.value,
                        step_id=event.get("run_id"),
                        step_details=tool_execution_step,
                    )
                )
            )

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

    for message in ls_messages:
        print(message.content)
    lg_messages = [HumanMessage(content=message.content) for message in ls_messages]

    return lg_messages
