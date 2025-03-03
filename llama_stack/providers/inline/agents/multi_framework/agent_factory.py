
from enum import Enum
from typing import Callable, Type, Union
from .langgraph.agent import LangGraphAgent

class AgentFramework(Enum):
    """Enumeration of supported frameworks"""
    CREWAI = "crewai"
    LANGGRAPH = "langgraph"

class AgentFactory:
    """Factory class for handling agent frameworks"""

    @staticmethod
    def create_agent(framework: Union[str, AgentFramework]) -> Callable[..., LangGraphAgent]:
        """Create an instance of the specified agent framework.

        Args:
            framework (Union[str, AgentFramework]): The framework to create. Must be a valid enum value.

        Returns:
            A new instance of the corresponding agent class.
        """
        # If the input is a string, convert it to an AgentFramework
        if isinstance(framework, str):
            try:
                framework = AgentFramework(framework.lower())
            except ValueError:
                raise ValueError(f"Unknown framework: {framework}")

        factories = {
            # AgentFramework.CREWAI: CrewAIAgent,
            AgentFramework.LANGGRAPH: LangGraphAgent,
        }

        if framework not in factories:
            raise ValueError(f"Unknown framework: {framework}")
        
        return factories[framework]

    @classmethod
    def get_factory(cls, framework: str) -> Callable[..., LangGraphAgent]:
        """Get a factory function for the specified agent type."""
        return cls.create_agent(framework)

