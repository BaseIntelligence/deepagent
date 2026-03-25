"""LLM client abstraction for swe_forge."""

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A message in a conversation with an LLM."""

    model_config = {"extra": "forbid"}

    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list["ToolCall"] | None = None
    tool_call_id: str | None = None

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str) -> "Message":
        return cls(role="assistant", content=content)

    @classmethod
    def assistant_with_tool_calls(
        cls, content: str, tool_calls: list["ToolCall"]
    ) -> "Message":
        return cls(role="assistant", content=content, tool_calls=tool_calls)

    @classmethod
    def tool_result(cls, call_id: str, content: str) -> "Message":
        return cls(role="tool", content=content, tool_call_id=call_id)


class ToolCall(BaseModel):
    """A tool call made by the model."""

    model_config = {"extra": "forbid"}

    id: str
    type: Literal["function"] = "function"
    function: "FunctionCall"


class FunctionCall(BaseModel):
    """Function name and arguments within a tool call."""

    model_config = {"extra": "forbid"}

    name: str
    arguments: str  # JSON string


class FunctionDefinition(BaseModel):
    """Definition of a function for tool calling."""

    model_config = {"extra": "forbid"}

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolDefinition(BaseModel):
    """A tool definition for function calling."""

    model_config = {"extra": "forbid"}

    type: Literal["function"] = "function"
    function: FunctionDefinition

    @classmethod
    def create(
        cls, name: str, description: str, parameters: dict[str, Any]
    ) -> "ToolDefinition":
        return cls(
            type="function",
            function=FunctionDefinition(
                name=name, description=description, parameters=parameters
            ),
        )


class ToolChoiceFunction(BaseModel):
    """Specifies which function to force."""

    model_config = {"extra": "forbid"}

    name: str


class ToolChoice(BaseModel):
    """Tool choice configuration."""

    model_config = {"extra": "forbid"}

    type: Literal["function"] = "function"
    function: ToolChoiceFunction

    @classmethod
    def auto(cls) -> str:
        return "auto"

    @classmethod
    def none(cls) -> str:
        return "none"

    @classmethod
    def force(cls, name: str) -> "ToolChoice":
        return cls(type="function", function=ToolChoiceFunction(name=name))


class GenerationRequest(BaseModel):
    """Request for text generation from an LLM."""

    model_config = {"extra": "forbid"}

    model: str
    messages: list[Message]
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    tools: list[ToolDefinition] | None = None
    tool_choice: ToolChoice | str | None = None

    def with_temperature(self, temperature: float) -> "GenerationRequest":
        return self.model_copy(update={"temperature": temperature})

    def with_max_tokens(self, max_tokens: int) -> "GenerationRequest":
        return self.model_copy(update={"max_tokens": max_tokens})

    def with_top_p(self, top_p: float) -> "GenerationRequest":
        return self.model_copy(update={"top_p": top_p})

    def with_tool(self, tool: ToolDefinition) -> "GenerationRequest":
        return self.model_copy(
            update={
                "tools": [tool],
                "tool_choice": ToolChoice.force(tool.function.name),
            }
        )


class Usage(BaseModel):
    """Token usage statistics for a generation request."""

    model_config = {"extra": "forbid"}

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    """A single generated choice from the LLM."""

    model_config = {"extra": "forbid"}

    index: int
    message: Message
    finish_reason: str


class GenerationResponse(BaseModel):
    """Response from an LLM generation request."""

    model_config = {"extra": "forbid"}

    id: str
    model: str
    choices: list[Choice]
    usage: Usage

    def first_content(self) -> str | None:
        if self.choices:
            return self.choices[0].message.content
        return None


class LLMClient(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def complete(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a response for the given request."""
        ...

    @abstractmethod
    async def complete_with_tools(
        self, request: GenerationRequest
    ) -> GenerationResponse:
        """Generate a response with tool calling support."""
        ...

    @abstractmethod
    async def stream(self, request: GenerationRequest) -> None:
        """Stream a response for the given request."""
        ...


# Update forward references
Message.model_rebuild()
