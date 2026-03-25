import pytest

from swe_forge.llm import (
    Choice,
    FunctionDefinition,
    GenerationRequest,
    GenerationResponse,
    LLMClient,
    Message,
    ToolCall,
    ToolChoice,
    ToolDefinition,
    Usage,
)
from swe_forge.llm.client import FunctionCall


class TestMessageFactoryMethods:
    def test_system_factory(self):
        msg = Message.system("You are helpful.")
        assert msg.role == "system"
        assert msg.content == "You are helpful."
        assert msg.tool_calls is None
        assert msg.tool_call_id is None

    def test_user_factory(self):
        msg = Message.user("Hello!")
        assert msg.role == "user"
        assert msg.content == "Hello!"
        assert msg.tool_calls is None
        assert msg.tool_call_id is None

    def test_assistant_factory(self):
        msg = Message.assistant("Hi there!")
        assert msg.role == "assistant"
        assert msg.content == "Hi there!"
        assert msg.tool_calls is None
        assert msg.tool_call_id is None

    def test_assistant_with_tool_calls_factory(self):
        tool_call = ToolCall(
            id="call_123",
            type="function",
            function=FunctionCall(name="get_weather", arguments='{"city": "NYC"}'),
        )
        msg = Message.assistant_with_tool_calls("Thinking...", [tool_call])
        assert msg.role == "assistant"
        assert msg.content == "Thinking..."
        assert msg.tool_calls == [tool_call]
        assert msg.tool_call_id is None

    def test_tool_result_factory(self):
        msg = Message.tool_result("call_123", '{"temp": 72}')
        assert msg.role == "tool"
        assert msg.content == '{"temp": 72}'
        assert msg.tool_calls is None
        assert msg.tool_call_id == "call_123"


class TestMessageValidation:
    def test_message_rejects_extra_fields(self):
        with pytest.raises(Exception):
            Message(role="user", content="test", extra_field="not_allowed")

    def test_message_valid_roles(self):
        for role in ["system", "user", "assistant", "tool"]:
            msg = Message(role=role, content="test")
            assert msg.role == role


class TestToolCall:
    def test_tool_call_creation(self):
        tool_call = ToolCall(
            id="call_abc",
            type="function",
            function=FunctionCall(name="test_func", arguments='{"arg": "value"}'),
        )
        assert tool_call.id == "call_abc"
        assert tool_call.type == "function"
        assert tool_call.function.name == "test_func"
        assert tool_call.function.arguments == '{"arg": "value"}'

    def test_tool_call_rejects_extra_fields(self):
        with pytest.raises(Exception):
            ToolCall(
                id="call_abc",
                type="function",
                function=FunctionCall(name="test", arguments="{}"),
                extra="not_allowed",
            )


class TestToolDefinition:
    def test_tool_definition_create(self):
        params = {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        }
        tool = ToolDefinition.create(
            name="get_weather", description="Get weather", parameters=params
        )
        assert tool.type == "function"
        assert tool.function.name == "get_weather"
        assert tool.function.description == "Get weather"
        assert tool.function.parameters == params

    def test_tool_definition_rejects_extra_fields(self):
        with pytest.raises(Exception):
            ToolDefinition(
                type="function",
                function=FunctionDefinition(name="test", description="test"),
                extra="not_allowed",
            )


class TestToolChoice:
    def test_tool_choice_auto(self):
        result = ToolChoice.auto()
        assert result == "auto"

    def test_tool_choice_none(self):
        result = ToolChoice.none()
        assert result == "none"

    def test_tool_choice_force(self):
        choice = ToolChoice.force("get_weather")
        assert choice.type == "function"
        assert choice.function.name == "get_weather"


class TestGenerationRequestBuilder:
    def test_basic_request(self):
        request = GenerationRequest(model="gpt-4", messages=[Message.user("Hello")])
        assert request.model == "gpt-4"
        assert len(request.messages) == 1
        assert request.temperature is None
        assert request.max_tokens is None

    def test_with_temperature(self):
        request = GenerationRequest(model="gpt-4", messages=[Message.user("Hello")])
        updated = request.with_temperature(0.7)
        assert updated.temperature == 0.7
        assert updated.model == "gpt-4"

    def test_with_max_tokens(self):
        request = GenerationRequest(model="gpt-4", messages=[Message.user("Hello")])
        updated = request.with_max_tokens(1000)
        assert updated.max_tokens == 1000

    def test_with_top_p(self):
        request = GenerationRequest(model="gpt-4", messages=[Message.user("Hello")])
        updated = request.with_top_p(0.9)
        assert updated.top_p == 0.9

    def test_with_tool(self):
        tool = ToolDefinition.create(
            name="get_weather",
            description="Get weather",
            parameters={"type": "object"},
        )
        request = GenerationRequest(model="gpt-4", messages=[Message.user("Hello")])
        updated = request.with_tool(tool)
        assert updated.tools == [tool]
        assert isinstance(updated.tool_choice, ToolChoice)
        assert updated.tool_choice.function.name == "get_weather"

    def test_rejects_extra_fields(self):
        with pytest.raises(Exception):
            GenerationRequest(model="gpt-4", messages=[], extra_field="not_allowed")


class TestGenerationResponse:
    def test_first_content_with_choices(self):
        response = GenerationResponse(
            id="resp_123",
            model="gpt-4",
            choices=[
                Choice(
                    index=0,
                    message=Message.assistant("Hello!"),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        assert response.first_content() == "Hello!"

    def test_first_content_empty_choices(self):
        response = GenerationResponse(
            id="resp_123",
            model="gpt-4",
            choices=[],
            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )
        assert response.first_content() is None


class TestUsage:
    def test_usage_creation(self):
        usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150


class TestChoice:
    def test_choice_creation(self):
        choice = Choice(
            index=0,
            message=Message.assistant("Response"),
            finish_reason="stop",
        )
        assert choice.index == 0
        assert choice.message.content == "Response"
        assert choice.finish_reason == "stop"


class TestLLMClientAbstract:
    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            LLMClient()

    async def test_subclass_must_implement_methods(self):
        class IncompleteClient(LLMClient):
            pass

        with pytest.raises(TypeError):
            IncompleteClient()


class TestModelSerialization:
    def test_message_json_serialization(self):
        msg = Message.user("Hello")
        json_str = msg.model_dump_json()
        assert '"role":"user"' in json_str
        assert '"content":"Hello"' in json_str

    def test_message_json_deserialization(self):
        msg = Message.model_validate({"role": "user", "content": "Hello"})
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_tool_call_json_roundtrip(self):
        tool_call = ToolCall(
            id="call_123",
            type="function",
            function=FunctionCall(name="test", arguments='{"x": 1}'),
        )
        json_str = tool_call.model_dump_json()
        parsed = ToolCall.model_validate_json(json_str)
        assert parsed.id == "call_123"
        assert parsed.function.name == "test"
        assert parsed.function.arguments == '{"x": 1}'
