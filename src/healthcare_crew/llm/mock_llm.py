from __future__ import annotations

from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


MOCK_START = "<MOCK_JSON_START>"
MOCK_END = "<MOCK_JSON_END>"


class MockChatModel(BaseChatModel):
	model_name: str = "mock-llm"

	@property
	def _llm_type(self) -> str:
		return "mock-llm"

	def _generate(
		self,
		messages: List[BaseMessage],
		stop: Optional[List[str]] = None,
		run_manager: Any = None,
		**kwargs: Any,
	) -> ChatResult:
		# Look through messages to find an embedded MOCK JSON section to echo back
		joined = "\n\n".join(getattr(m, "content", "") for m in messages)
		start_idx = joined.find(MOCK_START)
		end_idx = joined.find(MOCK_END)
		if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
			json_payload = joined[start_idx + len(MOCK_START):end_idx].strip()
			content = json_payload
		else:
			# Basic placeholder to avoid crashes
			content = "{}"

		message = AIMessage(content=content)
		generation = ChatGeneration(message=message)
		return ChatResult(generations=[generation])

	async def _agenerate(
		self,
		messages: List[BaseMessage],
		stop: Optional[List[str]] = None,
		run_manager: Any = None,
		**kwargs: Any,
	) -> ChatResult:
		return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)