import json
import asyncio
from functools import wraps
from typing import Literal, Union
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict


# Decorator that ensures `async_completion` uses the semaphore
def with_semaphore(acquire_semaphore):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with acquire_semaphore:
                return await func(*args, **kwargs)

        return wrapper

    return decorator


class AsyncOpenAISemaphore(AsyncOpenAI):
    def __init__(self, semaphore_limit: int = 10, *args, **kwargs):
        super(AsyncOpenAISemaphore, self).__init__(*args, **kwargs)
        self.set_semaphore(semaphore_limit)

    def set_semaphore(self, semaphore_limit):
        self.semaphore_limit = semaphore_limit
        self.semaphore = asyncio.Semaphore(self.semaphore_limit)
        self.async_completion = with_semaphore(self.semaphore)(
            self.chat.completions.create
        )
        self.async_completion_with_structure = with_semaphore(self.semaphore)(
            self.__completion_with_structure
        )

    def get_semaphore(self):
        return self.semaphore

    async def __completion_with_structure(self, structure: BaseModel, *args, **kwargs):
        completion = await self.chat.completions.create(*args, **kwargs)
        content = completion.choices[0].message.content
        return structure(**json.loads(content))


class AsyncOpenAIInterface(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Literal["gpt-4o", "gpt-4o-mini"] = "gpt-4o-mini"
    temperature: float = 0.01
    top_p: float = 1
    max_output_tokens: int = 2048
    semaphore_rate: int = 10
    system_prompt: Union[str, None] = None
    openai_api_key: str = None
    client: AsyncOpenAI = None
    base_messages: list = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = AsyncOpenAISemaphore(
            semaphore_limit=self.semaphore_rate, api_key=self.openai_api_key
        )
        self.base_messages = (
            [{"role": "system", "content": self.system_prompt}]
            if self.system_prompt
            else []
        )

    def acomplete(self, prompt, role: str = "user"):
        return self.client.async_completion(
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_output_tokens,
            messages=self.base_messages + [{"role": role, "content": prompt}],
        )

    def acomplete_with_structure(self, structure, prompt, role: str = "user"):
        return self.client.async_completion_with_structure(
            structure=structure,
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_output_tokens,
            messages=self.base_messages + [{"role": role, "content": prompt}],
            response_format={"type": "json_object"},
        )
    
