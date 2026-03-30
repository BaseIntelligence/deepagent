"""Generate natural prompts for SWE tasks using LLM."""

from logging import getLogger

logger = getLogger(__name__)


async def generate_task_prompt(llm_client, task) -> str:
    """Generate a natural prompt describing the PR changes.
    
    Args:
        llm_client: LLM client for generation
        task: SweTask with PR info
        
    Returns:
        Natural prompt like a user would write
    """
    if not llm_client:
        # Fallback to original prompt
        return task.prompt or f"Changes in {task.repo}"
    
    try:
        from swe_forge.llm.models import GenerationRequest, GenerationResponse
        
        system = "You write short, natural descriptions of code changes."
        
        user = f"""Describe this GitHub PR in one sentence, like a developer would explain it to a colleague.

PR title: {task.prompt[:200]}
Repo: {task.repo}

Just describe what changed, nothing else. Keep it under 50 words."""

        request = GenerationRequest(
            model=llm_client.default_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
            max_tokens=100,
        )
        
        response = await llm_client.complete(request)
        
        if response.choices and response.choices[0].message:
            return response.choices[0].message.content.strip()
    except Exception as e:
        logger.debug(f"LLM prompt generation failed: {e}")
    
    # Fallback
    return task.prompt or f"Changes in {task.repo}"
