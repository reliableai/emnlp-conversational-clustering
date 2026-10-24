"""Prompt catalog and loading helpers for the backend_v1 clustering backend."""

from ....prompts.registry import PromptFileSpec, load_system_prompt

_BACKEND_V1_PROMPT_FILE_REGISTRY: dict[str, PromptFileSpec] = {
    "baseline_v1": PromptFileSpec(filename="clustering_agent/backend_v1/baseline_v1.system.txt"),
    "baseline_v2": PromptFileSpec(filename="clustering_agent/backend_v1/baseline_v2.system.txt"),
}

_BACKEND_V1_INTERMITTENT_CLUSTERING_JUDGE_PROMPT_FILE_REGISTRY: dict[str, PromptFileSpec] = {
    "baseline_v1": PromptFileSpec(
        filename="clustering_agent/backend_v1/intermittent_clustering_judge_v1.system.txt",
        required_template_fields=frozenset({"conversation"}),
    ),
    "baseline_v2": PromptFileSpec(
        filename="clustering_agent/backend_v1/intermittent_clustering_judge_v2.system.txt",
        required_template_fields=frozenset({"conversation"}),
    ),
}


def get_backend_v1_system_prompt(prompt_id: str) -> str:
    return load_system_prompt(
        prompt_id=prompt_id,
        registry=_BACKEND_V1_PROMPT_FILE_REGISTRY,
        registry_name="backend_v1",
    )


def get_backend_v1_intermittent_clustering_judge_prompt(prompt_id: str) -> str:
    return load_system_prompt(
        prompt_id=prompt_id,
        registry=_BACKEND_V1_INTERMITTENT_CLUSTERING_JUDGE_PROMPT_FILE_REGISTRY,
        registry_name="backend_v1_intermittent_clustering_judge",
    )
