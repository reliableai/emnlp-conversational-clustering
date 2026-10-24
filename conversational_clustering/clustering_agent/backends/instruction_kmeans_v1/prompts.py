"""Prompt catalog and loading helpers for the instruction_kmeans_v1 backend."""

from ....prompts.registry import PromptFileSpec, load_system_prompt

_INSTRUCTION_KMEANS_V1_PROMPT_FILE_REGISTRY: dict[str, PromptFileSpec] = {
    "baseline_v1": PromptFileSpec(
        filename="clustering_agent/instruction_kmeans_v1/baseline_v1.system.txt"
    ),
}

_INSTRUCTION_KMEANS_V1_INTERMITTENT_JUDGE_PROMPT_FILE_REGISTRY: dict[str, PromptFileSpec] = {
    "baseline_v1": PromptFileSpec(
        filename="clustering_agent/instruction_kmeans_v1/intermittent_clustering_judge_v1.system.txt",
        required_template_fields=frozenset({"conversation"}),
    ),
}


def get_instruction_kmeans_v1_system_prompt(prompt_id: str) -> str:
    return load_system_prompt(
        prompt_id=prompt_id,
        registry=_INSTRUCTION_KMEANS_V1_PROMPT_FILE_REGISTRY,
        registry_name="instruction_kmeans_v1",
    )


def get_instruction_kmeans_v1_intermittent_clustering_judge_prompt(prompt_id: str) -> str:
    return load_system_prompt(
        prompt_id=prompt_id,
        registry=_INSTRUCTION_KMEANS_V1_INTERMITTENT_JUDGE_PROMPT_FILE_REGISTRY,
        registry_name="instruction_kmeans_v1_intermittent_clustering_judge",
    )
