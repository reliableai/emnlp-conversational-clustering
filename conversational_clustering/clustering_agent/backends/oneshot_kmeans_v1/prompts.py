"""Prompt loading helpers for the oneshot_kmeans_v1 clustering backend."""

from ....prompts.registry import PromptFileSpec, load_system_prompt

_ONESHOT_KMEANS_V1_PROMPT_REGISTRY: dict[str, PromptFileSpec] = {
    "baseline_v1": PromptFileSpec(
        filename="clustering_agent/oneshot_kmeans_v1/baseline_v1.system.txt"
    )
}


def get_oneshot_kmeans_v1_system_prompt() -> str:
    return load_system_prompt(
        prompt_id="baseline_v1",
        registry=_ONESHOT_KMEANS_V1_PROMPT_REGISTRY,
        registry_name="oneshot_kmeans_v1",
    )
