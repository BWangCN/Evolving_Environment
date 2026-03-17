"""Language instruction generator — rule-based templates with LLM extension point."""

from __future__ import annotations

import random
from typing import Optional

from src.task.planner import ManipulationTask
from src.scene.object import TaskType


# ============================================================
# Rule-based templates.
# Each task type has multiple templates for diversity.
# {obj} = manipulated object description, {target} = target description.
# ============================================================

TEMPLATES: dict[TaskType, list[str]] = {
    TaskType.PICK: [
        "pick up the {obj}",
        "grab the {obj}",
        "grasp the {obj} from the table",
    ],
    TaskType.PLACE_ON: [
        "place the {obj} on the {target}",
        "put the {obj} on top of the {target}",
        "set the {obj} down on the {target}",
    ],
    TaskType.PLACE_IN: [
        "place the {obj} in the {target}",
        "put the {obj} inside the {target}",
        "drop the {obj} into the {target}",
    ],
    TaskType.PLACE_NEXT_TO: [
        "move the {obj} next to the {target}",
        "place the {obj} beside the {target}",
        "put the {obj} to the right of the {target}",
        "put the {obj} to the left of the {target}",
    ],
    TaskType.STACK_ON: [
        "stack the {obj} on the {target}",
        "put the {obj} on top of the {target}",
        "balance the {obj} on the {target}",
    ],
    TaskType.PUSH: [
        "push the {obj} forward",
        "push the {obj} to the side",
        "slide the {obj} across the table",
    ],
}


class LanguageGenerator:
    """Generates language instructions for manipulation tasks.

    Current implementation: rule-based template filling.
    Future: plug in LLM backend for richer, more natural instructions.

    Usage:
        gen = LanguageGenerator()
        instructions = gen.generate(task, n_variants=3)
    """

    def __init__(self, backend: str = "template"):
        """
        Args:
            backend: "template" for rule-based, "llm" for future LLM integration.
        """
        self.backend = backend
        if backend == "llm":
            self._llm_client = None  # placeholder for future LLM API client

    def generate(
        self,
        task: ManipulationTask,
        n_variants: int = 3,
        seed: Optional[int] = None,
    ) -> list[str]:
        """Generate n_variants diverse language instructions for a task.

        Returns:
            List of instruction strings.
        """
        if self.backend == "template":
            return self._generate_template(task, n_variants, seed)
        elif self.backend == "llm":
            return self._generate_llm(task, n_variants)
        raise ValueError(f"Unknown backend: {self.backend}")

    def generate_single(self, task: ManipulationTask, seed: Optional[int] = None) -> str:
        """Generate one instruction."""
        return self.generate(task, n_variants=1, seed=seed)[0]

    def _generate_template(
        self,
        task: ManipulationTask,
        n_variants: int,
        seed: Optional[int] = None,
    ) -> list[str]:
        rng = random.Random(seed)
        templates = TEMPLATES.get(task.task_type, [])
        if not templates:
            return [f"{task.task_type.value} the {task.obj.description or task.obj.category}"]

        obj_names = self._get_object_names(task.obj)
        target_names = self._get_object_names(task.target) if task.target else [""]

        results = []
        used_templates = []
        for _ in range(n_variants):
            # Pick a template we haven't used yet if possible
            available = [t for t in templates if t not in used_templates] or templates
            template = rng.choice(available)
            used_templates.append(template)

            obj_name = rng.choice(obj_names)
            target_name = rng.choice(target_names)
            instruction = template.format(obj=obj_name, target=target_name)
            results.append(instruction)

        return results

    def _generate_llm(
        self,
        task: ManipulationTask,
        n_variants: int,
    ) -> list[str]:
        """Placeholder for LLM-based generation.

        To integrate:
        1. Set self._llm_client to an API client (e.g., anthropic.Anthropic())
        2. Construct a prompt describing the task, objects, and scene
        3. Ask for n_variants diverse natural language instructions
        4. Parse and return

        Example prompt:
            "Generate {n_variants} diverse natural language instructions for a robot
             to {task_type} the {obj_description} {target_context}.
             Vary sentence structure, vocabulary, and level of detail."
        """
        raise NotImplementedError(
            "LLM backend not yet configured. "
            "Set LanguageGenerator._llm_client and implement this method. "
            "Falling back to template generation."
        )

    @staticmethod
    def _get_object_names(obj) -> list[str]:
        """Get multiple ways to refer to an object for diversity."""
        if obj is None:
            return [""]
        names = []
        if obj.description:
            names.append(obj.description)
        names.append(obj.category)
        # "the bowl" vs "the white bowl" — description usually includes adj + category
        if obj.description and obj.category not in obj.description:
            names.append(f"{obj.description} {obj.category}")
        return names if names else [obj.object_id]
