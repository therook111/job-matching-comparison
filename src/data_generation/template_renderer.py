"""
template_renderer.py

Utility for rendering a CandidateProfile (Pydantic model) into a coherent
CV string using a Jinja2 template stored in the TEMPLATES directory.

Supports an optional CVPersona that probabilistically masks specific fields
before the template is rendered, simulating real-world CV variability.

Field masking keys support dot-notation for nested fields, e.g.::

    "work_history.bullet_points"  # masks bullet_points inside every WorkExperience
    "methodologies"               # masks the top-level field as before
"""

import copy
import random
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader, TemplateNotFound

from .models import CandidateProfile, CVPersona

# Resolve the TEMPLATES directory relative to this file so the renderer works
# regardless of the caller's working directory.
_TEMPLATES_DIR = Path(__file__).parent / "TEMPLATES"


def _mask_nested(obj: dict, path: list, probability: float, rng: random.Random) -> None:
    """
    Recursively walk *path* inside *obj* and mask the leaf key in-place.

    When an intermediate value is a ``list``, the mask roll is made
    **independently for each dict element** in that list.
    """
    if not path or not isinstance(obj, dict):
        return
    key, *rest = path
    if key not in obj:
        return
    if rest:
        child = obj[key]
        if isinstance(child, list):
            for item in child:
                _mask_nested(item, rest, probability, rng)
        elif isinstance(child, dict):
            _mask_nested(child, rest, probability, rng)
    else:
        # Leaf – roll once and mask if below the threshold
        if rng.random() < probability:
            obj[key] = None


def _apply_persona_mask(context: dict, persona: CVPersona, rng: random.Random) -> dict:
    """
    Return a deep copy of *context* with fields stochastically set to
    ``None`` according to the persona's masking probabilities.

    Keys in ``field_mask_probabilities`` may be:

    * **Top-level** (e.g. ``"methodologies"``): masks the whole field.
    * **Dot-notation nested** (e.g. ``"work_history.bullet_points"``): for
      every dict item in the parent list the child field is masked
      independently.

    A field is masked when a draw from [0, 1) is strictly less than its
    configured probability.  Unknown or unreachable paths are silently skipped.
    """
    masked = copy.deepcopy(context)
    for field_path, probability in persona.field_mask_probabilities.items():
        parts = field_path.split(".")
        if len(parts) == 1:
            # Top-level field – original behaviour
            if field_path in masked and rng.random() < probability:
                masked[field_path] = None
        else:
            parent_key, *rest = parts
            if parent_key not in masked:
                continue
            parent = masked[parent_key]
            if isinstance(parent, list):
                for item in parent:
                    _mask_nested(item, rest, probability, rng)
            elif isinstance(parent, dict):
                _mask_nested(parent, rest, probability, rng)
    return masked


def render_cv(
    profile: CandidateProfile,
    template_name: str = "cv_template.jinja2",
    persona: Optional[CVPersona] = None,
    seed: Optional[int] = None,
) -> str:
    """
    Render a ``CandidateProfile`` into a CV string using a Jinja2 template.
    """
    if not isinstance(profile, CandidateProfile):
        raise ValueError(
            f"Expected a CandidateProfile instance, got {type(profile).__name__!r}"
        )

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        # Keep whitespace predictable: strip leading/trailing blank lines that
        # come from block tags ({% %}) so the output stays tidy.
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )

    try:
        template = env.get_template(template_name)
    except TemplateNotFound:
        available = [p.name for p in _TEMPLATES_DIR.glob("*.jinja2")]
        raise TemplateNotFound(
            f"Template '{template_name}' not found in {_TEMPLATES_DIR}. "
            f"Available templates: {available}"
        )

    # Convert the Pydantic model to a plain dict so Jinja2 can access fields
    # directly by name (e.g. {{ headline_title }}).
    context = profile.model_dump()

    if persona is not None:
        rng = random.Random(seed)
        context = _apply_persona_mask(context, persona, rng)

    return template.render(**context)
