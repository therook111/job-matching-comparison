from src.data_generation.models import CVPersona

ImplicitCoderPersona = CVPersona(
    name="implicit_coder",
    field_mask_probabilities={
        "core_tech_stack": 0.95,          # Almost always hidden
        "tools_and_frameworks": 0.85,     # Usually hidden
        "scripting_and_secondary": 0.95,  
        "methodologies": 0.80
    }
)

StorytellerPersona = CVPersona(
    name="storyteller",
    field_mask_probabilities={
        "work_history.bullet_points": 0.70, # 90% chance to drop bullets (forcing paragraph fallback)
        "methodologies": 0.50
    }
)

ActionOrientedPersona = CVPersona(
    name="action_oriented",
    field_mask_probabilities={
        "work_history.paragraph_description": 1.0,  # ALWAYS hidden (forces bullets)
        "personal_interests": 1.0,                  # Always hidden
        "irrelevant_past_job": 1.0,                 # Always hidden
        "unrelated_certifications": 1.0             # Always hidden
    }
)

StrictlyCorePersona = CVPersona(
    name="strictly_core",
    field_mask_probabilities={
        "scripting_and_secondary": 0.95,  # Almost always hidden
        "methodologies": 0.90,            # Almost always hidden
        "tools_and_frameworks": 0.60,
        "personal_interests": 0.85        # Usually hidden
    }
)

DateImplicitSenior = CVPersona(
    name="date_implicit_senior",
    field_mask_probabilities={
        "total_yoe":                1.00,  # always omitted — infer from dates
        "seniority_level":          0.90,  # almost always omitted
        "personal_interests":       0.70,
        "unrelated_certifications": 0.80,
        "irrelevant_past_job":      0.75,
        "methodologies":            0.55,
        "scripting_and_secondary":  0.45,
    }
)

KeywordStufferPersona = CVPersona(
    name="keyword_stuffer",
    field_mask_probabilities={
    }
)

PERSONA_LIBRARY =[
    ImplicitCoderPersona, 
    StorytellerPersona, 
    ActionOrientedPersona, 
    StrictlyCorePersona,
    DateImplicitSenior,
    KeywordStufferPersona
]

__all__ = [
    "PERSONA_LIBRARY"
]