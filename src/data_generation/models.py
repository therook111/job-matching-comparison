from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class WorkExperience(BaseModel):
    role_title: str
    years_in_role: int
    # Allow the LLM to choose between bullets OR a rambling paragraph
    bullet_points: List[str] = Field(..., description("A list of 3-4 action-oriented achievements for this role. "
            "Must start with strong past-tense action verbs (e.g., 'Engineered', 'Migrated'). "
            "Explicitly mention the core technologies and tools used."))
    paragraph_description: str = Field(..., description="A dense, 3-sentence paragraph describing the role instead of bullets.")

class CandidateProfile(BaseModel):
    # --- IDENTITY & SENIORITY (Easy to mutate for Seniority Mismatch) ---
    headline_title: str = Field(..., description="The main candidate title matching the JD.")
    seniority_level: str = Field(..., description="Strictly one of: 'Junior', 'Mid-Level', 'Senior', 'Lead'.")
    total_yoe: int = Field(..., description="Total years of professional experience.")
    
    # --- SKILLS (Separated to make Tech Stack Mismatch trivial) ---
    core_tech_stack: List[str] = Field(..., description="Primary programming languages heavily emphasized in the JD.")
    tools_and_frameworks: List[str] = Field(..., description="CI/CD tools, platforms, databases (e.g., Azure DevOps).")
    scripting_and_secondary: List[str] = Field(..., description="Bonus/secondary languages (e.g., Python, Ruby).")
    methodologies: List[str] = Field(..., description="Ways of working (e.g., Agile, Scrum, TDD).")
    
    # --- DOMAIN (Kept identical during mutation to trick the model) ---
    domain_expertise: str = Field(..., description="The industry/niche mentioned in the JD (e.g., 'Flight Navigation Systems').")
    
    # --- EXPERIENCE ---
    work_history: List[WorkExperience] = Field(..., description="Chronological list of past jobs validating the JD requirements.")

    # --- IRRELEVANT FLUFF ---
    personal_interests: Optional[List[str]] = Field(default=None, description="Hobbies and interests (e.g., 'Hiking', 'Photography').")
    irrelevant_past_job: Optional[str] = Field(default=None, description="A non-tech job from early in their career (e.g., 'Bartender', 'Retail Sales').")
    unrelated_certifications: Optional[List[str]] = Field(default=None, description="A certification completely unrelated to the JD (e.g., 'CPR Certified', 'Scuba Diving').")


class CVPersona(BaseModel):
    """
    Defines a rendering persona that probabilistically masks specific fields
    of a CandidateProfile when building a CV from a Jinja2 template.

    Each entry in ``field_mask_probabilities`` maps a ``CandidateProfile``
    field name to the probability (0.0 – 1.0) that the field will be omitted
    from the rendered output.  A probability of 0.0 means the field is always
    shown; 1.0 means it is always hidden.

    Fields not listed in the dict are always rendered as-is.

    Example::

        StudentPersona = CVPersona(
            name="student",
            field_mask_probabilities={
                "methodologies": 0.6,
                "unrelated_certifications": 0.9,
                "personal_interests": 0.3,
            }
        )
    """
    name: str = Field(..., description="Human-readable label for this persona (e.g., 'student', 'career_changer').")
    field_mask_probabilities: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Maps CandidateProfile field names to masking probability (0.0=always show, "
            "1.0=always hide).  Intermediate values produce stochastic omission."
        ),
    )
