from pydantic import BaseModel
from typing import List, Literal


class ExtractedEntity(BaseModel):
    title: str
    tech_stack: List[str]
    soft_skills: List[str]
    domain: str
    seniority: Literal["intern", "junior", "mid", "senior", "lead", "principal"]
