from pydantic import BaseModel
from typing import Literal

class GeneratedCV(BaseModel):
    cv_text: str 

class PoisonedCV(BaseModel):
    modification_type: Literal[
        "Seniority Mismatch",
        "Stack Mismatch",
        "Role Mismatch",
    ]
    cv_text: str
