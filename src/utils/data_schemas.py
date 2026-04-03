from pydantic import BaseModel
from typing import Literal, List

class GeneratedSummary(BaseModel):
    summary: str
