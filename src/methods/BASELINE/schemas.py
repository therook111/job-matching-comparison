from pydantic import BaseModel, model_validator
from typing import List, Any
from enum import Enum

def replace_null_strings(value: Any):
    """Recursively replace 'null' strings with None in nested structures."""
    if isinstance(value, str):
        if value.strip().lower() == "null":
            return None
    elif isinstance(value, list):
        return [replace_null_strings(v) for v in value]
    elif isinstance(value, dict):
        return {k: replace_null_strings(v) for k, v in value.items()}
    return value

class CleanBaseModel(BaseModel):
    @model_validator(mode="after")
    @classmethod
    def clean_null_strings(cls, values: Any):
        return replace_null_strings(values)

class Experience(CleanBaseModel):
    company_name: str
    time_start: str | None = None # or datetime/date if you want strict typing
    time_end: str | None = None  # same here
    position: str
    responsibilities: List[str]
    company_country: str | None = None
    company_city: str | None = None
    
class Education(CleanBaseModel):
    institution: str 
    time_start: str | None = None
    time_end: str | None = None
    GPA: str | None = None
    degree: str

class Industry(str, Enum):
    AGRICULTURE_ENVIRONMENT = "Agriculture and Environment"
    CONSTRUCTION_REAL_ESTATE = "Construction and Real Estate"
    TECHNOLOGY_IT = "Technology and IT"
    MANUFACTURING_PRODUCTION = "Manufacturing and Production"
    HEALTHCARE_LIFE_SCIENCES = "Healthcare and Life Sciences"
    EDUCATION_TRAINING = "Education and Training"
    FINANCE_INSURANCE = "Finance and Insurance"
    MARKETING_ADVERTISING = "Marketing and Advertising"
    RETAIL_SALES_CUSTOMER_SERVICE = "Retail, Sales, and Customer Service"
    TRANSPORTATION_LOGISTICS = "Transportation and Logistics"
    SPORTS_FITNESS_RECREATION = "Sports, Fitness, and Recreation"
    MEDIA_ENTERTAINMENT = "Media and Entertainment"
    HOSPITALITY_TOURISM = "Hospitality and Tourism"
    LEGAL_PROFESSIONAL_SERVICES = "Legal and Professional Services"
    ADMINISTRATIVE = "Administrative"
    NONPROFIT_CHARITABLE_WORK = "Nonprofit and Charitable Work"
    SCIENCE_RESEARCH = "Science and Research"
    ARTS_DESIGN = "Arts and Design"
    HUMAN_RESOURCES = "Human Resources (HR)"
    OTHERS = "Others"

class CVInformation(CleanBaseModel):
    education: List[Education]
    experience: List[Experience]  # now a list of Experience objects
    soft_skills: List[str]
    technical_skills: List[str]
    sector: Industry

class JDInformation(CleanBaseModel):
    education: str = "Not available."
    experience: str = "Not available."
    soft_skills: List[str] = ["Not available."]
    technical_skills: List[str] = ["Not available."]
    sector: Industry