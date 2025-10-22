from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime

class Requirement(BaseModel):
    id: Optional[int] = None
    req_id: str
    text: str
    category: Optional[str] = None
    priority: Optional[str] = None

class RequirementUploadResponse(BaseModel):
    message: str
    total_requirements: int
    requirements: List[Requirement]

class Dependency(BaseModel):
    source_id: str
    target_id: str
    similarity_score: float

class DependencyGraphResponse(BaseModel):
    nodes: List[Dict]
    edges: List[Dict]
    total_dependencies: int
