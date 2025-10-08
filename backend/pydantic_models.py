from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
from typing import Optional


class ModelName(str, Enum):
    GPT4_O = 'gpt-4o'
    GPT4_O_MINI = 'gpt-4o-mini'

class QueryInput(BaseModel):
    question: str
    session_id: Optional[str] = Field(default=None)  # Fixed: made Optional and consistent naming
    model: ModelName = Field(default=ModelName.GPT4_O_MINI)

class QueryResponse(BaseModel):
    answer: str
    session_id: str
    model: ModelName

class DocumentInfo(BaseModel):
    id: int
    filename: str
    uploaded_timestamp: datetime  # Fixed: match your database column name

class DeleteFileRequest(BaseModel):
    file_id: int