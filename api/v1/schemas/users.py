from pydantic import BaseModel
from typing import Optional

class UserInputShema(BaseModel):
    tg_chat_id: str
    state: str
    role: Optional[int]

    class Config:
        orm_mode = True

class UserOutputShema(BaseModel):
    id: int
    tg_chat_id: str
    registration_date: str
    state: str
    role: int

    class Config:
        orm_mode = True


class UserAnswerShema(BaseModel):
    tg_chat_id: str
    word_id: int
    mode: int
    answer: str

    class Config:
        orm_mode = True