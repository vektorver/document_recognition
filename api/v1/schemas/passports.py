from pydantic import BaseModel
from typing import Optional

class PassportInputShema(BaseModel):

    name: str
    surname: str
    patronymic: str
    birth_date: str
    birth_place: str
    passport_series: str
    passport_number: str
    passport_date: str
    passport_code: str
    passport_issued_by: str
    passport_issued_by_code: str

    class Config:
        orm_mode = True


class PassportOutputShema(BaseModel):

    id: int
    tg_chat_id: str
    name: str
    surname: str
    patronymic: str
    birth_date: str
    birth_place: str
    passport_series: str
    passport_number: str
    passport_date: str
    passport_code: str
    passport_issued_by: str
    passport_issued_by_code: str

    class Config:
        orm_mode = True