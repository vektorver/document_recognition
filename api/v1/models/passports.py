from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import declarative_base
Base2 = declarative_base()

SCHEMA = 'public'


class Passport(Base2):
    __table_args__ = ({"schema": SCHEMA})
    __tablename__ = "passport"

    id = Column(type_=Integer, unique=True, primary_key=True,
                nullable=False, autoincrement=True)
    tg_chat_id = Column(type_=String, unique=False, nullable=True)
    name = Column(type_=String, unique=False, nullable=True)
    surname = Column(type_=String, unique=False, nullable=True)
    patronymic = Column(type_=String, unique=False, nullable=True)
    birth_date = Column(type_=String, unique=False, nullable=True)
    birth_place = Column(type_=String, unique=False, nullable=True)
    passport_series = Column(type_=String, unique=False, nullable=True)
    passport_number = Column(type_=String, unique=False, nullable=True)
    passport_date = Column(type_=String, unique=False, nullable=True)
    passport_code = Column(type_=String, unique=False, nullable=True)
    passport_issued_by = Column(type_=String, unique=False, nullable=True)
    passport_issued_by_code = Column(type_=String, unique=False, nullable=True)

    def __repr__(self):

        return 'Passport<id {}>'.format(self.id)
    
    def to_dict(self):

        return {
            'id': self.id,
            'tg_chat_id': self.tg_chat_id,
            'name': self.name,
            'surname': self.surname,
            'patronymic': self.patronymic,
            'birth_date': self.birth_date,
            'birth_place': self.birth_place,
            'passport_series': self.passport_series,
            'passport_number': self.passport_number,
            'passport_date': self.passport_date,
            'passport_code': self.passport_code,
            'passport_issued_by': self.passport_issued_by,
            'passport_issued_by_code': self.passport_issued_by_code
        }