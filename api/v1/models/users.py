from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import declarative_base
Base1 = declarative_base()

SCHEMA = 'public'


class User(Base1):
    __table_args__ = ({"schema": SCHEMA})
    __tablename__ = "users"

    id = Column(type_=Integer, unique=True, primary_key=True,
                nullable=False, autoincrement=True)
    tg_chat_id = Column(type_=String, unique=False, nullable=True)
    registration_date = Column(type_=String, unique=False, nullable=True)
    # sate in json format with various length
    state = Column(type_=String, unique=False, nullable=True)
    role = Column(type_=Integer, unique=False, nullable=True)
    
    def __repr__(self):
        return 'User<id {}>'.format(self.id)

    def to_dict(self):
        return {
            'id': self.id,
            'tg_chat_id': self.tg_chat_id,
            'registration_date': self.registration_date,
            'state': self.state, 
            'role': self.role
        }
    
