from dotenv import dotenv_values
from sqlalchemy import *
import os
import asyncio

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from api.v1.models.users import Base1
from api.v1.models.passports import Base2

dotenv_config = dotenv_values(".env")

URI = 'postgresql+asyncpg://' + dotenv_config['DB_USERNAME'] + ':' + dotenv_config['DB_PASSWORD'] + '@' + \
      dotenv_config['DB_HOST'] + '/' + \
    dotenv_config['DB_NAME']

engine = create_async_engine(URI, echo=True, future=True)


async def init_db():
    async with engine.begin() as conn:
        pass
        await conn.run_sync(Base1.metadata.drop_all)
        await conn.run_sync(Base1.metadata.create_all)
        await conn.run_sync(Base2.metadata.drop_all)
        await conn.run_sync(Base2.metadata.create_all)


async def get_session() -> AsyncSession:
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with async_session() as session:
        yield session


# async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
