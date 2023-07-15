from sqlalchemy import select
from typing import List
from fastapi.responses import JSONResponse
from api.v1.schemas.users import UserOutputShema
from api.v1.models.users import User
from config import get_session
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
import pandas as pd

from fastapi import APIRouter, Depends

router = APIRouter()
from sqlalchemy.sql import exists

    
@router.post('/api/v1/user/{tg_chat_id}',
             tags=["user"],
             response_model=UserOutputShema,
             responses={
                 200: {
                     "description": 'Success'
                 },
                 422: {
                     "description": "User already exists"
                 }
             })
async def register_new_user(
    tg_chat_id: str,
    session: AsyncSession = Depends(get_session)
):
    # check if user already exists
    stmt = select(User).where(User.tg_chat_id == tg_chat_id)
    result = await session.execute(stmt)
    user = result.scalars().first()
    if user:
        return JSONResponse(status_code=422, content='User already exists')

    # create new user
    new_user = User(
        tg_chat_id=str(tg_chat_id),
        registration_date=str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        state='registered', 
        role=0
    )
    session.add(new_user)
    await session.commit()
    await session.refresh(new_user)
    return new_user


# router for getting all users
@router.get('/api/v1/users',
            tags=["user"],
            response_model=List[UserOutputShema],
            responses={
                200: {
                    "description": 'Success'
                }
            })
async def get_all_users(
    session: AsyncSession = Depends(get_session)
):
    stmt = select(User)
    result = await session.execute(stmt)
    users = result.scalars().all()
    return users


# delete user by tg_chat_id
@router.delete('/api/v1/user/{tg_chat_id}',
                tags=["user"],
                responses={
                    200: {
                        "description": 'Success'
                    },
                    404: {
                        "description": "User not found"
                    }
                })
async def delete_user(
    tg_chat_id: str,
    session: AsyncSession = Depends(get_session)
):
    stmt = select(User).where(User.tg_chat_id == tg_chat_id)
    result = await session.execute(stmt)
    user = result.scalars().first()
    if not user:
        return JSONResponse(status_code=404, content='User not found')
    await session.delete(user)
    await session.commit()
    return JSONResponse(status_code=200, content='Success')


# router for setting state of user by tg_chat_id
@router.put('/api/v1/user/{tg_chat_id}/state/{state}',
            tags=["user"],
            response_model=UserOutputShema,
            responses={
                200: {
                    "description": 'Success'
                },
                422: {
                    "description": "User not found"
                }
            })
async def set_user_state(
    tg_chat_id: str,
    state: str,
    session: AsyncSession = Depends(get_session)
):
    # check if user already exists
    stmt = select(User).where(User.tg_chat_id == tg_chat_id)
    result = await session.execute(stmt)
    user = result.scalars().first()
    if not user:
        return JSONResponse(status_code=422, content='User not found')

    # set user state
    user.state = state
    await session.commit()
    await session.refresh(user)
    return user


# router for getting user by tg_chat_id
@router.get('/api/v1/user/{tg_chat_id}',
            tags=["user"],
            response_model=UserOutputShema,
            responses={
                200: {
                    "description": 'Success'
                },
                404: {
                    "description": "User not found"
                }
            })
async def get_user_by_tg_chat_id(
    tg_chat_id: str,
    session: AsyncSession = Depends(get_session)
):
    stmt = select(User).where(User.tg_chat_id == tg_chat_id)
    result = await session.execute(stmt)
    user = result.scalars().first()
    if not user:
        return JSONResponse(status_code=404, content='User not found')
    return user


# router for setting role of user by tg_chat_id
@router.put('/api/v1/user/{tg_chat_id}/role/{role}',
            tags=["user"],
            response_model=UserOutputShema,
            responses={
                200: {
                    "description": 'Success'
                },
                422: {
                    "description": "User not found"
                }
            })
async def set_user_role(
    tg_chat_id: str,
    role: int,
    session: AsyncSession = Depends(get_session)
):
    
    # check if user already exists
    stmt = select(User).where(User.tg_chat_id == tg_chat_id)
    result = await session.execute(stmt)
    user = result.scalars().first()
    if not user:
        return JSONResponse(status_code=422, content='User not found')

    # set user role
    user.role = role
    await session.commit()
    await session.refresh(user)
    return user
