from sqlalchemy import select
from typing import List
from fastapi.responses import JSONResponse
from api.v1.schemas.passports import PassportOutputShema, PassportInputShema
from api.v1.models.passports import Passport
from config import get_session
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
import os
import pandas as pd
import numpy as np
from PIL import Image
from api.model.segmentation import pipeline_predictor

from fastapi import APIRouter, Depends, File, UploadFile


router = APIRouter()
from sqlalchemy.sql import exists

# create new passport
@router.post('/api/v1/passport/{tg_chat_id}',
                tags=["passport"],
                response_model=PassportOutputShema,
                responses={
                    200: {
                        "description": 'Success'
                    },
                    422: {
                        "description": "Passport already exists"
                    }
                })
async def register_new_passport(
    tg_chat_id: str,
    passport: PassportInputShema,
    session: AsyncSession = Depends(get_session)
):
    # create new passport
    new_passport = Passport(
        tg_chat_id=str(tg_chat_id),
        name=passport.name,
        surname=passport.surname,
        patronymic=passport.patronymic,
        birth_date=passport.birth_date,
        birth_place=passport.birth_place,
        passport_series=passport.passport_series,
        passport_number=passport.passport_number,
        passport_date=passport.passport_date,
        passport_code=passport.passport_code,
        passport_issued_by=passport.passport_issued_by,
        passport_issued_by_code=passport.passport_issued_by_code
    )
    session.add(new_passport)
    await session.commit()
    await session.refresh(new_passport)
    return new_passport



# get all passports by tg_chat_id
@router.get('/api/v1/passports/{tg_chat_id}',
            tags=["passport"],
            response_model=List[PassportOutputShema],
            responses={
                200: {
                    "description": 'Success'
                }
            })
async def get_passports_by_tg_chat_id(
    tg_chat_id: str,
    session: AsyncSession = Depends(get_session)
):
    stmt = select(Passport).where(Passport.tg_chat_id == tg_chat_id)
    result = await session.execute(stmt)
    passports = result.scalars().all()
    return passports


# clear all passports by tg_chat_id
@router.delete('/api/v1/passports/{tg_chat_id}',
            tags=["passport"],
            response_model=List[PassportOutputShema],
            responses={
                200: {
                    "description": 'Success'
                }
            })
async def clear_passports_by_tg_chat_id(
    tg_chat_id: str,
    session: AsyncSession = Depends(get_session)
):
    stmt = select(Passport).where(Passport.tg_chat_id == tg_chat_id)
    result = await session.execute(stmt)
    passports = result.scalars().all()
    for passport in passports:
        session.delete(passport)
    await session.commit()
    return passports


# clear all passports at all
@router.delete('/api/v1/passports',
            tags=["passport"],
            response_model=List[PassportOutputShema],
            responses={
                200: {
                    "description": 'Success'
                }
            })
async def clear_all_passports(
    session: AsyncSession = Depends(get_session)
):
    stmt = select(Passport)
    result = await session.execute(stmt)
    passports = result.scalars().all()
    for passport in passports:
        session.delete(passport)
    await session.commit()
    return passports


# get predictions from image
@router.post('/api/v1/passport/predict/{tg_chat_id}',
            tags=["passport"],
            response_model=PassportOutputShema,
            responses={
                200: {
                    "description": 'Success'
                }
            })
async def predict_passport(
    tg_chat_id: str,
    file: UploadFile,
    session: AsyncSession = Depends(get_session)
):
    # read image with pillow
    image = Image.open(file.file)
    # convert image to numpy array
    image = np.array(image)

    vis, predictions, _ = pipeline_predictor(image)

    # save numpy.ndarray as uploads/result.png
    Image.fromarray(vis).save("/uploads/result.png")

    # predict passport
    # create some fake passport output

    # predictions keys "organization,getDate,code,sex,city,secondName,firstName,thirdName,birthDate,series1,series2"
    passport = PassportOutputShema(
        id=1,
        tg_chat_id=tg_chat_id,
        name=predictions["firstName"],
        surname=predictions["secondName"],
        patronymic=predictions["thirdName"],
        birth_date=predictions["birthDate"],
        birth_place=predictions["city"],
        passport_series=predictions["series1"][:4],
        passport_number=predictions["series1"][4:],
        passport_date=predictions["getDate"],
        passport_code=predictions["code"],
        passport_issued_by=predictions["organization"],
        passport_issued_by_code=predictions["code"]
    )

    # create new passport
    new_passport = Passport(
        tg_chat_id=str(tg_chat_id),
        name=passport.name,
        surname=passport.surname,
        patronymic=passport.patronymic,
        birth_date=passport.birth_date,
        birth_place=passport.birth_place,
        passport_series=passport.passport_series,
        passport_number=passport.passport_number,
        passport_date=passport.passport_date,
        passport_code=passport.passport_code,
        passport_issued_by=passport.passport_issued_by,
        passport_issued_by_code=passport.passport_issued_by_code
    )

    session.add(new_passport)
    await session.commit()
    await session.refresh(new_passport)

    return passport


# get all passports by tg_chat_id as excel file
@router.get('/api/v1/passports/excel/{tg_chat_id}',
            tags=["passport"],
            responses={
                200: {
                    "description": 'Success'
                }, 
                404: {
                    "description": "Passports not found"
                }
            })
async def get_excel_passports_by_tg_chat_id(
    tg_chat_id: str,
    session: AsyncSession = Depends(get_session)
):
    stmt = select(Passport).where(Passport.tg_chat_id == tg_chat_id)
    passports = await session.execute(stmt)
    
    if passports:
        df = pd.DataFrame([passport.__dict__ for passport in passports.scalars().all()])
        filename = '/uploads/' + \
            "passports_{}.csv".format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
        df.to_csv(filename, encoding='utf-8-sig')
        return {'filename': filename}
    else:
        return JSONResponse(status_code=404, content='Passports not found')