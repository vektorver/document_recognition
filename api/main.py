from fastapi import FastAPI
from api.v1.routers import users, passports

# from config import engine, session
from config import get_session, init_db
from api.v1.models.users import Base1
from api.v1.models.passports import Base2


app = FastAPI()

@app.on_event("startup")
async def on_startup():
    await init_db()

app.include_router(users.router)
app.include_router(passports.router)

