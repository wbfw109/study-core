from sqlalchemy import create_engine
from sqlalchemy.log import echo_property
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# engine = create_engine('sqlite:////tmp/test.db')
# engine = create_engine(
#     "mysql+mysqlconnector://{user}:{password}@{url}/{database}".format(
#         user="root", password="root", url="localhost:3306", database="my_db"
#     ),
#     echo=True,
# )
engine = create_engine(
    "postgresql+psycopg2://{user}:{password}@{url}/{database}".format(
        user="root", password="root", url="localhost:5432", database="my_db"
    ),
    echo=True,
)

db_session = scoped_session(
    sessionmaker(autocommit=False, autoflush=False, bind=engine)
)
Base = declarative_base()
Base.query = db_session.query_property()


def init_db():
    # import all modules here that might define models so that
    # they will be registered properly on the metadata.  Otherwise
    # you will have to import them first before calling init_db()
    from mysite import models

    # temporary models for command
    from mysite.service.command import creating_log

    Base.metadata.create_all(bind=engine)
