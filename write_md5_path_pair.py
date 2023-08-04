from get_file_path import get_file_path
from all_md5 import get_all_md5

path = r'./.assets'
file_list = []
dir_list = []
get_file_path(path, file_list, dir_list)

all_md5 = get_all_md5(file_list)

from sqlalchemy import create_engine, Column, Integer, String, Float, BigInteger, ARRAY, ForeignKey, UniqueConstraint, PrimaryKeyConstraint,VARCHAR,select
from sqlalchemy.orm import declarative_base, mapped_column
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector
import yaml

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

DATABASE_URL = config['db']['dburl']

engine = create_engine(DATABASE_URL, echo=True)

Base = declarative_base()

class Md5Path(Base):
	__tablename__ = 'md5_path'

	id = mapped_column(Integer, primary_key=True)
	md5_hash = mapped_column(VARCHAR, nullable=False)
	file_path = mapped_column(VARCHAR, nullable=False)

Session = sessionmaker(bind=engine)
session = Session()

for i in range(len(all_md5)):
	md5_path = Md5Path(md5_hash=all_md5[i], file_path=file_list[i])
	session.add(md5_path)
	session.commit()

session.close()