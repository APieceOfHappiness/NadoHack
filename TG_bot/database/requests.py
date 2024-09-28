import sqlite3
from functools import wraps

DATA_PATH = "database/database.db"


def db_request(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        connect = sqlite3.connect(DATA_PATH)
        cursor = connect.cursor()
        try:
            result = func(cursor, *args, **kwargs)
            return result
        finally:
            connect.close()

    return wrapper


def db_update(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        connect = sqlite3.connect(DATA_PATH)
        cursor = connect.cursor()
        try:
            result = func(cursor, *args, **kwargs)
            connect.commit()
            return result
        finally:
            connect.close()

    return wrapper


@db_update
def add_to_db(cursor, *args, **kwargs):

    question = kwargs.get('question')
    class1 = kwargs.get('class1')
    class2 = kwargs.get('class2')
    answer = kwargs.get('answer')

    cursor.execute("""INSERT INTO questions(question, class1, class2, answer) VALUES(?, ?, ?, ?)""",
                   (question, class1, class2, answer))
