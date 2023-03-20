import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="test",
  password="test",
  database="forensics"
)

mycursor = mydb.cursor()

def clear_node(table_name="blockchain_0"):
    sql_drop = "DROP TABLE IF EXISTS {}".format(table_name)
    sql_create = "CREATE TABLE {} (term INTEGER, height INTEGER, pt INTEGER, t VARCHAR(100))".format(table_name)
    sql_insert = "INSERT INTO {} (term, height, pt, t) values (%s, %s, %s, %s)".format(table_name)
    val = (-1, -1, -1, "tx")
    mycursor.execute(sql_drop)
    mycursor.execute(sql_create)
    #mycursor.execute(sql_insert, val)
    mydb.commit()

def clear_cc(table_name="cc_0"):
    sql_drop = "DROP TABLE IF EXISTS {}".format(table_name)
    sql_create = "CREATE TABLE {} (term INTEGER, height INTEGER, voters VARCHAR(100))".format(table_name)
    sql_insert = "INSERT INTO {} (term, height, voters) values (%s, %s, %s)".format(table_name)
    val = (-1, -1, "[]")
    mycursor.execute(sql_drop)
    mycursor.execute(sql_create)
    #mycursor.execute(sql_insert, val)
    mydb.commit()


def clear_lc(table_name="cc_0"):
    sql_drop = "DROP TABLE IF EXISTS {}".format(table_name)
    sql_create = "CREATE TABLE {} (term INTEGER, leader INTEGER, voters VARCHAR(100))".format(table_name)
    sql_insert = "INSERT INTO {} (term, leader, voters) values (%s, %s, %s)".format(table_name)
    val = (-1, -1, "[]")
    mycursor.execute(sql_drop)
    mycursor.execute(sql_create)
    #mycursor.execute(sql_insert, val)
    mydb.commit()

def clear_conflict():
    sql_drop = "DROP TABLE IF EXISTS conflict"
    sql_create = "CREATE TABLE conflict (time TIMESTAMP, term INTEGER, height INTEGER, diff INTEGER)"
    mycursor.execute(sql_drop)
    mycursor.execute(sql_create)
    mydb.commit()  

def clear_images(img_name=1):
    sql_drop = "DROP TABLE IF EXISTS images"
    sql_create = "CREATE TABLE images (normal INTEGER)"
    sql_insert = "INSERT INTO images (normal) values (%s)"
    val = (img_name,)
    mycursor.execute(sql_drop)
    mycursor.execute(sql_create)
    mycursor.execute(sql_insert, val)
    mydb.commit()


def clear_text():
    sql_drop = "DROP TABLE IF EXISTS text"
    sql_create = "CREATE TABLE text (time TIMESTAMP, content VARCHAR(1024))"
    # init the entry with default value
    mycursor.execute(sql_drop)
    mycursor.execute(sql_create)
    mydb.commit()

def insert_node(table_name, params):
    sql = "INSERT INTO {} (term, height, pt, t) VALUES (%s, %s, %s, %s)".format(table_name)
    val = params
    mycursor.execute(sql, val)
    mydb.commit()

def insert_cc(table_name, params):
    sql = "INSERT INTO {} (term, height, voters) VALUES (%s, %s, %s)".format(table_name)
    val = params
    mycursor.execute(sql, val)
    mydb.commit()

def insert_lc(table_name, params):
    sql = "INSERT INTO {} (term, leader, voters) VALUES (%s, %s, %s)".format(table_name)
    val = params
    mycursor.execute(sql, val)
    mydb.commit()

def insert_conflict(params):
    sql = "INSERT INTO conflict (time, term, height, diff) VALUES (%s, %s, %s, %s)"
    val = params
    mycursor.execute(sql, val)
    mydb.commit()

def insert_text(params):
    sql = "INSERT INTO text (time, content) VALUES (%s, %s)"
    val = params
    mycursor.execute(sql, val)
    mydb.commit()

def delete_node(table_name, x=3):
    sql = "DELETE FROM {} WHERE round > -1 limit %s".format(table_name)
    val = (x,)
    mycursor.execute(sql, val)
    mydb.commit()

