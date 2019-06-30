#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2019-04-17 18:45
@annotation = ''
"""
import pymongo
from bson import ObjectId, SON

mongo_config = {
    'host': 'localhost',
    'port': 27017,
}

db = pymongo.MongoClient(**mongo_config)
# dblist = client.list_database_names()
# print(dblist)
site = db.my.site
inventory = db.my.inventory


def output(cursor):
    for c in cursor:
        print(c)


def insert():
    d = {"name": "RUNOOB", "alexa": "10000", "url": "https://www.runoob.com"}
    res = site.insert_one(d)
    print(res.inserted_id)

    data_list = [
        {"name": "Taobao", "alexa": "100", "url": "https://www.taobao.com"},
        {"name": "QQ", "alexa": "101", "url": "https://www.qq.com"},
        {"name": "Facebook", "alexa": "10", "url": "https://www.facebook.com"},
        {"name": "知乎", "alexa": "103", "url": "https://www.zhihu.com"},
        {"name": "Github", "alexa": "109", "url": "https://www.github.com"}
    ]
    res = site.insert_many(data_list)
    print(res.inserted_ids)


def select():
    # 查询  文档中的第一条数据
    print(site.find_one())

    # 查询集合中所有数据
    for s in site.find():
        print(s)

    # where
    output(site.find_one({'_id': ObjectId('5cb73399efdf11190e096721')}))

    # 嵌入doc
    cursor = inventory.find(
        {"size": SON(
            [("h", 14), ("w", 21), ("uom", "cm")]
        )}
    )

    inventory.find({"size.uom": "in"})

    # and
    output(site.find_one({'name': 'RUNOOB', 'alexa': '100003', }))

    query = {"name": {"$lt": "H"}}
    mydoc = site.find(query)
    for x in mydoc:
        print(x)

    # 字段限制 or select xx .limit(1).skip(2) => limit(2,1)
    query = {"$or": [{'alexa': '101'}, {'alexa': '12345'}]}
    output(site.find(query, {'_id': 0}).limit(1).skip(2))

    # order by  1 为升序排列，而 -1 是用于降序排列。
    for x in site.find().sort([
        ('alexa', pymongo.DESCENDING), ('name', pymongo.ASCENDING)
    ]):
        print(x)
    # in
    query = {'name': {'$in': ['Facebook', 'Taobao']}}
    for x in site.find(query):
        print(x)

    # array

    # 严格 有 顺序一样
    inventory.find({"tags": ["red", "blank"]})
    # 包含这两个元素 不考虑数组中的顺序或其他元素
    inventory.find({"tags": {"$all": ["red", "blank"]}})
    # 包含
    inventory.find({"tags": "red"})
    # 数组位置2
    cursor = db.inventory.find({"dim_cm.1": {"$gt": 25}})
    # 长度
    cursor = db.inventory.find({"tags": {"$size": 3}})

    # 数组嵌入
    cursor = db.inventory.find({'instock.0.qty': {"$lte": 20}})
    cursor = db.inventory.find({"instock.qty": {"$gt": 10, "$lte": 20}})

    # 聚合
    query = [
        # {'$match': {'a': 1}},
        {
            '$group': {
                '_id': "$name",
                'a': {'$sum': 1},
                'b': {'$max': '$alexa'},
            }
        },
        {'$sort': {'a': pymongo.ASCENDING}},
        {'$sort': {'b': pymongo.DESCENDING}},
        {'$limit': 20},
        {'$match': {'a': 1}},

    ]
    # select _id, sum(*) as a, max(alexa) as b  from xx group by name as _id having a=1 order by a asce, b desc
    for x in site.aggregate(query):
        print(x)


def update():
    query = {'alexa': '10880'}
    value = {"$set": {"alexa": "12345"}}
    value = {'$inc': {'age': 3}}
    unvalue = {"$unset": {"name": "12345"}}
    # res = db.update_one(query,value)
    # res = db.update_many(query, value, upsert=True)
    res = site.update_one(query, unvalue)
    print(res)
    site.replace_one({'_id': ObjectId('5cb735b4efdf11198ebb92ac')},
                     {"name": "Taobao12323", "url": "https://www.taobao.com"}, )


def delete():
    site.delete_one({'_id': ObjectId('5cb73399efdf11190e096729')})
    site.delete_many({'_id': ObjectId('5cb73399efdf11190e096729')})


def bulk():
    from pymongo import InsertOne, DeleteOne, ReplaceOne

    requests = [InsertOne({'y': 1}), DeleteOne({'x': 1}),
                ReplaceOne({'w': 1}, {'z': 1}, upsert=True)]
    result = db.my.bulk_write(requests)


bulk()

# inventory.insert_one(
#     {"item": "canvas",
#      "qty": 100,
#      "tags": ["cotton"],
#      "size": {"h": 28, "w": 35.5, "uom": "cm"}})

# inventory.insert_many([
#     {"item": "journal",
#      "qty": 25,
#      "tags": ["blank", "red"],
#      "size": {"h": 14, "w": 21, "uom": "cm"}},
#     {"item": "mat",
#      "qty": 85,
#      "tags": ["gray"],
#      "size": {"h": 27.9, "w": 35.5, "uom": "cm"}},
#     {"item": "mousepad",
#      "qty": 25,
#      "tags": ["gel", "blue"],
#      "size": {"h": 19, "w": 22.85, "uom": "cm"}}])

cursor = inventory.find(
    {"size": SON(
        [("h", 14), ("w", 21), ("uom", "cm")]
    )}
)

inventory.find_one({"size.uom": "in"})
output(inventory.find({"size.uom": "cm"}))
