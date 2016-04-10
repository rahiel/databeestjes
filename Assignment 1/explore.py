#!/usr/bin/env python3
import csv


def load_data():
    with open("ODI-2016.csv") as f:
        fieldnames = f.readline().split(';')
        fieldnames = [x.split('-')[1].strip() for x in fieldnames]
        csv_reader = csv.reader(f, delimiter=';')
        data = []
        for p in csv_reader:
            data.append(p)
    data = data[:-1]            # excluding empty line
    return data

data = load_data()
(programme, ml, ir, st, db, gender, chocolate, birthday, neighbours, stand,
 stress, money, random, bedtime, good_day1, good_day2) = zip(*data)
