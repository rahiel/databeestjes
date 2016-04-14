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

def parse_course(data, t, f):
    true_list = ["Yes", "yes", "y","Y", t]
    false_list = ["No", "no", "n","N", f]
    pData = []
    for d in range(len(data)):
        if (data[d] in true_list):
            pData.append(True)
        elif (data[d] in false_list):
            pData.append(False)
        else:
            pData.append(None)
    return pData


data = load_data()
(programme, ml, ir, st, db, gender, chocolate, birthday, neighbours, stand,
 stress, money, random, bedtime, good_day1, good_day2) = zip(*data)


mlp = parse_course(ml,"y","n")
irp = parse_course(ir,"1","0")
stp = parse_course(st,"mu","sigma")
dbp = parse_course(db,"j","n")
genderp = parse_course(gender,"m","f")
