# G-SDD_For_RS
#dataset
python dictionary  from label to int 
label_map.py
1.data struct for examle and bounding box 
2.io api and parse label_txt which is encoded by utf-8
Note that original label_txt is encoded by ucs-2 little endian,I transform it to utf-8 by manual.
yanshen_reader.py

#statistic:
some statistics based on API of yanshen_reader.py
1. num_of_object_per_category 
num.py

#app
the remote sensing image object detection system for yanshen cup.
#object_detection
Google tensorflow object detection API
#slim
Google tensorflow image classification API

