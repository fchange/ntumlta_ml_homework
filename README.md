# ntumlta_ml_homework
Homework implementation of ntumlta 2019
## HW1

本次作業使用豐原站的觀測記錄，分成train set跟test set。train set是豐原站每個月的前20天所有資料。test set則是從豐原站剩下的資料中取樣出來。train.csv: 每個月前20天的完整資料。test.csv : 從剩下的資料當中取樣出連續的10小時為一筆，前九小時的所有觀測數據當作feature，第十小時的PM2.5當作answer。一共取出240筆不重複的test data，請根據feauure預測這240筆的PM2.5。Data含有18項觀測數據 AMB_TEMP, CH4, CO, NHMC, NO, NO2, NOx, O3, PM10, PM2.5, RAINFALL, RH, SO2, THC, WD_HR, WIND_DIREC, WIND_SPEED, WS_HR。 

作业链接： https://ntumlta2019.github.io/ml-web-hw1/


## HW2
- Task: **Binary Classification**
  Determine whether a person makes over 50K a year.
- Dataset: **ADULT**
  Extraction was done by Barry Becker from the 1994 Census database. A set of reasonably clean records was extracted using the following conditions: ((AGE>16) && (AGI>100) && (AFNLWGT>1) && (HRSWK>0)).
- Reference: https://archive.ics.uci.edu/ml/datasets/Adult
- 作业链接：https://ntumlta2019.github.io/ml-web-hw2/



## ATTENTION

需要注意的一点是：
HW1的教学代码编程思想，是面向过程

hw2_generative的代码编程思想，是面向对象

hw2_logistic的代码编程思想，是函数式编程