# 中文關鍵字提取

用google的bert_chinese做fine-tune，將病人線上提問的問題進行關鍵字提取

## 資料集
資料都存在 data 這個資料夾中，下方為各個檔案的簡介:
* full_data_v1.csv : 原始標記的資料，約700筆
* train_data_v1.csv : 根據full_data_v1分出來的訓練資料
* test_data_v1.csv : 根據full_data_v1分出來的測試資料
* compare?.csv : test data的預測結果與正確結果的比較，問號的數字代表對應幾號model

## pickle檔
pickle這個資料夾存放資料前處理後的檔案及模型，下方為各個檔案的簡介:
* train_data.pkl : 訓練資料經過前處理後的檔案
* test_data.pkl : 測試資料經過前處理後的檔案
* model_v1.pkl : learning rate:0.00002, epochs:30, no schedule, batch size:12
* model_v2.pkl : learning rate:0.00002, epochs:20, no schedule, batch size:4 
* model_v3.pkl : 0.00002, epochs:30, no schedule, batch size:8   X
* model_v4.pkl : 0.00002, epochs:40, no schedule, batch size:24 

## 執行方法
### 預處理:
可以到程式碼內選擇要預處理訓練資料或是測試資料

    python preprocess.py
### 訓練:
    python train.py
### 測試:
用測試資料進行測試

    python test.py
自行輸入測資測試

    python real_time_test.py

