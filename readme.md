# 中文關鍵字提取

用google的bert_chinese做fine-tune，將病人線上提問的問題進行關鍵字提取

## 資料集
資料都存在 data 這個資料夾中，下方為各個檔案的簡介:
* full_data_v1.csv : 原始標記的資料，約700筆
* 其他較新的資料集因為資料來源關係無法提供


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

