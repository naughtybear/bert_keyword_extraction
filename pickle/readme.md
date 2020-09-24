# pickle檔介紹
pickle這個資料夾存放資料前處理後的檔案及模型，下方為各個檔案的簡介:
* train_data.pkl : 訓練資料經過前處理後的檔案
* test_data.pkl : 測試資料經過前處理後的檔案
* model_v1.pkl : 0.00002, epochs:4, schedule, batch size:16  base
* model_v2.pkl : 0.00002, epochs:4, schedule, batch size:16  新增些許骨科及牙科資料，刪除2000句5914
* model_v3.pkl : 0.00002, epochs:6, schedule, batch size:12  刪除約4000句5914
* model_v4.pkl : 0.00002, epochs:6, schedule, batch size:10  改用bert-base-multilingual-cased 效果不好
* model_v5.pkl : 0.00002, epochs:3, schedule, batch size:6  使用chinese-bert-wwm-ext

[模型下載連結](https://drive.google.com/drive/folders/1bnl5MpsGUEs70Tl1X6gy4z-P2H5PBuCE?usp=sharing)