"""
負責對訓練資料進行預處理:
    1.統一標點符號，刪除空白及換行
    2.把關鍵字標記
"""
import pandas as pd
import re
import pickle
from transformers import BertTokenizer  # bert斷詞，以字為單位
from tqdm import tqdm


def preprocess(tokenizer, data_version, max_len=-1):
    """
    input:
        tokenizer: 要用哪種tokenizer，目前都是用bert的
        max_len: 句子的最大長度， 若是為-1，則會用train data中的最長長度為max_len
        data_version: 訓練資料的版本
    output:
        將下方四項存為pickle檔
        tokenized_question: 斷詞後的問題
        labeled_key: 把關鍵字轉換成label("O":非關鍵字 "B":關鍵字開始 "I":關鍵字內容)
        max_len: 最長長度，最大為512，因為bert模型最大只能輸入512
    """
    df_train = pd.read_csv(f"./data/train_data_v{data_version}.csv")
    questions = list(df_train["QuestionBody"])
    keys = list(df_train["關鍵字"])
    data_len = len(df_train)
    labeled_key = []
    tokenized_question = []
    full_question = []
    if max_len == -1:
        max_len = max([len(x) + 1 for x in questions])

    # 將關鍵字label
    for i in tqdm(range(data_len)):
        # print(f"num:{keys[i]}")
        # 統一逗號
        keys[i] = re.sub(",", "，", keys[i])
        # 刪除多餘換行
        questions[i] = re.sub("[\\r\\n]+", "，", questions[i])
        questions[i] = re.sub("\\n+", "，", questions[i])
        # 消除多餘空白
        keys[i] = re.sub(" ", "", keys[i])
        questions[i] = re.sub("\\s+", "", questions[i])
        # 刪除括號內英文
        questions[i] = re.sub("\\([a-z A-Z \\-]*\\)", "", questions[i])
        questions[i] = re.sub("，+", "，", questions[i])
        questions[i] = re.sub("？", "?", questions[i])
        # 把每個關鍵字都切出來
        key_token = keys[i].split("，")

        if len(questions[i]) > max_len:
            continue

        full_question.append(questions[i])
        # 將關鍵字label
        last = 1
        search_start = 1
        label = ["O"]
        question_token = ["[CLS]"] + tokenizer.tokenize(questions[i]) + ["[SEP]"]
        for token in key_token:
            token = tokenizer.tokenize(token)
            flag = True
            while flag:
                # 如果關鍵字不是在問題中， 這裡就會報錯， 讓它顯示error的位置並繼續跑以利修改
                try:
                    position_start = question_token.index(token[0], last)
                except ValueError:
                    print(f"num:{i}")
                    print(questions[i])
                    print(question_token)
                    print(token)
                    print("error end")
                    break

                if search_start < position_start:
                    search_start = position_start

                try:
                    position_end = question_token.index(token[-1], search_start)
                except ValueError:
                    print(f"num:{i}")
                    print(questions[i])
                    print(question_token)
                    print(token)
                    print("error end")
                    break

                # 將問題用關鍵字label
                if position_end - position_start + 1 == len(token):
                    label = (
                        label
                        + (position_start - last) * ["O"]
                        + ["B"]
                        + (position_end - position_start) * ["I"]
                    )
                    last = position_end + 1
                    flag = False
                elif position_end - position_start + 1 > len(token):
                    label = label + (position_start - last + 1) * ["O"]
                    last = position_start + 1
                elif position_end - position_start + 1 < len(token):
                    search_start = position_end + 1

        # 把最後沒有文字的地方補零
        label = label + (len(question_token) - last) * ["O"]
        # print(label)
        # print(question_token)
        # input()
        # print(f"label: {len(label)}   ques_token:{len(question_token)}")
        # print(f"{len(label)}, {len(question_token)}")
        assert len(label) == len(question_token)

        label = label + (max_len - len(label)) * ["O"]
        question_token = question_token + (max_len - len(question_token)) * ["[PAD]"]

        assert len(label) == len(question_token) == max_len

        labeled_key.append(label)
        tokenized_question.append(question_token)

    with open("./pickle/train_data.pkl", "wb") as file:
        pickle.dump((full_question, tokenized_question, labeled_key), file)


if __name__ == "__main__":
    # tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
    preprocess(tokenizer=tokenizer, data_version=8, max_len=512)
