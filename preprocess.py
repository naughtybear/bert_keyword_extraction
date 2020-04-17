'''
1.要處理標點符號 火星文
2.把關鍵字標記

***要處理輸入小於max len 情況***
'''
import pandas as pd
import re
import pickle
from transformers  import BertTokenizer #bert斷詞，以字為單位

def preprocess(tokenizer, mode, max_len = -1):
    assert mode in ["train", "test"]
    df_train = pd.read_csv(f"./data/{mode}_data_v1.csv")
    questions = list(df_train["QuestionBody"])
    keys = list(df_train["關鍵字"])
    data_len = len(df_train)
    labeled_key = []
    tokenized_question = []
    if max_len == -1:
        max_len = max([len(x) + 1 for x in questions])
    else:
        max_len = max_len

    # 將關鍵字label
    for i in range(data_len):
        print(f"num:{i}")
        # 統一逗號
        #print(keys[i])
        keys[i] = re.sub(",", "，", keys[i])
        # 消除多餘空白
        keys[i] = re.sub(" ", "", keys[i])
        questions[i] = re.sub("\\s+", "", questions[i])
        # 刪除多餘換行
        questions[i] = re.sub("\r*\n*", "", questions[i])
        # 刪除括號內英文
        questions[i] = re.sub("\\([a-z A-Z \\-]*\\)", "", questions[i])
        # 把每個關鍵字都切出來
        key_token = keys[i].split('，')
        
        # 將關鍵字label
        last = 1
        search_start = 1
        label = ["[PAD]"]
        question_token = ["[CLS]"] + tokenizer.tokenize(questions[i])
        for token in key_token:
            token = tokenizer.tokenize(token)
            flag = True
            while flag:
                try:
                    position_start = question_token.index(token[0], last)
                except ValueError:
                    print("error")
                    break
                if search_start < position_start:
                    search_start = position_start
                #print(position_start)
                #print(question_token[position_start])
                #print(token[-1])
                try:
                    position_end = question_token.index(token[-1], search_start)
                except ValueError:
                    print("error")
                    break
                #print(position_end)
                #print(f"minus:{position_end - position_start + 1}")
                #print(f"len:{len(token)}")
                #print("-------")
                if position_end - position_start + 1 == len(token):
                    label = label + (position_start - last) * ["O"] + ["B"] + (position_end - position_start) * ["I"]
                    last = position_end + 1
                    flag = False
                elif position_end - position_start + 1 > len(token):
                    label = label + (position_start - last + 1) * ["O"]
                    last = position_start + 1
                elif position_end - position_start + 1 < len(token):
                    search_start = position_end + 1


        label = label + (len(question_token) - last) * ["O"]
        #print(f"label: {len(label)}   ques_token:{len(question_token)}")
        assert len(label) == len(question_token)

        label = label + (max_len - len(label)) * ["[PAD]"]
        question_token = question_token + (max_len - len(question_token)) * ["[PAD]"]

        assert len(label) == len(question_token) == max_len


        labeled_key.append(label)
        tokenized_question.append(question_token)
        # print(f"{type(label)}  {type(question_token)}")
    
    # print(tokenized_question)

    with open(f"./pickle/{mode}_data.pkl", "wb") as file:
        pickle.dump((tokenized_question, labeled_key, max_len), file)

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    # preprocess(tokenizer = tokenizer, mode = "train")
    preprocess(tokenizer = tokenizer, mode = "test")