'''
將test檔案的預測結果輸出
'''
from transformers import BertTokenizer
import torch
import numpy as np
import pickle
import re
import pandas as pd
import csv
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test():
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    (_,
     tokenized_question,
     labeled_key,
     max_len) = pickle.load(open('./pickle/train_data.pkl', 'rb'))

    csvfile = open("./data/out99.csv", "w", newline="", encoding="utf-8")
    csvwriter = csv.writer(csvfile)

    df_test = pd.read_csv("./data/test_data_v3.csv")
    df_test = df_test.dropna()
    questions = list(df_test["Body"])

    model = torch.load("./pickle/model_v5.pkl")
    model.cuda()
    model.eval()

    with torch.no_grad():
        for question in tqdm(questions):
            # print(sentence)
            process_question = re.sub("\\s+", "", question)
            # 刪除多餘換行
            process_question = re.sub("\r*\n*", "", process_question)
            # 刪除括號內英文
            process_question = re.sub(
                "\\([a-z A-Z \\-]*\\)", "", process_question)
            if len(process_question) > max_len:
                continue

            # 將輸入加入cls並padding到max len
            sentence = ["[CLS]"] + tokenizer.tokenize(process_question)
            sentence = sentence + ["[PAD]"] * (max_len - len(sentence))

            # 將輸入斷詞並產生mask
            sentence_ids = tokenizer.convert_tokens_to_ids(sentence)
            mask = [float(i > 0) for i in sentence_ids]
            sentence_ids = torch.tensor(
                [sentence_ids], dtype=torch.long).to(device)
            mask = torch.tensor([mask], dtype=torch.long).to(device)

            output = model(
                sentence_ids, token_type_ids=None, attention_mask=mask)
            output = output[0].detach().cpu().numpy()
            prediction = [list(p) for p in np.argmax(output, axis=2)]

            for i in range(len(sentence)):
                sentence[i] = re.sub("##", "", sentence[i])

            # 比對label找出關鍵字
            key_word = []
            unk_count = 0
            for i in range(len(prediction[0])):
                if unk_count != 0 and sentence[i] != "[UNK]":
                    start = (process_question.find(
                        sentence[i-unk_count-1], i-unk_count-2)
                        + len(sentence[i-unk_count-1]))

                    end = process_question.find(sentence[i], i-1)
                    # print(process_question[start : end])
                    key_word.append(process_question[start: end])
                    unk_count = 0

                elif unk_count != 0 and sentence[i] == "[UNK]":
                    unk_count = unk_count + 1
                    continue

                if prediction[0][i] == 1:
                    key_word.append("，")
                    if sentence[i] == "[UNK]":
                        unk_count = unk_count + 1
                    else:
                        key_word.append(sentence[i])

                elif prediction[0][i] == 2:
                    if sentence[i] == "[UNK]":
                        unk_count = unk_count + 1
                    else:
                        key_word.append(sentence[i])

            # 刪除padding和不必要的符號
            if(len(key_word) == 0):
                csvwriter.writerow([question, "[]"])
                continue

            if key_word[0] == "，":
                key_word.remove("，")

            key_word = [x for x in key_word if x != "[PAD]"]

            flag = True
            while True:
                if(len(key_word) == 0):
                    csvwriter.writerow([question, "[]"])
                    flag = False
                    break
                if key_word[-1] == '，':
                    key_word.pop()
                else:
                    break

            if flag is False:
                continue

            key_word = "".join(key_word)
            csvwriter.writerow([question, key_word])


if __name__ == "__main__":
    test()
