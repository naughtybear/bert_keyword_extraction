'''
可以即時測試的test檔

***要處理輸入小於max len 情況***
'''
from transformers import BertTokenizer
import torch
import numpy as np
import pickle
import re

device = torch.device("cuda:0")
PRETRAINED_MODEL_NAME = "bert-base-chinese"
# PRETRAINED_MODEL_NAME = "bert-base-multilingual-cased"

def test():
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    (_,
     tokenized_question,
     labeled_key,
     max_len) = pickle.load(open('D:/python/keyword_extraction/pickle/train_data.pkl', 'rb'))
    model = torch.load("D:/python/keyword_extraction/pickle/model_v3.pkl")
    model.cuda()
    model.eval()

    with torch.no_grad():
        while True:
            sentence = input("")
            if sentence == "quit":
                break
            process_question = re.sub("\\s+", "，", sentence)
            # 刪除多餘換行
            process_question = re.sub("(\r\n)+", "，", process_question)
            process_question = re.sub("\n+", "，", process_question)
            process_question = re.sub("，+", "，", process_question)
            # 刪除括號內英文
            process_question = re.sub("\\([a-z A-Z \\-]*\\)", "", process_question)
            if len(process_question) > max_len:
                print("The sentence is too long")
                continue

            # 將輸入加入cls並padding到max len
            sentence = ["[CLS]"] + tokenizer.tokenize(process_question)
            sentence = sentence + ["[PAD]"] * (max_len - len(sentence))

            # print(sentence)

            # 將輸入斷詞並產生mask
            sentence_ids = tokenizer.convert_tokens_to_ids(sentence)
            mask = [float(i > 0) for i in sentence_ids]
            
            sentence_ids = torch.tensor(
                [sentence_ids], dtype=torch.long).to(device)
            mask = torch.tensor([mask], dtype=torch.long).to(device)
            '''
            sentence_ids = torch.tensor(
                [sentence_ids], dtype=torch.long)
            mask = torch.tensor([mask], dtype=torch.long)
            '''

            output = model(
                sentence_ids, token_type_ids=None, attention_mask=mask)
            output = output[0].detach().cpu().numpy()
            prediction = [list(p) for p in np.argmax(output, axis=2)]

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
                print("no keyword")
                continue

            if key_word[0] == "，":
                key_word.remove("，")

            key_word = [x for x in key_word if x != "[PAD]"]

            flag = True
            while True:
                if(len(key_word) == 0):
                    print("no keyword")
                    flag = False
                    break
                if key_word[-1] == '，':
                    key_word.pop()
                else:
                    break

            if flag is False:
                continue

            print("".join(key_word))


if __name__ == "__main__":
    test()
