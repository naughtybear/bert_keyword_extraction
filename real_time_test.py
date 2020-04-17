'''
可以即時測試的test檔

***要處理輸入小於max len 情況***
'''
from transformers  import BertForTokenClassification, BertTokenizer
from torch.utils.data import DataLoader
from dataset import QADataset
from torch import optim
import torch
import numpy as np
import pickle
import re

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tag2idx = {"[PAD]": 0, "B": 1, "I": 2, "O": 3}
tags_vals = ["[PAD]", "B", "I", "O"]

def test():
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    tokenized_question, labeled_key, max_len = pickle.load(open('./pickle/train_data.pkl', 'rb'))
    model = torch.load("./pickle/model_v1.pkl")
    model.cuda()
    model.eval()

    with torch.no_grad():
        while True:
            sentence = input(">>")
            if sentence == "quit":
                break
            sentence = re.sub("\\s+", "", sentence)
            # 刪除多餘換行
            sentence = re.sub("\r*\n*", "", sentence)
            # 刪除括號內英文
            sentence = re.sub("\\([a-z A-Z \\-]*\\)", "", sentence)

            # 將輸入加入cls並padding到max len
            sentence = ["[CLS]"] + tokenizer.tokenize(sentence)
            sentence = sentence + ["[PAD]"] * (max_len - len(sentence))

            # 將輸入斷詞並產生mask
            sentence_ids = tokenizer.convert_tokens_to_ids(sentence)
            mask = [float(i>0) for i in sentence_ids]
            sentence_ids = torch.tensor([sentence_ids], dtype = torch.long).to(device)
            mask = torch.tensor([mask], dtype = torch.long).to(device)

            output =  model(sentence_ids, token_type_ids=None, attention_mask=mask)
            output = output[0].detach().cpu().numpy()
            prediction = [list(p) for p in np.argmax(output, axis=2)]

            # 比對label找出關鍵字
            key_word = []
            for i in range(len(prediction[0])):
                if prediction[0][i] == 1:
                    key_word.append("，")
                    key_word.append(sentence[i])

                elif prediction[0][i] == 2:
                    key_word.append(sentence[i])
            
            
            # 刪除padding和不必要的符號
            if(len(key_word) == 0):
                    print("[]")
                    continue
                
            if key_word[0] == "，":
                key_word.remove("，")
            
            key_word = [x for x in key_word if x != "[PAD]"]
            
            while True:
                if key_word[-1] == '，':
                    key_word.pop()
                else:
                    break
            
            print("".join(key_word))

if __name__ == "__main__":
    test()