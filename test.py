'''
輸出test data 的結果
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

def test(batch_size = 4):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    test_dataset = QADataset("test", tokenizer=tokenizer)
    test_data_loader = DataLoader(test_dataset, batch_size = batch_size)
    print(test_dataset)

    model = torch.load("./pickle/model.pkl")
    model.cuda()
    model.eval()

    # 顯示test data
    predictions = []
    key_words = []
    with torch.no_grad():
        for test_data in test_data_loader:
            questions_ids, mask_ids, tokenized_question = test_data
            questions_ids = questions_ids.to(device)
            mask_ids = mask_ids.to(device)

            output =  model(questions_ids, token_type_ids=None, attention_mask=mask_ids)
            output = output[0].detach().cpu().numpy()
            prediction = [list(p) for p in np.argmax(output, axis=2)]
            predictions.extend(prediction)
            
            for i in range(len(prediction)):
                key_word = []
                for j in range(len(prediction[i])):
                    if prediction[i][j] == 1:
                        key_word.append("，")
                        key_word.append(tokenized_question[j][i])

                    elif prediction[i][j] == 2:
                        key_word.append(tokenized_question[j][i])
                    key_words.append(key_word)
                print(key_word)

if __name__ == "__main__":
    test()