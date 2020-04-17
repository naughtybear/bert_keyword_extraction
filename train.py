'''
訓練bert模型

看看要不要調整loss計算方法
'''
from transformers  import BertForTokenClassification, BertTokenizer, AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from dataset import QADataset
from torch import optim
import torch
import numpy as np


PRETRAINED_MODEL_NAME = "bert-base-chinese"

tag2idx = {"[PAD]": 0, "B": 1, "I": 2, "O": 3}
tags_vals = ["[PAD]", "B", "I", "O"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(batch_size = 4, learning_rate = 0.00002, max_norm = 1.0, epochs = 20, if_validation = False):
    '''
    input:
        batch_size: 數據batch的大小
        learning_rate: 數據學習率
        max_norm: 梯度正規剪裁的數字
        epochs: 迭代數
    output:
        將model參數用torch.save儲存
    '''
    model = BertForTokenClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels = 4) # 輸出的label最多到4
    if torch.cuda.is_available():
        model.cuda()
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    train_dataset = QADataset("train", tokenizer=tokenizer)
    if if_validation == True:
        train_size = int(0.9 * len(train_dataset))
        test_size = len(train_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])
        test_data = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)
    
    train_data = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # 用transformers的optimizer
    optimizer = AdamW(optimizer_grouped_parameters, lr = learning_rate)
    # 使用schedular調整learning rate
    #schedular = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 5, num_training_steps = epochs)

    # 開始訓練
    print("start training")
    max_grad_norm = max_norm
    for epoch in range(epochs):
        # 把model變成train mode
        model.train()
        total_train_loss = 0
        train_steps_per_epoch = 0
        for _, batch in enumerate(train_data):
            questions_ids, mask_ids, key_ids = batch
            # 將data移到gpu
            questions_ids = questions_ids.to(device)
            mask_ids = mask_ids.to(device)
            key_ids = key_ids.to(device)

            # 將optimizer 歸零
            optimizer.zero_grad()
            loss, _ = model(questions_ids, token_type_ids = None, 
                            attention_mask = mask_ids, labels = key_ids)

            loss.backward()

            total_train_loss += loss.item()
            train_steps_per_epoch += 1

            torch.nn.utils.clip_grad_norm_(parameters = model.parameters(), max_norm = max_grad_norm)

            optimizer.step()
            #schedular.step()
            model.zero_grad()

        print(f"Epoch:{epoch}\tTrain Loss: {total_train_loss/train_steps_per_epoch}")
        
        if if_validation == True:
            # calculate the accurancy of validation data
            model.eval()
            predictions = []
            eval_loss, eval_accuracy = 0, 0
            eval_steps_per_epoch, eval_examples_per_epoch = 0, 0
            predictions , true_labels = [], []
            for batch in test_data:
                questions_ids, mask_ids, key_ids = batch
                questions_ids = questions_ids.to(device)
                mask_ids = mask_ids.to(device)
                key_ids = key_ids.to(device)

                with torch.no_grad():
                    tmp_eval_loss = model(questions_ids, token_type_ids=None,
                                        attention_mask=mask_ids, labels=key_ids)
                    logits = model(questions_ids, token_type_ids=None,
                                        attention_mask=mask_ids)
                
                logits = logits[0].detach().cpu().numpy()
                predictions.extend([list(p) for p in np.argmax(logits, axis=2)])

                label_ids = key_ids.to('cpu').numpy()
                true_labels.append(label_ids)
                tmp_eval_accuracy = flat_accuracy(logits, label_ids)

                eval_loss += tmp_eval_loss[0].mean().item()
                eval_accuracy += tmp_eval_accuracy

                eval_examples_per_epoch += questions_ids.size(0)
                eval_steps_per_epoch += 1

            print(f"Validation loss: {eval_loss / eval_steps_per_epoch}")
            print(f"Validation Accuracy: {eval_accuracy / eval_steps_per_epoch}")
        

    torch.save(model, "./pickle/model_v1.pkl")

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
    
if __name__ == "__main__":
    train(batch_size = 4, 
            learning_rate = 0.00002, 
            max_norm = 1.0, 
            epochs = 20)