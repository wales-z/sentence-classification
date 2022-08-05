import numpy as np
import pandas as pd
import os
import torch.nn as nn
import torch
import datetime
import logging
import time
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report, accuracy_score
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from typing import List
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, SequentialSampler


# random seed
def set_seed(seed):
    random.seed(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# customized logger
def custom_logger(name: str, log_level: str):

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False
    logger.handlers = []
    #  自定义log 格式
    # formatter = logging.Formatter('[%(asctime)s]-[%(name)s:%(levelname)s]-[%(process)d-%(thread)d]:%(message)s')
    formatter = logging.Formatter('[%(asctime)s]-[%(name)s:%(levelname)s]: %(message)s')
    #  使用utc-时间
    def _utc8_aera(timestamp):
        now = datetime.datetime.utcfromtimestamp(timestamp) + datetime.timedelta(hours=8)
        return now.timetuple()

    formatter.converter = _utc8_aera
    custom_handler = logging.StreamHandler()
    custom_handler .setFormatter(formatter)

    logger.addHandler(custom_handler)

    return logger


# tokenize
def tokenize_corpus(text_list: List[str], tokenizer: BertTokenizer) -> List[List[int]]:
    tokenized_data = []
    for text in tqdm(text_list):
        token = tokenizer.tokenize(text)
        token = ["[CLS]"] + token
        token = tokenizer.convert_tokens_to_ids(token)
        tokenized_data.append(token)
    return tokenized_data


def mask_token_ids(token_ids: List[List[int]], padded_size):
    input_ids = []
    input_types = []
    input_masks = []

    for token in tqdm(token_ids):
        types = [0] * (len(token))
        masks = [1] * (len(token))

        # pad
        if len(token) < padded_size:
            types = types + [1] * (padded_size - len(token))
            masks = masks + [0] * (padded_size - len(token))
            token = token + [0] * (padded_size - len(token))
        else:
            types = types[:padded_size]
            masks = masks[:padded_size]
            token = token[:padded_size]

        assert len(token) == len(masks) == len(types) == padded_size

        input_ids.append(token)
        input_types.append(types)
        input_masks.append(masks)

    return input_ids, input_types, input_masks


# train test split
def split_train_dataset(input_ids: List[List[int]], input_types: List[List[int]],
                        input_masks: List[List[int]], labels: List[List[int]], batch_size: int, ratio: float) -> (
        DataLoader, DataLoader):
    random_order = list(range(len(input_ids)))
    np.random.shuffle(random_order)

    input_ids_train = np.array([input_ids[i] for i in random_order[:int(len(input_ids) * ratio)]])
    input_types_train = np.array([input_types[i] for i in random_order[:int(len(input_ids) * ratio)]])
    input_masks_train = np.array([input_masks[i] for i in random_order[:int(len(input_ids) * ratio)]])
    y_train = np.array([labels[i] for i in random_order[:int(len(input_ids) * ratio)]])

    input_ids_test = np.array(
        [input_ids[i] for i in random_order[int(len(input_ids) * ratio):]])
    input_types_test = np.array(
        [input_types[i] for i in random_order[int(len(input_ids) * ratio):]])
    input_masks_test = np.array(
        [input_masks[i] for i in random_order[int(len(input_ids) * ratio):]])
    y_test = np.array([labels[i] for i in random_order[int(len(input_ids) * ratio):]])

    train_data = TensorDataset(torch.LongTensor(input_ids_train),
                               torch.LongTensor(input_types_train),
                               torch.LongTensor(input_masks_train),
                               torch.LongTensor(y_train))
    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, drop_last=True)

    valid_data = TensorDataset(torch.LongTensor(input_ids_test),
                               torch.LongTensor(input_types_test),
                               torch.LongTensor(input_masks_test),
                               torch.LongTensor(y_test))
    valid_sampler = SequentialSampler(valid_data)
    valid_loader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_size, drop_last=True)
    return train_loader, valid_loader


# load data
def get_data(file_path, padded_size):
    corpus = []
    labels = []
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
    for i in range(len(df)):
        text = df.iloc[i]["微博中文内容"].strip()
        label = df.iloc[i]["情感倾向"]
        if label not in ['-1', '0', '1']:
            continue
        label=int(label)
        if label == -1:
            label= 2
        corpus.append(text)
        labels.append(label)
    assert len(corpus) == len(labels)
    corpus = [text[0:padded_size] for text in corpus]
    return corpus, labels


# model definition
class BertClassifier(nn.Module):
    def __init__(self, hidden_size=768, mid_size=256):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
        # self.bert = BertModel.from_pretrained("nghuyong/ernie-gram-zh")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, mid_size),
            nn.BatchNorm1d(mid_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mid_size, 3),
        )

    def forward(self, x):
        context = x[0]
        types = x[1]
        mask = x[2]

        # _, pooled = self.bert(context, token_type_ids=types,
        #                       attention_mask=mask,
        #                       output_all_encoded_layers=False)
        # last_hidden_state, pooled = self.bert(context, token_type_ids=types,
                            #   attention_mask=mask)
        # pooled = self.bert(context, token_type_ids=types,
        #                       attention_mask=mask).last_hidden_state[:,1,:]
        pooled = self.bert(context, token_type_ids=types,
                              attention_mask=mask).pooler_output
        # print(pooled)
        context_embedding = self.dropout(pooled)

        output = self.classifier(context_embedding)
        output = F.softmax(output, dim=1)
        return output


def train_step(model, device, train_loader, optimizer, epoch, scheduler):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (x1, x2, x3, y) in enumerate(train_loader):
        x1_g, x2_g, x3_g, y_g = x1.to(device), x2.to(device), x3.to(device), y.to(device)
        y_pred = model([x1_g, x2_g, x3_g])
        optimizer.zero_grad()

        loss = criterion(y_pred, y_g)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # if (batch_idx + 1) % 100 == 0:
        if (batch_idx + 1) % 500 == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(epoch, (batch_idx + 1) * len(x1),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))


def valid_step(model, device, valid_loader):
    model.eval()
    valid_loss = 0.0
    valid_true = []
    valid_pred = []
    criterion = nn.CrossEntropyLoss()
    # for batch_idx, (x1, x2, x3, y) in tqdm(enumerate(valid_loader)):
    for batch_idx, (x1, x2, x3, y) in enumerate(valid_loader):
        x1_g, x2_g, x3_g, y_g = x1.to(device), x2.to(device), x3.to(device), y.to(device)
        with torch.no_grad():
            y_pred_pa_all = model([x1_g, x2_g, x3_g])

        valid_loss += criterion(y_pred_pa_all, y_g)
        batch_true = y_g.cpu()
        batch_pred = y_pred_pa_all.detach().cpu().numpy()
        for item in batch_pred:
            valid_pred.append(item.argmax(0))
        for item in np.array(batch_true):
            valid_true.append(item)

    valid_loss /= len(valid_loader)
    logger.info('Test set: Average loss: {:.4f}'.format(valid_loss))
    valid_true = np.array(valid_true)
    valid_pred = np.array(valid_pred)
    avg_acc = accuracy_score(valid_true, valid_pred)
    avg_f1s = f1_score(valid_true, valid_pred, average='macro')

    logger.info('Average: Accuracy: {:.3f}%, F1Score: {:.3f}'.format(100 * avg_acc, 100 * avg_f1s))
    logger.info(classification_report(valid_true, valid_pred))

    return avg_acc, avg_f1s, valid_loss


if __name__ == '__main__':
    # some arguments
    epochs = 20
    padded_size = 512
    batch_size = 4
    learning_rate = 1e-5
    ratio = 0.8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set logger
    logger = custom_logger('train', 'INFO')

    # load data and tokenize/mask
    train_path = os.path.join('data', 'nCoV_100k_train.labled.csv')
    corpus,labels = get_data(train_path, padded_size)
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    tokenized_data = tokenize_corpus(corpus, tokenizer)
    logger.info('tokenized')

    input_ids, input_types, input_masks = mask_token_ids(tokenized_data, padded_size)
    logger.info('masked')

    # split dataset
    train_loader, valid_loader = split_train_dataset(input_ids, input_types, input_masks,
                                                 labels,
                                                 batch_size,
                                                 ratio)
    
    # init model
    model = BertClassifier()
    model.to(device)
    logger.info(f"+++ model init on {device} +++")

    # init optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_group_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_group_parameters,
                        lr=learning_rate,
                        eps=1e-8)
    logger.info("+++ optimizer init +++")
    num_training_steps = int(len(input_ids) * ratio) * epochs
    warmup = 0.1
    num_warmup_steps = num_training_steps * warmup
    scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps-num_warmup_steps)


    # model training
    best_acc = 0.0
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        train_step(model, device, train_loader, optimizer, epoch, scheduler)
        acc, fis, loss = valid_step(model, device, valid_loader)
        if best_acc < acc:
            best_acc = acc
            best_epoch = epoch
    logger.info(f"+++ bert train done, best epoch: {best_epoch} +++")