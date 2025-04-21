import os
import json
import torch
import numpy as np

from collections import namedtuple
from model import BertNer
from seqeval.metrics.sequence_labeling import get_entities
from transformers import BertTokenizer


def get_args(args_path, args_name=None):
    with open(args_path, "r") as fp:
        args_dict = json.load(fp)
    # 注意args不可被修改了
    args = namedtuple(args_name, args_dict.keys())(*args_dict.values())
    return args


class Predictor:
    def __init__(self, data_name):
        self.data_name = data_name
        self.ner_args = get_args(os.path.join("./checkpoint/{}/".format(data_name), "ner_args.json"), "ner_args")
        self.ner_id2label = {int(k): v for k, v in self.ner_args.id2label.items()}
        self.tokenizer = BertTokenizer.from_pretrained(self.ner_args.bert_dir)
        self.max_seq_len = self.ner_args.max_seq_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ner_model = BertNer(self.ner_args)
        self.ner_model.load_state_dict(torch.load(os.path.join(self.ner_args.output_dir, "pytorch_model_ner.bin")))
        self.ner_model.to(self.device)
        self.data_name = data_name

    def ner_tokenizer(self, text):
        # print("文本长度需要小于：{}".format(self.max_seq_len))
        text = text[:self.max_seq_len - 2]
        text = ["[CLS]"] + [i for i in text] + ["[SEP]"]
        tmp_input_ids = self.tokenizer.convert_tokens_to_ids(text)
        input_ids = tmp_input_ids + [0] * (self.max_seq_len - len(tmp_input_ids))
        attention_mask = [1] * len(tmp_input_ids) + [0] * (self.max_seq_len - len(tmp_input_ids))
        input_ids = torch.tensor(np.array([input_ids]))
        attention_mask = torch.tensor(np.array([attention_mask]))
        return input_ids, attention_mask

    def ner_predict(self, text):
        input_ids, attention_mask = self.ner_tokenizer(text)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        output = self.ner_model(input_ids, attention_mask)
        attention_mask = attention_mask.detach().cpu().numpy()
        length = sum(attention_mask[0])
        logits = output.logits
        logits = logits[0][1:length - 1]
        logits = [self.ner_id2label[i] for i in logits]
        entities = get_entities(logits)
        result = {}
        for ent in entities:
            ent_name = ent[0]
            ent_start = ent[1]
            ent_end = ent[2]
            if ent_name not in result:
                result[ent_name] = [("".join(text[ent_start:ent_end + 1]), ent_start, ent_end)]
            else:
                result[ent_name].append(("".join(text[ent_start:ent_end + 1]), ent_start, ent_end))
        return result
if __name__ == "__main__":
    data_name = "dgre"
    predictor = Predictor(data_name)
    if data_name == "dgre":
        texts = [
            "吴淞交管HUA HANG SHANG HAI在三号码头可以离泊了吗？",
            "HUA  SHANG HAI是去上上行的是吧。对上行目的港去武汉。离开在航道边稍微等一下，有空档再穿。好的好的。",
            "啊，YI FENG LUN，请讲，YI FENG LUN。注意一下，跟那个进口船联系好好吧，协调好以后再穿过来。啊联系了在联系了，那个谁那个我HUI XI 三的前面划。",
            "ZHONG ZHONG 六啊，你个五点几的车度涨水你害人啊在警戒区慢车哦。我开了开着呢好吧。都像你们这么自私人家船怎么搞。",
            "吴淞交管，WAN BANG XIANG YUN。请讲。老师WAN BANG XIANG YUN两号锚地一号锚地之间这个地方，呃，马上快出锚地，准备走四十七号那个进口海轮那银宁那尾部，呃，划江去去外五你看行不行？那你银宁后面的锦海顺两有没有联系过啊？呃，它这个档位，我看可以，呃如果那个我就联系一下这个QING HAI那个。联系着啊，需要他同意配合的。",
            "呃，吴淞控制中心弘泰六幺三。那你看到没有也在那个锦海顺两左舷都都一起协调一下。呃吴淞控制中心弘泰六幺三，啊六幺三往前面再走一点往。三号，三号锚这个方向再走一点。再往三号锚地再走一点，就，额抛它的中心线好吗？对啊对的，五号的中心线再往前走一点，走到等一下附近好的，明白。给南边的再留着一个深水锚位，好吧。",
        ]
    for text in texts:
        ner_result = predictor.ner_predict(text)
        print("文本>>>>>：", text)
        print("实体>>>>>：", ner_result)
        print("="*100)

