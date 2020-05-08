
import os
import csv
import numpy as np
import shelve

import numpy as np
import re

import argparse

class Evaluator():
    def __init__(self, gold_filename, guess_filename, gold_header=False, guess_header=False):
        self.gold_filename = gold_filename
        self.guess_filename = guess_filename
        self.gold_header = gold_header
        self.guess_header = guess_header

        self.check_gold_guss_file_are_match()
        
        self.gold_data = Evaluator.get_slotset(self.gold_filename, header=self.gold_header)
        self.guess_data = Evaluator.get_slotset(self.guess_filename, header=self.guess_header)

        
        self.slots_template = Evaluator.get_slots_template(self.gold_filename, self.gold_header, self.guess_filename, self.guess_header)
    def _strict_matching(self):
        count = dict()
        for l in self.slots_template:
            count[l] = dict()
            for j in ['TP', 'FP', 'FN']:
                count[l][j] = 0

        for mention in self.gold_data:
            l = re.split(':', mention)[-1]
            if mention in self.guess_data:
                count[l]['TP'] += 1
            else:
                count[l]['FN'] += 1
        for mention in self.guess_data:
            l = re.split(':', mention)[-1]
            if mention not in self.gold_data:
                count[l]['FP'] += 1
        
        self.count = count
        self.number_of_goldslot = Evaluator.get_number_of_gold_slot(self.gold_data)
    def _relax_matching(self):
        gold_relaxset = Evaluator._strictset_to_relaxset(self.gold_data)
        guess_relaxset = Evaluator._strictset_to_relaxset(self.guess_data)

        gold_relaxset_corres_slot = dict()
        guess_relaxset_corres_slot = dict()
    
        count = dict()
    
        for slot in self.slots_template:
            gold_relaxset_corres_slot[slot] = list()
            guess_relaxset_corres_slot[slot] = list()

            for mention in gold_relaxset:
                if mention[2] == slot: #mention[2] is slot_name
                    gold_relaxset_corres_slot[slot].append(mention)

            for mention in guess_relaxset:
                if mention[2] == slot: #mention[2] is slot_name
                    guess_relaxset_corres_slot[slot].append(mention)

        for slot in self.slots_template:
            count[slot] = dict()
            count[slot]['TP'] = 0
            count[slot]['FP'] = 0
            count[slot]['FN'] = 0
    
        for slot in self.slots_template:

            #Đếm TP - mention có trong tập gold và cả trong tập guess
            #Đếm FN - mention có trong tập gold mà không có trong tập guess
            for gold_mention in gold_relaxset_corres_slot[slot]:
                TP = False
                for guess_mention in guess_relaxset_corres_slot[slot]:
                    if gold_mention[0] == guess_mention[0]: #same sentence_id
                        gold_ids_word_set = gold_mention[1]
                        guess_ids_word_set = guess_mention[1]
                    
                        if bool(gold_ids_word_set & guess_ids_word_set): #only True if gold_ids_word_set intersect guess_ids_word_set
                            TP = True
                            break
                if TP:
                    count[slot]['TP'] += 1
                else:
                    count[slot]['FN'] += 1

            #Đếm FP - mention có trong tập guess mà không có trong tập gold
            for guess_mention in guess_relaxset_corres_slot[slot]:
                notFP = False
                for gold_mention in gold_relaxset_corres_slot[slot]:
                    if gold_mention[0] == guess_mention[0]: #same sentence_id
                        gold_ids_word_set = gold_mention[1]
                        guess_ids_word_set = guess_mention[1]

                        if bool(gold_ids_word_set & guess_ids_word_set): #only True if gold_ids_word_set intersect guess_ids_word_set
                            notFP = True
                            break
                if not notFP:
                    count[slot]['FP'] += 1

        self.count = count
        self.number_of_goldslot = Evaluator.get_number_of_gold_slot(self.gold_data)

    def _token_matching(self):
        gold_tokenset = Evaluator._slotset_to_tokenset(self.gold_data)
        guess_tokenset = Evaluator._slotset_to_tokenset(self.guess_data)

        count = dict()
        for l in self.slots_template:
            count[l] = dict()
            for j in ['TP', 'FP', 'FN']:
                count[l][j] = 0

        for mention in gold_tokenset:
            l = re.split(':', mention)[-1]
            if mention in guess_tokenset:
                count[l]['TP'] += 1
            else:
                count[l]['FN'] += 1
        for mention in guess_tokenset:
            l = re.split(':', mention)[-1]
            if mention not in gold_tokenset:
                count[l]['FP'] += 1
        
        self.count = count
        self.number_of_goldslot = Evaluator.get_number_of_gold_slot(gold_tokenset)
    def evaluate(self, type_matching):
        #"strict": strict matching
        #"relax": relax matching

        
        assert type_matching in ['strict', 'relax', 'token']
        if type_matching == 'strict':
            self._strict_matching()
        elif type_matching == 'relax':
            self._relax_matching()
        else:
            self._token_matching()

        flatten_tp_fp_fn = np.zeros(shape=(len(self.slots_template), 3), dtype=np.int32)
        for idx, label in enumerate(self.slots_template):
            for subidx, j in enumerate(['TP', 'FP', 'FN']):
                flatten_tp_fp_fn[idx, subidx] = self.count[label][j]

        result = list()   

        #Overall - Macro
        TP = flatten_tp_fp_fn[:, 0].sum()
        FP = flatten_tp_fp_fn[:, 1].sum()
        FN = flatten_tp_fp_fn[:, 2].sum()
        precision, recall, F1 = Evaluator.calculate_p_r_f1(TP, FP, FN)
        result.append(['Macro', precision, recall, F1])

        sum_p = 0
        sum_r = 0
        #Each entity
        for i, label in enumerate(self.slots_template):
            TP = flatten_tp_fp_fn[i, 0]
            FP = flatten_tp_fp_fn[i, 1]
            FN = flatten_tp_fp_fn[i, 2]
            precision, recall, F1 = Evaluator.calculate_p_r_f1(TP, FP, FN)
            sum_p += precision
            sum_r += recall
            result.append([label, precision, recall, F1])

        avg_p = sum_p / len(self.slots_template)
        avg_r = sum_r / len(self.slots_template)
        avg_F1 = 2*avg_r*avg_p/(avg_p + avg_r)
        result.insert(1, ["Micro", avg_p, avg_r, avg_F1])
        self.result = result

    
    def pretty_print(self):
        print('%20s %20s %20s %20s %20s' % ('ENTITY', 'Precision', 'Recall', 'F1-Score', ''))
        for row in self.result:
            slot_name, precision, recall, F1 = row
            if slot_name in self.number_of_goldslot:
                print('%20s %20.2f%% %20.2f%% %20.2f %20d' % (slot_name, precision, recall, F1, self.number_of_goldslot[slot_name]))
            else:
                print('%20s %20.2f%% %20.2f%% %20.2f' % (slot_name, precision, recall, F1))

            
    def check_gold_guss_file_are_match(self):
        with open(self.gold_filename, encoding='utf-8') as gold_file, open(self.guess_filename, encoding='utf-8') as guess_file:
            if self.gold_header:
                gold_file.readline()
            if self.guess_header:
                guess_file.readline() #title

            for gold_line, guess_line in zip(gold_file.readlines(), guess_file.readlines()):
                gold_line_splitted = re.split('\t', gold_line.strip())
                guess_line_splitted = re.split('\t', guess_line.strip())
                # print(gold_line)
                # print(guess_line)
                # print()
                if len(gold_line_splitted) != len(guess_line_splitted):
                    raise Exception("Gold - guess data không khớp")
                if len(gold_line_splitted) > 1:
                    if gold_line_splitted[0].lower() != guess_line_splitted[0].lower():
                        print(gold_line)
                        print(guess_line)
                        raise Exception("Gold - guess data không khớp")

    @staticmethod
    def _slotset_to_tokenset(slotset):
        tokenset = list()
        for mention in slotset:
            id_sentence, ids_token, slot_name = mention.split(':')
            for id_token in ids_token.split(','):
                token_mention = "%s:%s:%s" % (id_sentence, id_token, slot_name)
                tokenset.append(token_mention)
        return tokenset

    @staticmethod
    def _mention_str_to_list(mention):
        id_sentence, ids_token, slot_name = mention.split(':')
        ids_token_set = set(eval(ids_token)) if ',' in ids_token else set([eval(ids_token)])
        return [id_sentence, ids_token_set, slot_name]

    @staticmethod
    def _strictset_to_relaxset(slotset):
        return [Evaluator._mention_str_to_list(mention) for mention in slotset]

    @staticmethod
    def calculate_p_r_f1(TP, FP, FN):
        if TP == 0 and (TP + FP) == 0:
            precision = 0
        else:
            precision = float(TP)/(TP + FP)
                
        if TP == 0 and (TP + FN) == 0:
            recall = 0
        else:
            recall = float(TP)/(TP + FN)
            
        if precision == 0.0 and recall == 0.0:
            F1 = 0.0
        else:
            F1 = 2*recall*precision/(precision + recall)

        return precision*100, recall*100, F1*100

    @staticmethod
    def get_slotset(fn, header):
        slotset = list()

        idx_sent = 0
        idx_word = 0
        cur_slot = None #Label đang xét
        pre_slot = None #Label ở trên đó một dòng, nếu dòng đang xét là đầu câu thì bằng None
        
        leng = 0
        ids_slot = list()
        with open(fn, encoding='utf-8') as fin:
            if header:
                fin.readline()
            for line in fin.readlines():
                row = re.split("\t", line.strip())
                if len(row) == 1:
                    if leng > 0:
                        if pre_slot != 'O': #Nếu slot khác O ở cuối câu
                            men = "S%d:%s:%s" % (idx_sent, ','.join([str(i) for i in ids_slot]), pre_slot)
                            slotset.append(men)
                        idx_sent += 1
                    pre_slot = None
                    cur_slot = None
                    ids_slot = list()
                    leng = 0
                    
                else:
                    cur_slot = row[1].strip()

                    if pre_slot is None: #Con trỏ đang ở đầu câu
                        if cur_slot != 'O': #Label ở đầu câu khác O
                            ids_slot.append(leng)

                    else: #Con trỏ đang ở giữa câu
                        if pre_slot != 'O': #Label trước đó khác O
                            if cur_slot == pre_slot: #Vẫn còn mở slot
                                ids_slot.append(leng)

                            else: #Kết thúc slot
                                men = "S%d:%s:%s" % (idx_sent, ','.join([str(i) for i in ids_slot]), pre_slot)
                                slotset.append(men)
                                ids_slot = list()
                                if cur_slot != 'O': #Con trỏ hiện tại có một slot khác
                                    ids_slot.append(leng)

                        else:
                            if cur_slot != 'O': #Slot trước đó là O, có slot mới
                                ids_slot.append(leng)
                    pre_slot = cur_slot
                    leng += 1

        return set(slotset)

    @staticmethod
    def get_slots_template(gold_fn, gold_header, guess_fn, guess_header):
        import os
        import re
        slots = []
        with open(gold_fn, encoding='utf-8') as f:
            if gold_header:
                f.readline()
            for line in f.readlines():
                row = re.split("\t", line.strip())
                if len(row) != 1:
                    slot = row[1]
                    if slot not in slots:
                        slots.append(slot)
        with open(guess_fn, encoding='utf-8') as f:
            if guess_header:
                f.readline()
            for line in f.readlines():
                row = re.split("\t", line.strip())
                if len(row) != 1:
                    slot = row[1]
                    if slot not in slots:
                        slots.append(slot)
        
        if 'O' in slots:
            del slots[slots.index('O')]
        
        return sorted(slots)

    @staticmethod
    def get_number_of_gold_slot(gold_slot):
        slots = dict()
        for mention in gold_slot:
            slot = mention.split(':')[-1]
            if slot not in slots:
                slots[slot] = 0
            slots[slot] += 1
        return slots
if __name__ == '__main__':
    gold_filename = '../cx_predicted/ground_truth_type4_distance.txt'
    guess_filename = '../cx_predicted/Predict_type4_distance_noCNN.txt'


    description = 'Đánh giá kết quả mô hình'
    epilog = 'Nếu không kiểu đánh giá nào được chọn, mặc định strict matching sẽ được chọn'
    parser = argparse.ArgumentParser(description=description, epilog=epilog, fromfile_prefix_chars='@')

    

    parser.add_argument('--gold-fn', help='Đường dẫn đến gold file')
    parser.add_argument('--gold-header', action='store_true',
                        help='True nếu file gold data có header, false nếu không (Mặc định: False)')
    parser.add_argument('--guess-fn', help='Đường dẫn đến guess file')
    parser.add_argument('--guess-header', action='store_true',
                        help='True nếu file guess data có header, false nếu không (Mặc định: False)')
    parser.add_argument('-s', '--strict', '--strict-matching', action='store_true',
                        help='Đánh giá bằng strict matching (Mặc định: False)')
    parser.add_argument('-r', '--relax', '--relax-matching', action='store_true',
                        help='Đánh giá bằng relax matching (Mặc định: False)')
    parser.add_argument('-t', '--token', '--token-matching', action='store_true',
                        help='Đánh giá bằng token matching (Mặc định: False)')
    args = parser.parse_args()
    args.strict = args.strict or (not(args.strict or args.relax or args.token))

    gold_filename = args.gold_fn
    gold_header = args.gold_header

    guess_filename = args.guess_fn
    guess_header = args.guess_header

    evaluation = Evaluator(gold_filename, guess_filename, gold_header=gold_header, guess_header=guess_header)
    if args.strict:
        print()
        print("\tSTRICT MATCHING")
        evaluation.evaluate("strict")
        evaluation.pretty_print()
        print()
    
    if args.relax:
        print()
        print("\tRELAX MATCHING")
        evaluation.evaluate("relax")
        evaluation.pretty_print()
        print()

    if args.token:
        print()
        print("\tTOKEN MATCHING")
        evaluation.evaluate("token")
        evaluation.pretty_print()
        print()