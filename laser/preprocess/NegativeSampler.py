import spacy
import json
import os
from argparse import ArgumentParser
from tqdm import tqdm
from spacy.language import Language
import spacy_fastlang

nlp = spacy.load('en_core_web_lg')
nlp.add_pipe("language_detector")
remove_name_ls = ['people', 'individual', 'person', 'female', 'appear', 'one', 'two', 'three', 'four', "head"] 
remove_single_name_ls = ['is', 'are', 'as', 'at', 'by', 'for', 'has', 'from', 'try']

def get_neg_examples_helper(candidate_reps, kw_reps, threshold):

    all_neg_sim = []
    all_sim = []

    for entity_rep in candidate_reps:

        is_sim = False
        for unary_rep in kw_reps:
            unary_sim_score = entity_rep.similarity(unary_rep)
            if unary_sim_score > threshold:
                is_sim = True

        if is_sim:
            all_sim.append(entity_rep)
        else:
            all_neg_sim.append(entity_rep)

    return all_sim, all_neg_sim

def get_negative_examples(spec, all_entity_rep, all_unary_rep, all_binary_rep):
    entity_kw_reps = [nlp(kw) for kw in list(set(spec['consts']))]
    unary_kw_reps = [nlp(kw) for kw in list(set(spec['unary_kws']))]
    binary_kw_reps = [nlp(kw) for kw in list(set(spec['binary_kws']))]

    _, all_neg_entity_example = get_neg_examples_helper(all_entity_rep, entity_kw_reps, threshold=0.4)
    _, all_neg_unary_example = get_neg_examples_helper(all_unary_rep, unary_kw_reps, threshold=0.3)
    _, all_neg_binary_example = get_neg_examples_helper(all_binary_rep, binary_kw_reps, threshold=0.3)

    return  [e.text for e in all_neg_entity_example], [u.text for u in all_neg_unary_example], [b.text for b in all_neg_binary_example]


def get_all_neg_examples(gpt_specs, neg_spec_output_path, all_entity_rep, all_unary_rep, all_binary_rep):

    print("start getting all negative examples")
    
    for data_id, datapoint in tqdm(gpt_specs.items()):

        neg_entity, neg_unary, neg_binary = get_negative_examples(datapoint,
                                                                    all_entity_rep,
                                                                    all_unary_rep,
                                                                    all_binary_rep)
        negative_example = {}

        negative_example['video_id'] = data_id
        negative_example['neg_entity'] = neg_entity
        negative_example['neg_unary'] = neg_unary
        negative_example['neg_binary'] = neg_binary
       
        json_line = json.dumps(negative_example.copy())
        with open(neg_spec_output_path, 'a') as f:
            f.write(json_line + '\n')

def parse_args():

    dataset = "LLaVA"

    data_dir = os.path.abspath(os.path.join(__file__, "../../../../data/LLaVA-Video-178K-v2"))
    nl2spec_dir = os.path.join(data_dir, "nl2spec")

    assert os.path.exists(data_dir)

    # Setup argument parser
    parser = ArgumentParser(dataset)

    # Setting up directories
    parser.add_argument("--data-dir", type=str, default=data_dir)
    parser.add_argument("--nl2spec-dir", type=str, default=nl2spec_dir)

    args = parser.parse_args()

    return args

def dict2sorted_ls(all_cates, min_ct, remove_top_num):
    all_cate_ls = []
    for key, ct in all_cates.items():
        if ct > min_ct:
            all_cate_ls.append((ct, key))
    all_cate_ls = sorted(all_cate_ls, reverse=True)[remove_top_num:]
    all_cate_ls = [i[1] for i in all_cate_ls]
    return all_cate_ls

def update_dict(kw_ls, all_cates):
    removed = []
    
    for i in kw_ls:
        to_add = True
        for remove_name in remove_name_ls:
            if remove_name in i.split(' '):
                removed.append(i)
                to_add = False
            
        if i in remove_single_name_ls:
            removed.append(i)
            to_add = False
        
        if to_add:
            if not i in all_cates:
                all_cates[i] = 0
                
            all_cates[i] += 1
    return all_cates, removed

def collect_kws(nl_data, remove_top_ct=20, min_ct=15):
    all_cates = {}
    all_unaries = {}
    all_binary_preds = {}

    all_removed_cates = []
    all_removed_unaries = []
    all_removed_binaries = []
    for data_id, datapoint in nl_data.items():
        
        all_cates, removed_cates = update_dict(datapoint['consts'], all_cates)
        all_unaries, removed_unaries = update_dict(datapoint['unary_kws'], all_unaries)
        all_binary_preds, removed_binary = update_dict(datapoint['binary_kws'], all_binary_preds)

        all_removed_cates += (removed_cates)
        all_removed_unaries += (removed_unaries)
        all_removed_binaries += (removed_binary)
        
    all_cate_ls = dict2sorted_ls(all_cates, min_ct, remove_top_ct)
    all_unary_ls = dict2sorted_ls(all_unaries, min_ct, remove_top_ct)
    all_binary_ls = dict2sorted_ls(all_binary_preds, min_ct, remove_top_ct)
        
    return all_cate_ls, all_unary_ls, all_binary_ls

def clean_kws(key_words):
    cleaned_kws = []
    for key_word in key_words:
        if "'" == key_word[0]:
            continue
        key_word_norm = key_word.replace('-', ' ').replace(',', ' ').replace('.', ' ').replace('_', ' ')
        key_word_ls = key_word_norm.split(' ')
        if len(key_word_ls) < 4:
            cleaned_kws.append(key_word_norm)
    return cleaned_kws

def remove_non_english(key_words_nlp):
    new_kws = []
    for keyword in key_words_nlp:
        if keyword._.language == 'en' and keyword._.language_score >= 0.8:
            new_kws.append(keyword)
    return new_kws
