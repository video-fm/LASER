from pyparsing import nestedExpr, Word, alphas
import pyparsing as pp

import os
import json
import re
from functools import reduce
from spacy.language import Language
import spacy_fastlang

import spacy
nlp = spacy.load('en_core_web_lg')
nlp.add_pipe("language_detector")

from laser.preprocess.utils import bool_token, kw_preds, not_kw_preds, var2vid, rec_sub_val

##########################################################################
##############                                              ##############
##############         STSL language building blocks        ##############
##############                                              ##############
##########################################################################


##########################################################################
##############                Helper Functions              ##############
##########################################################################

def process_arg_num(raw_arg, consts):
    if len(raw_arg) == 2 and raw_arg[0] == '?':
        arg = var2vid[raw_arg[1:]]
    else:
        arg = -consts.index(raw_arg) - 1
    return arg

def process_arg_str(raw_arg):
    if raw_arg[0] == '?':
        arg = raw_arg[1:]
    else:
        arg = raw_arg
    return arg

def process_arg(raw_arg, consts, num=True):
    if num:
        return process_arg_num(raw_arg, consts)
    else:
        return process_arg_str(raw_arg)

def merge_dict_ls(d1, d2):
    for k, v in d2.items():
        if not k in d1:
            d1[k] = []
        d1[k] += v
    return d1

def chunk_list(input_list, n):
    progs = []
    # Split the list into chunks of size n
    for i in range(0, len(input_list), n):
        windowed_prog = input_list[i:i+n]
        prog_window_start = i
        prog_window_end = i+len(windowed_prog)
        progs.append(((prog_window_start, prog_window_end), windowed_prog))
    return progs



##########################################################################
##############                 STSL Language                ##############
##########################################################################

class BoolFormula():

    def __init__(self, name, args) -> None:
        self.name = name
        self.args = args
        if not len(args) == 0:
            self.v = set.union(*[arg.v for arg in args])
        else:
            self.v = set()

    def __eq__(self):
        arg_num = len(self.args)
        for ct1 in range(arg_num):
            has_match = False
            for ct2 in range(arg_num):
                if not ct1 == ct2:
                    if self.args[ct1] == self.args[ct2]:
                        has_match = True
            if not has_match:
                return False
        return True

    def collect_preds(self):
        all_preds = set()
        for arg in self.args:
            all_preds = all_preds.union(arg.collect_preds())
        return all_preds

    def collect_tuples(self):
        all_preds = []
        for arg in self.args:
            tuple = arg.collect_tuples()
            if not len(tuple) == 0:
                all_preds += (tuple)
        return all_preds

    def collect_kws(self):
        result = {'unary': set(), 'binary': set()}
        for arg in self.args:
            arg_kws = arg.collect_kws()
            [result['unary'].add(i) for i in arg_kws['unary']]
            [result['binary'].add(i) for i in arg_kws['binary']]
        return result

    def to_scl(self, expression_id, consts):

        expressions = {}
        if len(self.args) == 0:
            return expression_id, expressions

        current_eid = expression_id
        for arg in self.args:
            current_eid, arg_scl = arg.to_scl(current_eid, consts)
            expressions = merge_dict_ls(expressions, arg_scl)

        # Should only have this case
        assert self.name == "and"

        return current_eid, expressions

    def to_str(self, consts):

        expressions = {}

        expression_strs = []
        for arg in self.args:
            try:
                arg_str = arg.to_str(consts)
            except InvalidArgException:
                continue
            expression_strs.append(arg_str)

        if len(expression_strs) == 0:
            return ""

        expressions = reduce(lambda accu, elem: f"And({elem}, {accu})", reversed(expression_strs))
        
        # expressions = "And(" + ', '.join(expression_strs) + ')'
        # Should only have this case
        assert self.name == "and"

        return expressions

class PosPredicate():

    def __init__(self, pred, args) -> None:
        if not isinstance(args, list):
            raise InvalidArgException

        for arg in args:
            if not isinstance(arg, str):
                raise InvalidArgException

        self.name = pred.replace('-', '_')
        self.args = args
        self.v = set([arg for arg in args if '?' == arg[0]])

    def __eq__(self, other):
        if not self.name == other.name:
            return False
        if not self.args == other.args:
            return False
        return True

    def collect_preds(self):
        return set([self.name])

    def collect_tuples(self):
        return [(True, self.name, self.args)]

    def collect_kws(self):

        if len(self.args) == 1:
            return {'unary': [self.name], 'binary': []}
        elif len(self.args) == 2:
            return {'unary': [], 'binary': [self.name]}
        else:
            # Invalid Case
            return {'unary': [], 'binary': []}

    def to_scl(self, expression_id, consts):

        expressions = {}
        current_eid = expression_id
        next_eid = exp_gen.get()

        if len(self.args) == 1:
            if not 'positive_unary_atom' in expressions:
                expressions['positive_unary_atom'] = []
            expressions['positive_unary_atom'].append(
                (next_eid, self.name,
                process_arg(self.args[0], consts),
                current_eid))


        elif len(self.args) == 2:

            if not 'positive_binary_atom' in expressions:
                expressions['positive_binary_atom'] = []
            expressions['positive_binary_atom'].append(
                (next_eid, self.name,
                process_arg(self.args[0], consts),
                process_arg(self.args[1], consts),
                current_eid))
        else:
            print (f"Warning: ignoring {self.name}(', '.join({self.args})")

        return next_eid, expressions

    def to_str(self, consts):
        wrapped_args = [f"\"{self.name}\""]

        for arg in self.args:
            arg = process_arg(arg, consts)
            if arg < 0:
                wrapped_args.append(f"Const({arg})")
            elif arg > 0:
                wrapped_args.append(f"Var({arg})")
            else:
                raise InvalidArgException()

        if len(self.args) == 1:
            logic_pred = "Unary"
        elif len(self.args) == 2:
            logic_pred = "Binary"
        else:
            raise InvalidArgException

        return f"{logic_pred}({','.join(wrapped_args)})"

class NegPredicate():

    def __init__(self, pred, args) -> None:
        for arg in args:
            if not isinstance(arg, str):
                raise InvalidArgException

        self.name = pred.replace('-', '_')
        self.args = args
        self.v = set([arg for arg in args if '?' == arg[0]])

    def __eq__(self, other):
        if not self.name == other.name:
            return False
        if not self.args == other.args:
            return False
        return True

    def collect_preds(self):
        return set([self.name])

    def collect_tuples(self):
        return [(False, self.name, self.args)]

    def to_scl(self, expression_id, consts):

        expressions = {}
        current_eid = expression_id
        next_eid = exp_gen.get()

        if len(self.args) == 1:
            expressions['negative_unary_atom'] = [
                (next_eid, self.name,
                process_arg(self.args[0], consts), current_eid)]
        else:
            assert(len(self.args) == 2)
            expressions['negative_binary_atom'] = [
                (next_eid, self.name,
                process_arg(self.args[0], consts),
                process_arg(self.args[1], consts),
                current_eid)]

        return next_eid, expressions

    def collect_kws(self):

        if len(self.args) == 1:
            return {'unary': [self.name], 'binary': []}
        elif len(self.args) == 2:
            return {'unary': [], 'binary': [self.name]}
        else:
            # Invalid Case
            return {'unary': [], 'binary': []}

    def to_str(self, consts):
        wrapped_args = [f"\"{self.name}\""]
        for arg in self.args:
            arg = process_arg(arg, consts)
            if arg < 0:
                wrapped_args.append(f"Const({arg})")
            elif arg > 0:
                wrapped_args.append(f"Var({arg})")
            else:
                raise InvalidArgException()

        if len(self.args) == 1:
            logic_pred = "NegUnary"
        elif  len(self.args) == 2:
            logic_pred = "NegBinary"
        else:
            raise InvalidArgException

        return f"{logic_pred}({','.join(wrapped_args)}))"

class InEqPred():

    def __init__(self, pred, args) -> None:
        assert pred == '!='
        self.args = args
        self.v = set([arg for arg in args if '?' == arg[0]])

    def __eq__(self, other) -> bool:
        return self.args == other.args

    def collect_preds(self):
        return set()

    def collect_tuples(self):
        return []

    def to_scl(self, expression_id, consts):
        current_eid = expression_id
        next_eid = exp_gen.get()
        expressions = {}
        expressions['inequality_constraint'] = [
            (next_eid,
            process_arg(self.args[0], consts),
            process_arg(self.args[1], consts),
            current_eid)]

        return next_eid, expressions

##########################################################################
##############                                              ##############
##############             STSL language Parser             ##############
##############              (OpenPVSG Version)              ##############
##############                                              ##############
##########################################################################

##########################################################################
##############               Utility Functions              ##############
##########################################################################

class UnseenParamException(Exception):
    pass

class InvalidArgException(Exception):
    pass

class InvalidPrediction(Exception):
    pass

re_num = "\([0-9a-zA-Z\,\- ]+\)"

def normalize_text(text):
    return text.replace(',', ' ').replace("'", ' ').replace("_", ' ').replace("-", ' ')

def clean_cap(caption):
    current_var_id = 0
    caption = '. '.join(caption)
    caption = normalize_text(caption)
    description_ls = caption.split(' ')
    new_description = []
    to_ignore = re.findall(re_num, caption)
    new_cap = caption
    for tk in to_ignore:
        new_cap = new_cap.replace(tk, '')
    new_cap = new_cap.replace('  ', ' ')
    new_cap = new_cap.replace(' .', '.')
    new_cap = new_cap.replace(' ,', ',')
    new_cap = new_cap.strip()

    return new_cap

def replace_nested_list(l, from_words, to_words):
    new_list = []
    for e in l:
        if isinstance(e, list):
            new_list.append(replace_nested_list(e, from_words, to_words))
        elif e in from_words:
            new_list.append(to_words[from_words.index(e)])
        elif type(e) == str:
            new_list.append(e)
        else:
            print("Warning: should not be here")
    return new_list

def var_to_lower(l):
    new_list = []
    for e in l:
        if isinstance(e, list):
            new_list.append(var_to_lower(e))
        elif type(e) == str:
            result, lower_var = is_var_name(e, ret_upper=False)
            if result:
                new_list.append(lower_var)
            else:
                new_list.append(e)
        else:
            print("Warning: should not be here")
    return new_list


def flatten_arg_list(l, args):
    new_list = []
    consts = set()
    pred = []

    for e in l:
        is_leaf = True
        for ei in e:
            if type(ei) == list:
                is_leaf = False
                break

        if type(e) == str and not e == ',':
            # new_list.append(e)
            pred.append(e)

        if type(e) == list:
            if len(e) == 0:
                continue

            if is_leaf:
                new_args = []
                for arg in e:
                    if not type(arg) == str:
                        raise InvalidArgException()
                    if arg == ',':
                        continue
                    if arg not in args and arg not in consts:
                        consts.add(arg)
                    new_args.append(arg)
                new_list += new_args

            else:
                try:
                    fl, cs = flatten_arg_list(e, args)
                    new_list.append(fl)
                    for c in cs:
                        consts.add(c)
                except UnseenParamException:
                    # remove the predicates that include unseen params
                    print('Remove unseen missing params')
                except InvalidArgException:
                    print ('invalid arguments')

    if not len(pred) == 0:
        new_list = [' '.join(pred)] + new_list

    return new_list, consts

def flatten_logic_transformation(program):
        # Deal with the edge case that multiple programmatic results sticked together with logic operations
        new_clean_prog = []
        
        for p in program:
            new_program = [ i for i in re.split(' or | and |^or\(|^and\(|;', p) if not len(i) == 0]

            if len(new_program) > 1:
                if new_program[0][0] == '(' and new_program[-1][-1] == ')':
                    new_program[0] = new_program[0][1:]
                    new_program[-1] = new_program[-1][:-1]
                print('here')
            new_clean_prog += (new_program)
            
        return new_clean_prog

def flatten_comma_transformation(program):
    # Deal with the edge case that multiple programmatic results sticked together with tuples

    clean_prog = []
    
    for p in program:
        new_program = re.split('\),', p)

        for pred in new_program[:-1]:
            clean_prog.append(pred + ')')

        clean_prog.append(new_program[-1])

        if 'neq' in p:
            print('here')
            
    return clean_prog

def transform_inline_names(program, arg2const):
    new_program = []
    for clause in program:
        if not(type(clause) == list and clause[0] == 'name'):
            if type(clause) == list:
                new_clause = rec_sub_val(clause, arg2const)
                new_program.append(new_clause)
            else:
                new_program.append(clause)
    return new_program

attributes = ['color', 'material', 'size']
def transfrom_attribute_to_unary(program):
    new_program = []
    for clause in program:
        if type(clause) == list and clause[0] in attributes and len(clause) == 3:
            new_clause = [clause[2], clause[1]]
            new_program.append(new_clause)
        else:
            new_program.append(clause)
    return new_program

def kw_to_ls(key_word):
    key_word_norm = key_word.replace('-', ' ').replace(',', ' ').replace('.', ' ').replace('_', ' ').replace("'", ' ')
    key_word_ls = key_word_norm.split(' ')
    return key_word_ls

def remove_args_in_kw(keyword, args):
    new_kw = []
    missing_args = set()
    key_word_ls = kw_to_ls(keyword)
    for tk in key_word_ls:
        res, var_name = is_var_name(tk, ret_upper=True)
        if res and not var_name in args:
            missing_args.add(var_name)
        elif not tk in args and len(tk) > 0:
            new_kw.append(tk)
    if not len(new_kw) == 0:
        new_kw = '_'.join(new_kw)
    return new_kw, missing_args

def remove_args_in_token(program, args):
    new_missing_args = set()
    if type(program) == str:
        new_kw, missing_args = remove_args_in_kw(program, args)
        return new_kw, missing_args
    
    new_program = []
    for key_word in program:
        new_kw, missing_args = remove_args_in_token(key_word, args)
        # assert not len(new_kw) == 0
        new_program.append(new_kw)
        new_missing_args.update(missing_args)
        
    return new_program, new_missing_args
    
def is_var_name(s, ret_upper=False):
    if len(s) == 2 and s[0] == '?':
        if ret_upper:
            return True, s[1].upper()
        return True, s
    if len(s) == 1 and s.isalpha() and s.isupper():
        if ret_upper:
            return True, s
        return True, '?' + s.lower()
    return False, None

def remove_name_as_binary(program):
    
    new_program = []
    for clause in program:
        if not type(clause) == list:
            new_program.append(clause)
        elif len(clause) > 0 and clause[0] != 'name':
            new_program.append(clause)
            
    return new_program

def is_english(keyword):
    keyword_doc = nlp(keyword)
    if keyword_doc._.language == 'en' and keyword_doc._.language_score >= 0.8:
        return True
    return False

def normalize_kws(program, thres=20):
    
    new_program = []
    for cid, clause in enumerate(program):
        # This is the leaf keyword
        if type(clause) == str:
            
            new_kw = None
            # Skipping the non English strings
            if not clause.isascii():
                if not is_english(clause):
                    return None
                
            # Keyword is too long, use spacy to extract the main noun chunk
            if len(clause) > thres and cid > 0:
                kw_doc = nlp(clause)
                for chunk in kw_doc.noun_chunks:
                    if chunk.root.dep_ == 'ROOT':
                        new_kw = chunk.text
            else:
                new_kw = clause                
            
            if new_kw is None:
                return None
            
            new_kw = normalize_text(new_kw)
            new_program.append(new_kw)
            
        elif type(clause) == list and len(clause) > 0:
            new_clause = normalize_kws(clause, thres)
            if not new_clause is None:
                new_program.append(new_clause)
            else: 
                print(f"Skipping invalid keyword: {clause}")
            
    return new_program
    
    
def validate_variable_names(program):
    var_lookup = {}
    new_program = []
    for clause in program:
        new_clause = []
        for p in clause: 
            # Remove negations
            if p[0] == 'not':
                continue
            
            # Remove ill-formed names
            if p[0] == 'name':
                if not len(p) == 3:
                    continue
                
                key = None
                val = None
                
                for arg in p[1:]:
                    result, var_name = is_var_name(arg)
                    if result:
                        key = var_name
                    else:
                        val = arg
                        
                # This is an invalid name
                if key is None or val is None:
                    continue
                # TODO: improve by select the best fit or merge into the same
                if key in var_lookup:
                    continue
                var_lookup[key] = val
                new_clause.append(p)
            
            else: 
                new_clause.append(p)
                
        new_program.append(new_clause)
            
    return var_lookup, new_program

def find_ordered_pairs(lst):
    ordered_pairs = []
    
    # Use two for loops to create pairs
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            # Add the pair (lst[i], lst[j]) to the result if they are in order
            ordered_pairs.append((lst[i], lst[j]))
    
    return ordered_pairs

def normalize_args_unary_binary_helper(clause, description, var_map):
    # Count how many varibles are there in the clause
    # Ignore the rest, and perform enumeration over the varibles into binary args, while preserve order
    # Connect them with or relationship
    
    new_var_list = []
    new_clauses = []
    predicate = clause[0]
    
    for arg in clause[1:]:
        if arg[0] == '?':
            new_var_list.append(arg)
            
    if len(new_var_list) <= 1:
        print('here')
        raise InvalidArgException
    else:
        new_possible_lst = find_ordered_pairs(new_var_list)
        for pair in new_possible_lst:
            new_clauses.append([predicate, pair[0], pair[1]])
    
    output = []
    if len(new_clauses) == 1:
        output = new_clauses[0]
    else: 
        # TODO: we shall query a language model for what make sense.
        output = ['and'] + new_clauses
    return output
    
def normalize_args_unary_binary(program, description, var_map):
    new_program = []
    for event in program:
        new_event = []
        for p in event:
            if type(p) == str or (len(p) <= 3 and len(p) > 1):
                new_event.append(p)
            else:
                print(f"ignoring {p}")
                # TODO: recover multiple args 
                # new_clause = normalize_args_unary_binary_helper(p, description, var_map)
                # new_event.append(new_clause)
        new_program.append(new_event)
    return new_program
    
gpt_normalize_lookup_table = {
    "sequential_descriptions": "sequential descriptions",
    "video_location": "video location", 
    "location": "video location",
    "event_length": "duration",
    "event length": "duration",
    "time_stamps": "time stamps",
    "video_location": "video location", 
    "video_location_precise": "video location precise",
    "duration_precise": "duration precise"
}

def replace_keys_recursive(data, lookup_table):
    # Check if the input is a dictionary
    if isinstance(data, dict):
        new_data = {}
        for key, value in data.items():
            # Replace the key if it exists in the lookup table
            new_key = lookup_table.get(key, key)
            # Recursively call the function on the value
            new_data[new_key] = replace_keys_recursive(value, lookup_table)
        return new_data
    
    # Check if the input is a list, apply the function recursively to all elements
    elif isinstance(data, list):
        return [replace_keys_recursive(item, lookup_table) for item in data]
    
    # If it's neither a dict nor a list, just return the value itself
    else:
        return data
    
def normalize_key(gpt_result):
    new_gpt_result = replace_keys_recursive(gpt_result, gpt_normalize_lookup_table)
    return new_gpt_result

def collect_const_binary(result, var_map):
    consts = set()
    binaries = set()
    
    for value in var_map.values():
        consts.add(value)
        
    for clause in result:
        
        if type(clause) == str:
            assert clause in bool_token
            
        elif type(clause) == list: 
            
            if clause[0] in bool_token and type(clause[1]) == list:
                sub_consts, sub_binaries = collect_const_binary(clause, var_map)
                consts = consts.union(sub_consts)
                binaries = binaries.union(sub_binaries)
            else:
                collect_binary = False
                if len(clause) == 3 and not clause[0] == 'name':
                    collect_binary = True
                    binary_clause = []
                    binary_clause.append(clause[0])
                    
                for arg in clause[1:]:
                    
                    if arg in var_map and collect_binary:
                        binary_clause.append(var_map[arg])
                    else:
                        if collect_binary:
                            binary_clause.append(arg)
                        if not (len(arg) == 2 and '?' == arg[0]):
                            consts.add(arg)
                
                if collect_binary:
                    binaries.add(tuple(binary_clause))
                    
    return consts, binaries

noun_pos = ['PROPN', 'NOUN', ]
wrong_labeled_noun = ['humanoid', 'hand', 'cricket_player', 'train', 'hands', 'figure', 'boy', 'individual']
def look_back_till_noun(token_ls, current_idx):
    prev_tokens = token_ls[:current_idx]
    for token in reversed(prev_tokens):
        if not (len(token.text)==1) and (token.pos_ in noun_pos or token.text in wrong_labeled_noun): 
            return token.text
    return prev_tokens[-1].text

def extract_missing_var(doc, all_vars, var_map):
    all_vars_target = get_lower_args(all_vars)
    missing_var_target = set(all_vars_target) - set(var_map)
    missing_vars = {v[1].upper(): v for v in missing_var_target}
    
    new_var_map = {}
    if not len(missing_vars) == 0:
        for tid, token in enumerate(doc):
            for missing_var in missing_vars:
                if missing_var in token.text and not tid == 0:
                    new_name = look_back_till_noun(doc, tid)
                    if not new_name is None:
                        new_var_map[missing_vars[missing_var]] = normalize_text(new_name)
        print(f"Extracted new var map: {new_var_map}")
        if len(new_var_map) == 0:
            print('here')
            
    return new_var_map

def get_lower_args(args):
    arg_replace = ['?' + e.lower() for e in args]
    return arg_replace

##########################################################################
##############                Helper Functions              ##############
##########################################################################

class ExpGen():

    def __init__(self, ct=0):
        self.ct = ct

    def get(self):
        self.ct += 1
        return self.ct

    def reset(self):
        self.ct = 0

exp_gen = ExpGen()

def process_logic_pred(logic_ls):

    if logic_ls[0] in bool_token:
        args = []

        # Negative leaf predicates
        if logic_ls[0] == 'not' and len(logic_ls) > 1 and not logic_ls[1][0] in bool_token:
            # Not equivalent
            if logic_ls[1][0] in kw_preds:
                args = logic_ls[1][1:]
                return InEqPred(not_kw_preds[logic_ls[1][0]], args)

            # Negtive predicates
            else:
                args = logic_ls[1][1:]
                return NegPredicate(logic_ls[1][0], args)

        # Ordinary boolean formula
        else:
            for element in logic_ls[1:]:
                b = process_logic_pred(element)
                if not b is None:
                    args.append(b)
            if len(args) > 0:
                return BoolFormula(logic_ls[0], args)
            return None

    # Equvalent ?
    elif logic_ls[0] in kw_preds:
        raise Exception("No direct equlency should be used")
    elif not logic_ls[0][0].isalpha():
        return None

    # Positive predicates
    else:
        if not type(logic_ls) == list:
            return None
            # raise InvalidArgException
        args = logic_ls[1:]
        return PosPredicate(logic_ls[0], args)

##########################################################################
##############           Parser for GPT Caption             ##############
##########################################################################

class GptCaption():

    def __init__(self, caption, gpt_result) -> None:
        self.caption = caption
        self.gpt_result = normalize_key(gpt_result)
        self.consts = set()
        self.binary_predicates = set()
        self.var_map = {}
        self.doc = nlp(clean_cap(self.gpt_result['sequential descriptions']))

        self.parse_from_gpt_json()
        self.process_time_step()
        self.collect_kws()
        self.collect_tuples()
        self.args = list(self.args)
        
    def clean_missing_program(self):
        
        # Check programmatic description exists
        if not ("sequential descriptions" in self.gpt_result or "programmatic" in self.gpt_result):

            # Check if only one single time stamp is recorded
            # If so, rewrite it as a single program
            if  'video_id' in self.gpt_result and \
                ('video_location' in self.gpt_result or 'location' in self.gpt_result) and \
                ('relations' in self.gpt_result or 'attributes' in self.gpt_result) and \
                ('event_length' in self.gpt_result or 'event length' in self.gpt_result):

                new_prog = {}
                new_prog['programmatic'] = []
                if 'relations' in self.gpt_result:
                    new_prog['programmatic'] += self.gpt_result['relations']
                if 'attributes' in self.gpt_result:
                    new_prog['programmatic'] += self.gpt_result['attributes']

                if 'video_location' in self.gpt_result:
                    new_prog['video location'] = self.gpt_result['video_location']
                elif 'location' in self.gpt_result:
                    new_prog['video location'] = self.gpt_result['location']

                if 'event_length' in self.gpt_result:
                    new_prog['duration'] = self.gpt_result['event_length']
                elif 'event length' in self.gpt_result:
                    new_prog['duration'] = self.gpt_result['event length']

                new_prog['description'] = self.caption

                assert not 'time stamps' in self.gpt_result
                self.gpt_result['time stamps'] = {}
                self.gpt_result['time stamps']['1'] = new_prog

                if not "sequential descriptions" in self.gpt_result:
                    self.gpt_result["sequential descriptions"] = self.caption
            else:
                print("No program exists in the gpt output")
                raise InvalidPrediction
            
    def extract_arguments_from_nl(self):
        # tokens = set()
        # if "sequential descriptions" in self.gpt_result:
        #     description = self.gpt_result["sequential descriptions"]
        #     self.args = set()
        #     for sentence in description:
        #         sentence = sentence.replace(",", " ")
        #         sentence = sentence.replace("'s", " ")
        #         [tokens.add(t) for t in sentence.split(" ")]
        # else:
        #     program = self.gpt_result["programmatic"]
        #     self.args = set()
        #     for clause in program:
        #         tokens = re.findall('\(.*\)', clause)
        #         assert len(tokens) == 1
        #         tokens = tokens[0]
        #         tokens = tokens[1: -1].split(',')
        self.args = set()
        for token in self.doc:
            if len(token.text) == 1 and token.text.isalpha() and token.text.isupper():
                self.args.add(token.text)
                    
    def parse_from_gpt_json(self):
        self.clean_missing_program()
        self.extract_arguments_from_nl()
    
    def parse_program(self, description, program):
        #### Check validity
        if len(program) == 0:
            return None

        # Lexing
        program = flatten_logic_transformation(program)
        program = flatten_comma_transformation(program)
        
        #### Wrap and parse the program
        connect_programmatic = '(and(' + ')('.join(program)+ '))'
        connect_programmatic = connect_programmatic.replace('neq(', 'not( = ').replace('_', ' ')
        if ' or ' in connect_programmatic:
            print('here')
        connect_programmatic = connect_programmatic.replace(' or ', 'and')

        argument = pp.Word(pp.alphanums + pp.pyparsing_unicode.Latin1.alphas + '_' + ' '+ '-' + '\'' + '[' + ']') | pp.Word(',')

        try:
            result = nestedExpr('(',')', content=argument).parseString(connect_programmatic).asList()
        except pp.exceptions.ParseException:
            raise InvalidPrediction

        result, _ = flatten_arg_list(result, list(self.args))
        result = var_to_lower(result)
        var_map, result = validate_variable_names(result)
        self.var_map.update(var_map)
        [self.args.add(i[1].upper()) for i in var_map]
        result = normalize_args_unary_binary(result, description, self.var_map)
        result = transform_inline_names(result, self.var_map)
        result = transfrom_attribute_to_unary(result)
        
        
        if not len(result) == 1:
            raise InvalidPrediction

        result = result[0]
        result, missing_args = remove_args_in_token(result, self.args)
        self.args.update(missing_args)
        
        result = remove_name_as_binary(result)
        result = normalize_kws(result, thres=20)
        
        if result is None:
            raise InvalidPrediction
        
        new_result = []

        for predicate in result:
            correct_pred = True
            if type(predicate) == list:
                # Process recursive bool tokens
                if len(predicate) == 2 and type(predicate[1]) == list:
                    if not predicate[0] in bool_token:
                        correct_pred = False
                elif len(predicate) <= 0 or len(predicate) > 3: 
                    correct_pred = False
                else:
                    for arg in predicate:
                        if not type(arg) == str:
                            correct_pred = False
                            
                if correct_pred:
                    new_result.append(predicate)
                else:
                    print(f"Unseen predicate: {predicate}")
                    
            else:
                if predicate in bool_token:
                    new_result.append(predicate)
        
        extra_var_map = extract_missing_var(doc=self.doc, all_vars=self.args, var_map=self.var_map)
        if len(extra_var_map) > 0:
            self.var_map.update(extra_var_map)
        
        
        consts, binaries = collect_const_binary(new_result,  self.var_map)
        for const in consts:
            self.consts.add(const)

        for binary in binaries:
            self.binary_predicates.add(binary)
            
        return new_result

    def process_time_step(self):
        self.time_progs = []
        self.time_duration = []
        self.time_location = []
        self.time_precise_duration = []
        self.time_precise_location = []

        if not 'time stamps' in self.gpt_result:
            raise InvalidPrediction()

        arg2const = {}
        occurred_preds = set()

            
        for time, event in self.gpt_result['time stamps'].items():
            #### Enforce the json structure formatting
            if not ('programmatic' in event and 
                    'duration' in event and 
                    'video location' in event and
                    'video location precise' in event and
                    'duration precise' in event):
                raise InvalidPrediction()

            program = self.parse_program(event['description'], event['programmatic'])
            
            #### Continue if the parsed program is empty
            if program == [] or program is None:
                continue
            
            for i in program:
                if type(i) == list and i[0] == 'name':
                    if not len(i) == 3:
                        continue
                    arg2const[i[1]] = i[2]
                elif type(i) == list:
                    occurred_preds.add(i[0])
            
            time_prog = process_logic_pred(program)
            if time_prog is None:
                continue
            
            extra_var_map = extract_missing_var(doc=self.doc, all_vars=self.args, var_map=self.var_map)
            if len(extra_var_map) > 0:
                self.var_map.update(extra_var_map)
                
            self.time_progs.append(time_prog)
            self.time_location.append(event['video location'])
            self.time_duration.append(event['duration'])
            self.time_precise_duration.append(event['video location precise'])
            self.time_precise_duration.append(event['duration precise'])
        
        # All programs are invalid
        if len(self.time_progs) == 0:
            raise InvalidArgException
            
        self.consts = list(self.consts)
        self.binary_predicates = list(self.binary_predicates)

    def collect_kws(self):
        
        kws = {'unary': set(), 'binary': set()}

        for prog in self.time_progs:
            prog_kws = prog.collect_kws()
            [kws['unary'].add(i) for i in prog_kws['unary']]
            [kws['binary'].add(i) for i in prog_kws['binary']]

        kws['unary'] = list(kws['unary'])
        kws['binary'] = list(kws['binary'])
        
        self.unary_kws = kws['unary']
        self.binary_kws = kws['binary']
    
    def collect_tuples(self):
        self.all_tuples = []
        for prog in self.time_progs:
            prog_tuples = prog.collect_tuples()
            self.all_tuples += prog_tuples
        
    def to_scl(self):
        scl_facts = []
        scl_eids = []

        for prog in self.time_progs:
            eid, facts = prog.to_scl(0, self.consts)
            scl_eids.append(eid)
            scl_facts.append(facts)

        return scl_eids, scl_facts

    def to_str(self):
        scl_prog_strs = []

        #### TODO: Delete the try and catch later
        try:
            for prog in self.time_progs:
                scl_prog_strs.append(f"Finally(Logic({prog.to_str(self.consts)}))")

            if len(self.time_progs) == 0:
                raise InvalidPrediction()

            #  Until(Finally(e1), Until(Finally(e2), Finally(e3)))
            program = reduce(lambda accu, elem: f"Until({elem}, {accu})", reversed(scl_prog_strs))
            return program
        except AttributeError:
            return []
        
    def to_str_windowed(self, window_size=3):
        
        windowed_time_progs = chunk_list(self.time_progs, window_size)
        prog_list = []
        
        for prog_idx, time_progs in windowed_time_progs:
            #### TODO: Delete the try and catch later
            try:
                scl_prog_strs = []
                for prog in time_progs:
                    scl_prog_strs.append(f"Finally(Logic({prog.to_str(self.consts)}))")

                if len(time_progs) == 0:
                    raise InvalidPrediction()

                #  Until(Finally(e1), Until(Finally(e2), Finally(e3)))
                program = reduce(lambda accu, elem: f"Until({elem}, {accu})", reversed(scl_prog_strs))
                prog_list.append((prog_idx, program))
            except AttributeError:
                continue
        
        return prog_list

# def process_gpt_spec(gpt_specs):
#     all_actions = {}
#     for action, spec in gpt_specs.items():

#         prog = GptCaption(action, spec)
#         scl_exp_ids, scl_facts = prog.to_scl()
#         all_actions[action] = {}
#         all_actions[action]['time_stamp_ids'] = scl_exp_ids
#         all_actions[action]['time_stamp_facts'] = scl_facts

#     return all_actions

class GPTSpecPart2:
    def __init__(self, load_cache_paths, store_cache_path):
        self.store_cache_path = store_cache_path
        self.gpt_specs = {}
        for load_cache_path in load_cache_paths:
            file_name, file_extension = os.path.splitext(load_cache_path)
            if file_extension == '.json':
                self.gpt_specs.update(json.load(open(load_cache_path, 'r')))
            elif file_extension == '.jsonl':
                with open(load_cache_path, 'r') as file:
                    for line in file:
                        self.gpt_specs.update(json.loads(line))
        print('here')

    def create_specs(self):
        all_action_scl = {}
        all_action = {}
        total_spec_ct = 0
        wrong_pred_ct = 0
        wrong_arg_ct = 0
        description_length = {}

        # clear cached
        with open(self.store_cache_path, "w") as f:
            pass 

        for action, spec in self.gpt_specs.items():
            # print(action)
            # if not action == 'GZwJPQFW-20':
            #     continue
            
            # if action == 'GZwJPQFW-20':
            #     print("here")
                
            total_spec_ct += 1
            try:
                prog = GptCaption(action, spec)
                prog_string = prog.to_str()
                windowed_prog_str = prog.to_str_windowed()

                if prog_string == []:
                    # print("action:", action)
                    # global_count += 1
                    continue
                
                if windowed_prog_str == []:
                    continue

            except InvalidPrediction:
                wrong_pred_ct += 1
                continue
            except InvalidArgException:
                wrong_arg_ct += 1
                continue

            # new_prog = prog

            # scl_exp_ids, scl_facts = prog.to_scl()
            new_prog = {}
            # all_action_scl[action]['time_stamp_ids'] = scl_exp_ids
            # all_action_scl[action]['time_stamp_facts'] = scl_facts
            new_prog['video_id'] = action
            new_prog['unary_kws'] = prog.unary_kws
            new_prog['binary_kws'] = prog.binary_kws
            new_prog['var_map'] = prog.var_map
            new_prog['tuples'] = prog.all_tuples
            
            new_prog['prog'] = prog_string
            new_prog['windowed_prog'] = windowed_prog_str
            new_prog['consts'] = prog.consts
            new_prog['binary_predicates'] = prog.binary_predicates

            new_prog['args'] = prog.args
            new_prog['caption'] = action
            new_prog['duration'] = [ts_info['duration'] for ts_info in prog.gpt_result['time stamps'].values()]
            new_prog['video location'] = [ts_info['video location'] for ts_info in prog.gpt_result['time stamps'].values()]
            new_prog['duration precise'] = [ts_info['duration precise'] for ts_info in prog.gpt_result['time stamps'].values()]
            new_prog['video location precise'] = [ts_info['video location precise'] for ts_info in prog.gpt_result['time stamps'].values()]
            description_length[action] = [ len(tp.args) for tp in prog.time_progs]
            # json_line = json.dumps(new_prog.copy())
            # with open(self.store_cache_path, 'a') as f:
            #     f.write(json_line + '\n')
            
        # json.dump(all_action_scl, open(self.store_cache_path, 'w'))
        return wrong_pred_ct, wrong_arg_ct