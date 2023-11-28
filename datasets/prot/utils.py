
import numpy as np

import os
from Bio import SeqIO
import torch

from goatools.anno.gaf_reader import GafReader


from dataclasses_json import dataclass_json
from dataclasses import dataclass


from sklearn.preprocessing import LabelEncoder

from goatools.base import get_godag
go = get_godag('go-basic.obo', optional_attrs='relationship')
EMB_LAYER = 33

@dataclass_json
@dataclass

class ProtSample:
    input_seq: torch.Tensor
    annot: torch.Tensor
    entry: str

def get_ancestor_dict(file_path):
    """ Returns dictionary of ancestors for given annotation"""
    file = open(file_path, 'r')
    filereader = file.readlines()
    ancestor_dict = {}
    for i, rows in enumerate(filereader):
        if i > 0:
            rows = rows.split('|')
            if i > 0:
                ancestor_dict[rows[0]] = eval(rows[1])
    return ancestor_dict

def get_level(annots, level):
    """ Returns annotations in annots from the given level """
    return {annot for annot in annots if go[annot].level == level}

def select_single(annots):
    return annots.pop()

def select_annot(annots, ancestors, level=5):
    """ Select annotations of a given level from propagation of annots """
    propagated = annots | ancestors
    propagated_level = get_level(propagated, level)
    return select_single(propagated_level)

def get_term_frequency(root, reader):
    """ Returns dictionary of term frequency for each GO term in the dataset """
    term_frequency = {}
    fasta = SeqIO.parse(open(os.path.join(root,"uniprot_sprot.fasta")), 'fasta')

    for i in fasta:
        entry = i.id.split("|")[1]
        try:
            annots = reader[entry]
        except:
            continue   
            
        if (len(annots) == 0): continue 
        else:  
            for a in annots:
                term_frequency[a] = term_frequency.get(a, 0) + 1

    return term_frequency, max(term_frequency.values())

def select_annot_via_ic(annots, term_frequency, max_freq):
    # lower is more informative
    annots_with_freq = [(term_frequency.get(a, max_freq+1), a) for a in annots]
    # sort by frequency
    annots_with_freq.sort()
    return annots_with_freq[0][1]

def get_samples_using_ic(root):
    samples = []
    fasta = SeqIO.parse(open(os.path.join(root,"uniprot_sprot.fasta")), 'fasta')
    reader = GafReader(os.path.join(root,"filtered_goa_uniprot_all_noiea.gaf")).read_gaf()
    adict = get_ancestor_dict(os.path.join(root,"sprot_ancestors.txt"))

    term_frequency, max_freq = get_term_frequency(root, reader)

    for i in fasta:
        entry = i.id.split("|")[1]
        try:
            annots = reader[entry]
        except:
            continue   
            
        if (len(annots) == 0): continue 
        else:
            try:  
                ancestors = set()
                for a in annots:
                    ancestors |= adict[a] | {a}
                ancestors = list(ancestors)
                annot = select_annot_via_ic(ancestors, term_frequency, max_freq)
                samples.append(ProtSample(
                    input_seq = get_embedding(os.path.join(root, "embeds"), entry),
                    annot= annot,
                    entry= entry
                ))
            except:
                continue
    return samples

def get_samples(root, level = 5):
    """ preprocess samples for cryptic with annotations from a given level """
    samples = []
    fasta = SeqIO.parse(open(os.path.join(root,"uniprot_sprot.fasta")), 'fasta')
    reader = GafReader(os.path.join(root,"filtered_goa_uniprot_all_noiea.gaf")).read_gaf()
    adict = get_ancestor_dict(os.path.join(root,"sprot_ancestors.txt"))
    for i in fasta:
        entry = i.id.split("|")[1]
        try:
            annots = reader[entry]
        except:
            continue   
           
        if (len(annots) == 0): continue 
        else:  
            try:
                ancestors = set()
                for a in annots:
                    ancestors |= adict[a]
                annot = select_annot(annots, ancestors, level)
                samples.append(ProtSample(
                    input_seq = get_embedding(os.path.join(root, "embeds"), entry),
                    annot= annot,
                    entry= entry
                ))
            except:
                continue
    return samples

def check_min_samples(samples, min_samples):
    """ reduces samples to samples from classes with > min_sample datapoints """
    count_dict = get_annot_counts(samples)
    rm_annots = {annot for annot in count_dict if count_dict[annot] < min_samples}
    return [sample for sample in samples if sample.annot not in rm_annots]

def get_mode_ids(samples, train_test = 0.9, train_val = 0.8):
    """ split annotations from sample list to train, val, test 
        returns dictionary of list of annotations for each mode
    """
    count_dict = get_annot_counts(samples)
    sorted_annots = sorted(count_dict)
    train_val = int(train_test*train_val*len(sorted_annots))
    train_test = int(train_test*len(sorted_annots))
    return {
            'train': sorted_annots[0:train_val], 
            'val': sorted_annots[train_val:train_test], 
            'test': sorted_annots[train_test:]
    }

def get_ids(samples):
    """ gets set of all annotations from a list of samples """
    annots = set()
    for sample in samples:
        annots.add(sample.annot)
    return annots

def get_annot_counts(samples):
    """ Returns a dictionary of the number of times an annotation appears in samples 
        where keys are annotations in samples
    """
    counts = {}
    for i in range(len(samples)):
        sample = samples[i]
        try:
            counts[sample.annot] += 1
        except:
            counts[sample.annot] = 1
    return counts

def get_embedding(emb_path, entry):
    """ Returns the ESM embedding for a protein entry """
    fn = f'{emb_path}/{entry}.pt'
    embs = torch.load(fn)
    emb = embs['mean_representations'][EMB_LAYER]
    return emb
    
def encodings(root, level = 5):
    """ Returns dictionary of label encodings of annotations"""
    adict = get_ancestor_dict(os.path.join(root,"sprot_ancestors.txt"))
    all_annots = get_level(set(adict.keys()), level)
    for key, value in adict.items():
        all_annots |= value
    le = LabelEncoder()
    all_annots = np.array(list(all_annots))
    encoded = le.fit_transform(all_annots)
    encodings = {}
    for annot, encode in zip(all_annots, encoded):
        encodings[annot] = encode
    return encodings

def correct_pred(entry, annotation, entry_dict):
    """ Returns true if annotation is valid for a protein sequence entry 
        given entry dict mapping possible annotations to entry name """
    if annotation in entry_dict[entry]: return True
    return False

def get_entry_dict(root, level = 5):
    """ Returns dictionary that maps all possible annotations of an entry to the entry name """
    entry_dict = {}
    fasta = SeqIO.parse(open(os.path.join(root,"uniprot_sprot.fasta")), 'fasta')
    reader = GafReader(os.path.join(root,"filtered_goa_uniprot_all_noiea.gaf")).read_gaf()
    adict = get_ancestor_dict(os.path.join(root,"sprot_ancestors.txt"))
    for i in fasta:
        entry = i.id.split("|")[1]
        try:
            annots = reader[entry]
        except:
            continue   
        
        if len(annots) == 0: 
            continue 
        try:
            ancestors = set()
            for a in annots:
                ancestors |= adict[a]
            propagated = annots | ancestors
            propagated_level = get_level(propagated, level)
            entry_dict[entry] = propagated_level
        except:
            continue
    return entry_dict