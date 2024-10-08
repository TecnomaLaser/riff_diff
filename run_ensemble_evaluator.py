#!/home/mabr3112/anaconda3/bin/python3.9
# import builtins
from ctypes import util
import logging
import os
import sys
import json
from collections import defaultdict, ChainMap
from time import time
import functools
from itertools import chain
from copy import deepcopy
from glob import glob
import re
from webbrowser import get
from itertools import product
from subprocess import run

# import dependencies
from matplotlib import pyplot as plt
import pandas as pd
import Bio
import Bio.PDB
from Bio.PDB import PDBIO, Structure, Model, Chain
from Bio import SVDSuperimposer
from regex import W
from scipy.spatial.transform import Rotation as R

# import custom modules
sys.path.append("/home/mabr3112/riff_diff/")
import utils.helix_randomization_tools as helix_randomization_tools
from models.riff_diff_models import *
import utils.plotting as plots
import utils.biopython_tools
from utils.rdkit_tools import convert_pdb_to_mol
from utils.obabel_tools import obabel_fileconverter

########### Collecting Unique Elements from DF #######################
def extract_unique_frags_from_dict(input_dict: dict):
    '''
    '''
    t = time()
    rows = list(input_dict.keys())
    cols = list(input_dict[rows[0]].keys())
    dict_array = [[input_dict[row][col] for row in rows] for col in cols]

    unique_dicts = defaultdict(int)
    unique_frags_dict = dict()

    for colname, col in zip(cols, dict_array):
        j = 1
        for i, row in enumerate(col):
            row_dict = json.dumps(row, sort_keys=False)
            if row_dict not in unique_dicts:
                unique_dicts[row_dict] = i
                unique_frags_dict[f"{colname}{str(j)}"] = json.loads(row_dict)
                j += 1

    logging.info(f"Extracted unique fragments in {round(time() - t, 3)} Seconds.")
    return unique_frags_dict

def collect_coords_from_dict(input_dict: dict, coords_key:str="bb_coords", atoms=["N", "CA", "O"]) -> tuple:
    '''
    '''
    return tuple([tuple(input_dict[coords_key]["Nterm"][atom]) for atom in atoms], [tuple(input_dict[coords_key]["Cterm"][atom]) for atom in atoms])

def collect_unique_array_from_df(df: pd.DataFrame) -> np.array:
    '''
    '''
    return [tuple(df[col].unique()) for col in df.columns]

def compile_unique_coords_dict_from_df(df: pd.DataFrame) -> dict:
    '''AAA'''
    t = time()
    # transform df to coords_df
    coords_df = df.applymap(collect_coords_from_dict)
    
    # collect unique coords:
    unique_coords = collect_unique_array_from_df(coords_df)
    
    # assembly of unique coords into a dictionary:
    unique_coords_dict = dict()
    for col, coords_list in zip(df.columns, unique_coords):
        for i, coords in enumerate(coords_list):
            unique_coords_dict[f"{col}{str(i+1)}"] = coords
    print(f"Extracted unique coordinates in {time() - t} Seconds.")
    return unique_coords_dict

def compile_unique_elements_dict_from_df(df: pd.DataFrame, dict_key: str) -> dict:
    '''AAA'''
    # transform df to whatever dictionary value you want to read out:
    elem_df = df.applymap(lambda x: x.get(dict_key))
    
    # collect unique fragments in this df:
    unique_df = collect_unique_array_from_df(elem_df)
    
    # assembly of unique elements into dict:
    unique_elements_d = dict()
    for col, elem_list in zip(df.columns, unique_df):
        for i, element in enumerate(elem_list):
            unique_elements_d[f"{col}{str(i+1)}"] = element
    
    return unique_elements_d

############ Compiling Unique Pairings #######################

def extract_coords(input_dict: dict, atoms=["N", "CA", "O"]) -> tuple:
    '''AAA'''
    coords_nterm = tuple(tuple(input_dict['Nterm'][atm]) for atm in atoms)
    coords_cterm = tuple(tuple(input_dict['Cterm'][atm]) for atm in atoms)
    return coords_nterm, coords_cterm

def key_coords_from_pairing(input_tuple_list: list) -> tuple:
    '''Combine pairing fragments ("A1", "B1") and pairing coords (A1-cterm, B1-nterm)'''
    for input_tuple in input_tuple_list:
        key = (input_tuple[0][0], input_tuple[1][0])
        coords = (input_tuple[0][1][1], input_tuple[1][1][0])
        yield key, coords

def generate_pairs(input_list: list, double_pairs=False):
    result = []
    for i in range(len(input_list) - 1):
        for j in range(i + 1, len(input_list)):
            result.append([input_list[i], input_list[j]])
            if double_pairs:
                result.append([input_list[j], input_list[i]])
    return result

def generate_columnwise_pairs(col_a: list, col_b: list) -> list:
    ''''''
    pairings_list = [[(a, b) for b in col_b] for a in col_a]
    return  pairings_list

def combine_lists(col_a, col_b):
    return list(product(col_a, col_b))

def flatten_list_of_tuples(nested_list):
    return list(chain(*nested_list))

def generate_pairs_from_array(input_array: "list[list]") -> list:
    ''''''
    # sample column pairings
    column_pairings = generate_pairs(input_array, double_pairs=True)
    
    # sample key pairings for each column pairing:
    return flatten_list_of_tuples([combine_lists(a, b) for a, b in column_pairings])

############## Transformation of Coords to Orientations (Input for Model) ##################

def superimpose_triangles(set1, set2, set1_id:int=0, set2_id:int=0):
    '''
    '''
    rot, tran = helix_randomization_tools.get_rotran(set1[set1_id], set2[set2_id])
    #logging.debug("set1:\n", set1)
    #logging.debug("rot: \n", rot)
    #logging.debug("tran: \n", tran)
    return set1@rot + tran

def calc_rotran_for_pairing(pairing: tuple, normalize_translation=False, return_non_rotated_vector=False) -> np.array:
    '''AAA'''
    def normalize_vec(input_vector):
        mag = np.linalg.norm(input_vector)
        norm = input_vector / mag
        return np.append(norm, np.array(mag))
    
    a, b = pairing
    rot, trans = helix_randomization_tools.get_rotran(a, b, return_non_rotated_vector=return_non_rotated_vector)
    rot = R.from_matrix(rot).as_quat()
    
    if normalize_translation: trans = normalize_vec(trans)
    return trans, rot

def load_structure_from_pdbfile(path_to_pdb: str, all_models=False) -> Bio.PDB.Structure:
    '''AAA'''
    pdb_parser = Bio.PDB.PDBParser(QUIET=True)
    if all_models: return pdb_parser.get_structure("pose", path_to_pdb)
    else: return pdb_parser.get_structure("pose", path_to_pdb)[0]

def get_coords(structure: Bio.PDB.Structure, chain: str, res_id: int, atoms:list) -> list:
    '''AAA'''
    atom_list = [structure[chain][(" ", res_id, " ")][atom] for atom in atoms]
    return np.array([atom.coord for atom in atom_list])

def transform_pairing_coords(pairing_coords: tuple, fragment: Bio.PDB.Structure, fragres:int=7):
    '''
    Transform pairing coordinates to match the reference fragment by superimposing residues.
    
    Parameters:
    pairing_coords (tuple): A tuple of coordinates to be transformed.
    fragment (Bio.PDB.Structure): The reference fragment structure.
    res1 (int, optional): The residue number of the first pairing coordinate to copy from the reference fragment. Defaults to 7.
    res2 (int, optional): The residue number of the second pairing coordinate to copy from the reference fragment. Defaults to 1.
    fragres (int, optional): The residue number of the reference fragment to use for superimposition. Defaults to 4.
    
    Returns:
    tuple: A tuple of transformed pairing coordinates.
    '''    
    # load reference fragment and extract coords
    fragres_coords = get_coords(fragment, chain="A", res_id=fragres, atoms=["N", "CA", "O"])
    
    # Copy reference Fragment residue 7 coords as first pairing coords (pairing coordinates are always transformed to origin)
    pairing1_coords = np.copy(fragres_coords)
    
    # superimpose pairing set onto reference fragment residue 7
    pairing_superimposed = superimpose_triangles(np.array(pairing_coords), np.array([fragres_coords]), set1_id=0, set2_id=0)

    # copy reference fragment residue 1 coords as second pairing coords
    pairing2_coords = np.copy(pairing_superimposed[1])

    return tuple([pairing1_coords, pairing2_coords])

#################### Scaling and Normalization of Input Orientations ####################################
def scale_dataset(input_data: np.array, scaling_parameters: dict) -> np.array:
    """
    Scales each column in input_data to unit variance using the scaling_parameters dictionary.
    :param input_data: A numpy array of data.
    :param scaling_parameters: A dictionary containing the scaling parameters for each column.
    :return: A numpy array of scaled data.
    """
    assert len(scaling_parameters.keys()) == input_data.shape[1] #: "Scaling Params must have same number of indeces than input_data has columns!"
    scaled_data = np.copy(input_data)
    for i, (mean, var) in scaling_parameters.items():
        scaled_data[:, i] = (scaled_data[:, i] - mean) / (np.sqrt(var)*2)
    return scaled_data

def violinplot_dataset(input_data) -> np.array:
    '''
    '''
    plt.violinplot((s := [[x[i] for x in input_data] for i in range(input_data.shape[1])]))
    plt.show()
    return s

def add_values_to_array(input_array: np.array, values: np.array) -> np.array:
    '''
    '''
    # repeat array to add the linkers later
    repeated_array = np.repeat(input_array, len(values), axis=0)
    
    # create column with linker lengths
    addcol = np.tile(values, len(input_array))
    
    # add linker lengths to expanded input_array
    added = np.concatenate((repeated_array, addcol[:, np.newaxis]), axis=1)
    return added

#################### Torch ##########################################################################
def prediction_dataloader(ds, device, shuffle=False, batch_size=1024):
    '''Dataloader for running predictions'''
    # dataloader for cpu
    dl_cpu = DataLoader(ds, batch_size=batch_size, shuffle=False)
    
    # dl on device
    dl = DeviceDataLoader(dl_cpu, device)
    print(f"Created Dataloader for device {device}")
    return dl

#################### Prediction Postprocessing ######################################################
def rescale_dataset(input_data: np.array, scaling_parameters: dict) -> np.array:
    """
    """
    #assert len(scaling_parameters.keys()) == input_data.shape[1] #: "Scaling Params must have same number of indeces than input_data has columns!"
    scaled_data = np.copy(input_data)
    mean, var = scaling_parameters
    scaled_data = scaled_data * np.sqrt(var)*2 + mean
    return scaled_data


def add_linker_lengths_to_preds(input_array: np.array, linker_lengths: "list[int]") -> np.array:
    '''asdf'''
    n_linkers = len(linker_lengths)
    reshaped_array = np.reshape(input_array, (int(len(input_array)/n_linkers), n_linkers))
    
    x, n = reshaped_array.shape
    output_array = np.empty(shape=(x,n), dtype=object)
    for i in range(x):
        for j in range(n):
            output_array[i,j] = (reshaped_array[i, j], linker_lengths[j])
    return output_array

####################### Qality Scores ###############################################################
def calc_quality_score(linker_lddt: float, motif_rmsd: float, linker_length: int, distance: float, favor_short_linkers:float=0, plddt_threshold:float=0.7, rmsd_threshold:float=0.15, rmsd_strength:float=1) -> float:
    """
    Computes the quality score of an inpaint using the given parameters.
    """
    def estimate_chainbreak(linker_length:int, distance:float) -> int:
        return 0 if (distance + ((linker_length-1) * 1)) / linker_length > 3.8 else 1
    
    rmsd_score = np.maximum((motif_rmsd-rmsd_threshold), 0) * rmsd_strength
    return estimate_chainbreak(linker_length, distance) * np.maximum(linker_lddt - rmsd_score - plddt_threshold, 0)

def calc_pairings_quality_scores(plddt_pairings_dict, rmsd_pairings_dict, distance_dict, plddt_threshold:float=0.7, rmsd_threshold:float=0.15, rmsd_strength:float=1):
    '''
    '''
    t = time()
    quality_scores_dict = dict()
    pairings = list(plddt_pairings_dict.keys())
    for pairing in pairings:
        plddts = plddt_pairings_dict[pairing]
        rmsds = rmsd_pairings_dict[pairing]
        dist = distance_dict[pairing]
        quality_scores = [(calc_quality_score(plddt[0], rmsd[0], i+1, dist, plddt_threshold=plddt_threshold, rmsd_threshold=rmsd_threshold, rmsd_strength=rmsd_strength), i+1) for i, plddt, rmsd in zip(range(len(plddts)), plddts, rmsds)]
        quality_scores_dict[pairing] = quality_scores
    logging.info(f"Calculated Structure-Quality Scores for {len(pairings)} Pairings in {round(time() - t, 3)} Seconds.")
    return quality_scores_dict

def adjust_qs_for_linker_length(qs_list: "list[tuple]", tolerance:int=2, strength:float=0.5) -> "list[tuple]":
    '''AAA'''
    return [(x[0] - x[0] * (np.maximum(x[1]-tolerance,0)*strength/10), x[1]) for x in qs_list]

def reverse_adjust_qs_for_linker_length(qs_list: "list[tuple]", tolerance:int=2, strength:float=0.5) -> "list[tuple]":
    '''AAA'''
    return [(x[0]/(1 - (np.maximum(x[1]-tolerance,0)*strength/10)), x[1]) for x in qs_list]

########################## Pathsearch ###############################################################
def normalize_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    ''''''
    median = df[col].median()
    std = df[col].std()
    df[f"{col}_normalized"] = (df[col] - median) / std
    return df

def all_paths(graph, path, used, result):
    if len(path) == len(graph):
        result.append(path[:])
        return

    for i in range(len(graph)):
        if i in used:
            continue

        used.add(i)
        path.append(graph[i])
        all_paths(graph, path, used, result)
        path.pop()
        used.remove(i)

def permutations(graph):
    result = []
    all_paths(graph, [], set(), result)
    return result

def convert_paths(path: list):
    return [(x, path[i+1]) for i, x in enumerate(path[:-1])]

def compile_nodes_from_coords(coords_row, conversion_dict: dict) -> list:
    '''AAA'''
    return [(conversion_dict[coord], coord) for coord in coords_row]

def old_create_paths_from_nodes(nodes: "list[tuple]") -> dict:
    '''AAA'''
    return {"-".join([x[0] for x in path]): convert_paths([x[1] for x in path]) for path in permutations(nodes)}

def create_paths_from_nodes(nodes: list) -> dict:
    '''AAA'''
    return {"-".join(path): convert_paths(path) for path in permutations(nodes)}

def calc_path_mean_plddt(path_df):
    ''''''
    path_df["mean_plddt"] = path_df[[x for x in path_df.columns if "plddt" in x]].mean(axis=1)
    return None


def filter_paths(path_df, min_quality:float=0.01, max_linker_length:int=5):
    '''
    '''
    f1_path_df = path_df.loc[path_df[[x for x in path_df.columns if re.search("^[0-9]+_quality$",x)]].min(axis=1) >= min_quality]
    f2_path_df = f1_path_df.loc[f1_path_df[[x for x in f1_path_df.columns if re.search("^[0-9]+_linker$", x)]].max(axis=1) <= max_linker_length]
    return f2_path_df

def sample_from_path_df(df: pd.DataFrame, n: int, random_sample_subset_fraction:float=1.0):
    ''''''
    if random_sample_subset_fraction:
        # take subset (if fraction is 1, subset = full set!!)
        subset_df = df.sort_values(by="rotprob_and_quality", ascending=False).head(np.max([n, int(round(len(df)*random_sample_subset_fraction))])) # if n is lower than subset, take
        sampled_df = subset_df.sample(n)
    else:
        sampled_df = df.sort_values(by="rotprob_and_quality", ascending=False).head(n)
    return sampled_df

######################### PDB-File Reassembly #####################################################
def assemble_pdb(path_series: pd.Series, out_path: str, fragment_dict_dict: dict, input_dir: str, add_ligand=None, add_channel:str=None, auto_superimpose_channel=True):
    '''assembles pdb-files for input to either hallucination of RFdiffusion (recommended).
    Parameters:
        add_ligand, str:  name of the ligand chain. (e.g. Z, or X are most often Ligand Chains.)
    
    '''
    def get_fragments(input_series: pd.Series, fragment_str:str="fragment"):
        return input_series[[x for x in input_series.index if x.startswith(fragment_str)]].to_list()
    
    # sanity
    if not os.path.isdir(out_path): os.makedirs(out_path)
    
    # load fragments from path
    pdb_parser = Bio.PDB.PDBParser(QUIET=True)
    
    # create empty model:
    pose_model = Structure.Structure("pose")
    empty_pose = Model.Model(0)
    pose_model.add(empty_pose)
    pose = pose_model[0]
    
    # go over each fragment and add it into the empty pose. Add chain names alphabetically
    for i, fragment in enumerate(get_fragments(path_series)):
        fragment_dict = fragment_dict_dict[fragment]
        frag_pose = load_structure_from_pdbfile(f"{input_dir}/{fragment_dict['origin']}", all_models=True)[fragment_dict["frag_num"]]["A"]
        frag_pose.detach_parent()
            
        # set name of chains alphabetically
        chain_name = chr(ord('@')+(i+1))
        frag_pose.id = chain_name
        pose.add(frag_pose)

    # add ligand from last fragment, if option is set:
    if add_ligand:
        lig = load_structure_from_pdbfile(add_ligand)[args.ligand_chain]
        lig.detach_parent()
        pose.add(lig)
        
        # adding a channel only works if a ligand is present:
        if add_channel and auto_superimpose_channel:
            pose = utils.biopython_tools.add_polyala_to_pose(pose, polyala_path=add_channel, polyala_chain="Q", ligand_chain=args.ligand_chain)
        elif add_channel:
            pose.add(load_structure_from_pdbfile(add_channel)["Q"])

    # store the pose
    out_pdb_name = f"{out_path}/{path_series.name}.pdb"
    io = PDBIO()
    io.set_structure(pose)
    io.save(out_pdb_name)
    
    return out_pdb_name

############################ Inpainting Contigs Writers ####################################
def get_fragments(input_series: pd.Series) -> list:
    '''Extracts fragments from PD.Series'''
    return input_series[[x for x in input_series.index if x.startswith("fragment")]].to_list()

def get_fragments_contigs(fragments, fragments_dict: dict) -> list:
    '''Extracts fragment contig string from '''
    return [f"{chr(ord('A')+i)}1-{int(fragments_dict[fragment]['frag_length'])}" for i, fragment in enumerate(fragments)]
                                                                                                             
def get_linkers(input_series: pd.Series) -> list:
    '''Collect linkers (linker lengths) from input_series'''
    return [str(l) for l in input_series[[x for x in input_series.index if x.endswith("linker")]].to_list()]

def divide_flanking_residues(residual: int, flanking: str) -> tuple:
    ''''''
    def split_flankers(residual, flanking) -> tuple:
        ''''''
        cterm = residual // 2
        nterm = residual - cterm
        return nterm, cterm

    residual = int(residual)
    if residual < 6 or flanking == "split":
        return split_flankers(residual, flanking)
    elif flanking == "nterm":
        return residual-3, 3
    elif flanking == "cterm":
        return 3, residual-3
    else:
        raise ValueError(f"Paramter <flanking> can only be 'split', 'nterm', or 'cterm'. flanking: {flanking}")

def get_flankers(fragments: list, linkers: list, max_res: int, flanking: str) -> tuple:
    '''Calculate length of terminal flanker regions for inpainting.'''
    def get_frag_length(fragment):
        return int(fragment.split("-")[-1])
    
    # calculate length of all fragments, linkers and then assign flanker length:
    all_frag_res = sum([int(get_frag_length(frag)) for frag in fragments])
    linker_length = sum([int(x) for x in linkers])
    residual = max_res - linker_length - all_frag_res
    return divide_flanking_residues(residual=residual, flanking=flanking)

def compile_contig_string(fragments: list, linkers: list, nterm:str=None, cterm:str=None) -> str:
    '''Generate contig string for inpainting (RFDesign)'''
    fl_list = [[fragment, linker] for fragment, linker in zip(fragments[:-1], linkers)]
    fl_list = [item for sublist in fl_list for item in sublist]
    contig_str = f'{nterm},{",".join(fl_list)},{fragments[-1]},{cterm}'
    return contig_str

def compile_inpaint_seq(fragments, fragment_dict):
    '''Writes inpaint_seq string for inpainting'''
    inpaint_str_l = list()
    for i, frag in enumerate(fragments):
        # remove Catalytic residue from inpaint_seq:
        res_ids = list(range(1, fragment_dict[frag]["frag_length"]+1))
        res_ids.remove(int(fragment_dict[frag]["res_num"]))
        frag_str = ",".join([f"{chr(ord('A')+i)}{res}" for res in res_ids])
        inpaint_str_l.append(frag_str)
        
    return ",".join(inpaint_str_l)

def compile_translate_str(fragments: "list[str]", fragment_dict: dict) -> str:
    '''AAA'''
    translate_str_l = list()
    for i, frag in enumerate(fragments):
        length = fragment_dict[frag]["frag_length"]
        translate_str_l.append(f"{chr(ord('A')+i)}1-{length},translate_sampling_magnitude")
    return ":".join(translate_str_l)

def compile_rotate_str(fragments: "list[str]", fragment_dict: dict) -> str:
    '''AAA'''
    rotate_str_l = list()
    for i, frag in enumerate(fragments):
        length = fragment_dict[frag]["frag_length"]
        rotate_str_l.append(f"{chr(ord('A')+i)}1-{length},rotate_sampling_degrees")
    return ":".join(rotate_str_l)

def compile_inpaint_pose_opts(input_series: pd.Series, fragment_dict: dict, max_length=74, flanking="split") -> str:
    '''Compile and write pose options for inpainting'''
    # Calculate fragment_contigs, linkers and flankers for inpainting contig:
    fragments = get_fragments(input_series)
    fragment_contigs = get_fragments_contigs(fragments, fragment_dict)
    linkers = get_linkers(input_series)
    nterm, cterm = get_flankers(fragment_contigs, linkers, max_length, flanking=flanking)
    
    # compile contig_string and inpaint_seq string:
    contig_str = compile_contig_string(fragment_contigs, linkers, nterm, cterm)
    inpaint_seq_str = compile_inpaint_seq(fragments, fragment_dict)
    translate_str = compile_translate_str(fragments, fragment_dict)
    rotate_str = compile_rotate_str(fragments, fragment_dict)
    
    # combine into pose_options string:
    return f"--contigs {contig_str} --inpaint_seq {inpaint_seq_str} --tie_translate {translate_str} --block_rotate {rotate_str}"

def get_force_aa(fragments: list, fragment_dict: dict) -> str:
    ''''''
    aa_3to1_dict = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}
    return ",".join([f"{chr(ord('A')+i)}{fragment_dict[fragment]['res_num']}{aa_3to1_dict[fragment_dict[fragment]['identity']]}" for i, fragment in enumerate(fragments)])

def compile_hallucination_pose_opts(input_series: pd.Series, fragment_dict: dict, max_length=74, flanking="split"):
    '''Compile and write pose options for hallucination'''
    # calc fragment contigs, linkers and flankers for contig:
    fragments = get_fragments(input_series)
    fragment_contigs = get_fragments_contigs(fragments, fragment_dict)
    linkers = get_linkers(input_series)
    nterm, cterm = get_flankers(fragment_contigs, linkers, max_length, flanking=flanking)

    # compile force_aa_str:
    contig_str = compile_contig_string(fragment_contigs, linkers, nterm, cterm)
    force_aa_str = get_force_aa(fragments, fragment_dict)

    return f"--mask {contig_str} --force_aa {force_aa_str}"

def find_ligand_name(fragment, fragment_dict: dict, input_dir: str, ligand_chain="Z") -> str:
    '''returns name of Ligand in fragment'''
    fragment_d = fragment_dict[fragment]
    frag_pose = load_structure_from_pdbfile(f"{input_dir}/{fragment_d['origin']}", all_models=True)[fragment_d["frag_num"]][ligand_chain]
    return [x for x in frag_pose.get_residues()][0].get_resname()

def compile_rfdiffusion_pose_opts(input_series: pd.Series, fragment_dict: dict, input_dir: str, max_length=74, flanking="split", ligand_chain="Z", add_channel:str=None):
    '''Compile and write pose opts for RFdiffusion'''
    fragments = get_fragments(input_series)
    fragment_contigs = get_fragments_contigs(fragments, fragment_dict)
    linkers = get_linkers(input_series)
    nterm, cterm = get_flankers(fragment_contigs, linkers, max_length, flanking=flanking)

    # compile inpaint_seq and contig str:
    contig_str = compile_contig_string(fragment_contigs, linkers, nterm, cterm).replace(",", "/")
    inpaint_seq = compile_inpaint_seq(fragments, fragment_dict).replace(",", "/")

    # find ligand name:
    lig_name = find_ligand_name(fragments[-1], fragment_dict=fragment_dict, input_dir=input_dir, ligand_chain=ligand_chain)

    return f"'contigmap.contigs=[{contig_str}]' 'contigmap.inpaint_seq=[{inpaint_seq}]' potentials.substrate={lig_name}"

def get_covalent_bonds(input_series: pd.DataFrame, fragments_dict: dict, input_dir: str) -> str:
    ''''''
    def get_lig_info(fragment_d: dict, input_dir: str) -> "tuple[int,str]":
        frag_pose = load_structure_from_pdbfile(f"{input_dir}/{fragment_d['origin']}", all_models=True)[fragment_d["frag_num"]][fragment_d['fragment_picking_info']['ligand_chain']]
        lig_res = [x for x in frag_pose.get_residues()][0]
        return lig_res.id[1], lig_res.get_resname()
    
    def compile_covalent_bond_str(i_d: dict, fragment_index: int, lig_resnum: int, lig_name: str) -> str:
        cov_bond_list = [x.split(":") for x in i_d["fragment_picking_info"]["covalent_bond"].split(",") if x]
        chain = chr(ord('A')+fragment_index)
        return ','.join([f"{i_d['res_num']}{chain}_{i_d['identity']}_{cov_bond[0]}:{str(lig_resnum)}{i_d['fragment_picking_info']['ligand_chain']}_{lig_name}_{cov_bond[1]}" for cov_bond in cov_bond_list])

    # collect all fragments of fragment ensemble (row)
    extracted_frag_dicts = [fragments_dict[fragment] for fragment in get_fragments(input_series)]

    # collect lig info for cov_bond str:
    lig_resnum, lig_name = get_lig_info(extracted_frag_dicts[0], input_dir=input_dir)

    # find all covalent bonds in all fragments and compile them into a list:
    cov_bonds = list()
    for i, frag_d in enumerate(extracted_frag_dicts):
        if "covalent_bond" in frag_d["fragment_picking_info"]:
            cov_bonds.append(compile_covalent_bond_str(frag_d, fragment_index=i, lig_resnum=1, lig_name=lig_name)) # changed lig_resnum to 1, because it is always changed to 1 by molfile_to_params.py

    return ",".join(cov_bonds)

def write_inpaint_contigs_to_json(input_df: pd.DataFrame, json_path: str, fragment_dict: dict, max_length: int) -> dict:
    '''AAA'''
    out_dict = {index: compile_inpaint_pose_opts(input_df.loc[index], fragment_dict=fragment_dict, max_length=max_length, flanking="split") for index in input_df.index}
    with open(json_path, 'w') as f:
        json.dump(out_dict, f)
    return out_dict

def get_motif_res(input_series: pd.Series, fragment_dict: dict) -> dict:
    '''Get Motif for fixed residues from series.'''
    return {f"{chr(ord('A')+(i))}": list(range(1, fragment_dict[fragment]["frag_length"]+1)) for i, fragment in enumerate(get_fragments(input_series))}

def get_fixed_res(input_series: pd.Series, fragment_dict: dict) -> dict:
    '''Get Motif for motif residues from series.'''
    return {f"{chr(ord('A')+(i))}": [fragment_dict[fragment]["res_num"]] for i, fragment in enumerate(get_fragments(input_series))}

def get_res_identity(input_series: pd.Series, fragment_dict: dict) -> dict:
    ''''''
    return {f"{chr(ord('A')+(i))}{fragment_dict[fragment]['res_num']}": fragment_dict[fragment]["identity"] for i, fragment in enumerate(get_fragments(input_series))}

def write_fixedres_to_json(input_df: pd.DataFrame, fragments_dict: dict, json_path: str) -> dict:
    '''AAA'''
    out_dict = {index: get_fixed_res(input_df.loc[index], fragments_dict) for index in input_df.index}
    with open(json_path, 'w') as f:
        json.dump(out_dict, f)
    return out_dict

def write_motif_res_to_json(input_df: pd.DataFrame, fragments_dict: dict, json_path: str) -> dict:
    '''AAA'''
    out_dict = {index: get_motif_res(input_df.loc[index], fragments_dict) for index in input_df.index}
    with open(json_path, 'w') as f:
        json.dump(out_dict, f)
    return out_dict

def write_residue_identities_to_json(input_df: pd.DataFrame, fragments_dict: dict, json_path: str) -> dict:
    '''AAA'''
    out_dict = {index: get_res_identity(input_df.loc[index], fragments_dict) for index in input_df.index}
    with open(json_path, 'w') as f:
        json.dump(out_dict, f)
    return out_dict

def parse_atom_identifier(atom_identifier: str) -> tuple[str]:
    '''parses an atom identifier PDB-chain, PDB-resnum, PDB-atom-name [A5CO]'''
    pattern = r"([a-zA-Z])(\d+)([a-zA-Z]+)"
    result = re.match(pattern, atom_identifier)
    return result.group(1), result.group(2), result.group(3)

def get_custom_center_from_theozyme(input_theozyme, atom_identifier: str):
    '''Collects coordinates of custom center for RFdiffusion given an atom identifier (PDB chain name, resnum and atom name).
    Example atom identifier PDB-chain (A), PDB-resnum (5), PDB-atom-name (CO): "A5CO"'''
    chain, resnum, atom_name = parse_atom_identifier(atom_identifier)
    tz = load_structure_from_pdbfile(input_theozyme)
    try: 
        return ",".join([str(x) for x in tz[chain][(" ", int(resnum), " ")][atom_name].coord])
    except KeyError:
        raise KeyError(f"Atom with identifiers chain={chain}, resnum={resnum}, atom_name={atom_name}, was not found in the theozyme at {input_theozyme}")
    

def main(args):
    ### Code ####
    if not os.path.isdir(args.output_dir): os.makedirs(args.output_dir, exist_ok=True)

    if os.environ.get('SLURM_SUBMIT_DIR'):
        script_dir = "/home/mabr3112/riff_diff/"
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    scaling_params_file = f"{script_dir}/models/scaling_params_10k2.json"
    path_to_fragment = f"{script_dir}/utils/helix.pdb"
    
    if args.debug: logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename=f"{args.output_dir}/log.txt")
    else: logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=f"{args.output_dir}/log.txt")

    # load input data
    logging.info(f"Loading fragments from ensemble file {args.input_json}")
    with open(args.input_json, 'r') as f:
        input_dict = json.loads(f.read())

    input_dir = "/".join(args.input_json.split("/")[:-1])

    # look for how many chains are in the dictionary
    chains = list(input_dict["0"].keys())
    logging.info(f"Found {len(chains)} chains in input fragment ensemble: {', '.join(chains)}. Beginning collection of unique fragment coords.")

    ################### Collect unique pairings between Fragments ######################################
    # compile list of unique fragments:
    extracted_fragments_dictlist = [{f"{fchain}{input_dict[row][fchain]['frag_num']}": extract_coords(input_dict[row][fchain]['bb_coords']) for row in input_dict} for fchain in chains]
    logging.info(f"Extracted unique fragments. Found: {', '.join([fchain + ': ' + str(num) for fchain, num in zip(chains, [len(x) for x in extracted_fragments_dictlist])])}")

    # change dictionary into nested list:
    extracted_fragments_array = [[tuple([key, coords]) for key, coords in fragdict.items()] for fragdict in extracted_fragments_dictlist]

    # generate pairings:
    pairings_array = generate_pairs_from_array(extracted_fragments_array)

    # compile pairings array into dictionary (Only use proper C-term and N-term coordinates of fragments)
    pairings_dict = {key: coords for key, coords in key_coords_from_pairing(pairings_array)}
    logging.info(f"Extracted {len(pairings_dict)} unique pairings from fragment ensemble.")

    # collect unique fragments into central dictionary:
    def extract_cols_from_dict(in_dict, cols_list: list) -> dict:
        return {k: v for k, v in in_dict.items() if k in cols_list}

    out_cols = ['identity', 'origin', 'res_num', 'rot_prob', "frag_length", "frag_num", "fragment_picking_info"]
    unique_fragments_dict = {f"{fchain}{input_dict[row][fchain]['frag_num']}": extract_cols_from_dict(input_dict[row][fchain], out_cols) for row in input_dict for fchain in chains}
    logging.info(f"Extracted data of {len(unique_fragments_dict)} unique fragments.")

    ################## PREPROCESSING OF PAIRING ORIENTATIONS ########################
    fragment = load_structure_from_pdbfile(path_to_fragment)

    # transform pairing coords to fragment center for prediction.
    logging.info(f"Tranforming Pairing coordinates onto coordinate system of pLDDT predictor.")
    pairings_dict_transformed = {key: transform_pairing_coords(pairing_coords, fragment, fragres=7) for key, pairing_coords in pairings_dict.items()}

    # calculate translation and rotation from pairing
    logging.info(f"Calculating Pairing Geometry (rotation and translation).")
    pairings_geom_dict = {key: np.concatenate(calc_rotran_for_pairing(pairing_coords, normalize_translation=True, return_non_rotated_vector=True)) for key, pairing_coords in pairings_dict_transformed.items()}

    # extract distances for pairings into pairings_distance_dict:
    pairings_distance_dict = {pairings_dict[key]: geom[3] for key, geom in pairings_geom_dict.items()}
    keys_pairings_distance_dict = {key: geom[3] for key, geom in pairings_geom_dict.items()}

    ### scale translation and rotation by scaling parameters used for network training (data preprocessing)
    # read in scaling_params
    logging.info(f"Scaling Pairings geometries for prediction.")
    with open(scaling_params_file, 'r') as f:
        scaling_params = json.loads(f.read())

    # extract scaling params for inputs
    scaling_params_inputs = {int(k): scaling_params[k] for k in list(scaling_params.keys())[:9]}

    # extract pairings geometries
    input_geoms_raw = np.array(list(pairings_geom_dict.values()))

    # add linker lengths (will expand the array)
    logging.info(f"Adding Linker lengths to inputs")
    input_geoms_wll = add_values_to_array(input_geoms_raw, [(x+1)/10 for x in range(10)])

    # now scale input geometries:
    logging.info(f"Scaling input Geometries")
    input_geoms = scale_dataset(input_geoms_wll, scaling_parameters=scaling_params_inputs)

    ################## PREDICTION OF PAIRING-PLDDTs #################################
    # Run Predictions!
    device = get_default_device()
    cpu = torch.device("cpu")
    cuda = torch.device("cuda:0")
    device = cpu

    # load model
    model_path = f"{script_dir}/models/model_plddt_march.pth"
    model = Model_FNN_BN(layers=[9, 512, 512, 512, 512, 64, 1], device=device, act_function=F.gelu, dropout=0.17)
    model.load_state_dict(torch.load(model_path, map_location=cpu))
    model.eval()
    to_device(model, device)
    model.to(device)

    # convert inputs to tensor
    inputs = to_device(torch.from_numpy(input_geoms), device)

    # create outputs
    t = time()

    with torch.no_grad():
        inputs_dl = prediction_dataloader(inputs, device, batch_size=1024)
        batch_outs = list()
        for batch in inputs_dl:
            batch_outs.append(model(batch))
        outputs = torch.cat(batch_outs, dim=0)

    logging.info(f"Prediction of {len(inputs)} pLDDTs on {device} finished in {round(time() - t, 3)} Seconds.")

    ################## pLDDT-Prediction Postprocessing ###################################
    # rescale prediction to plddt range (0,1)
    preds = outputs.cpu().numpy()
    preds = preds.reshape((preds.shape[0],))
    preds_rescaled = rescale_dataset(preds, scaling_params["9"])
    #plt.violinplot(preds_rescaled)
    #plt.show()

    # add linkers to raw outputs:
    preds_with_linkers = add_linker_lengths_to_preds(preds_rescaled, linker_lengths=range(1, 11))

    # add linkers to raw outputs:
    preds_with_linkers = add_linker_lengths_to_preds(preds_rescaled, linker_lengths=range(1, 11))

    # Recombine with pairings:
    logging.info(f"Collecting pLDDT predictions for {len(preds_with_linkers)} orientations.")
    keys_pairings_dict_full = {key: prediction for key, prediction in zip(list(pairings_dict.keys()), preds_with_linkers)}

    ################## PREDICTION OF PAIRING-RMSDs #################################
    # Run Predictions!
    device = get_default_device()
    cpu = torch.device("cpu")
    device = cpu

    # load model
    rmsd_model_path = f"{script_dir}/models/model_rmsd_march.pth"
    rmsd_model = Model_FNN_BN(layers=[9, 512, 512, 512, 128, 64, 1], device=device, act_function=F.gelu, dropout=0.17)
    rmsd_model.load_state_dict(torch.load(rmsd_model_path, map_location=cpu))
    rmsd_model.eval()
    to_device(rmsd_model, device)
    rmsd_model.to(device)

    # convert inputs to tensor
    inputs = to_device(torch.from_numpy(input_geoms), device)

    # create outputs
    t = time()
    with torch.no_grad():
        inputs_dl = prediction_dataloader(inputs, device, batch_size=1024)
        batch_outs = list()
        for batch in inputs_dl:
            batch_outs.append(rmsd_model(batch))
        rmsd_outputs = torch.cat(batch_outs, dim=0)

    logging.info(f"Prediction of {len(inputs)} RMSDs on {device} finished in {round(time() - t, 3)} Seconds.")

    ################## RMSD-Prediction Postprocessing ###################################
    # rescale prediction to plddt range (0,1)
    rmsd_preds = rmsd_outputs.cpu().numpy()
    rmsd_preds = rmsd_preds.reshape((rmsd_preds.shape[0],))
    rmsds_rescaled = rescale_dataset(rmsd_preds, scaling_params["10"])
    #plt.violinplot(rmsds_rescaled)
    #plt.show()

    # add linkers to raw outputs:
    rmsds_with_linkers = add_linker_lengths_to_preds(rmsds_rescaled, linker_lengths=range(1, 11))

    # Recombine with pairings:
    rmsd_keys_pairings_dict_full = {key: prediction for key, prediction in zip(list(pairings_dict.keys()), rmsds_with_linkers)}

    ###################### Calculation of Structure Quality from RMSD and pLDDT #####################
    # calculate quality scores
    keys_quality_scores_dict = calc_pairings_quality_scores(keys_pairings_dict_full, rmsd_keys_pairings_dict_full, keys_pairings_distance_dict, plddt_threshold=args.plddt_threshold, rmsd_threshold=args.rmsd_threshold, rmsd_strength=args.rmsd_strength)

    # if option is set, factor in short_linker preference:
    keys_quality_scores_dict = {key: adjust_qs_for_linker_length(qs_list, tolerance=1, strength=args.short_linker_preference) for key, qs_list in keys_quality_scores_dict.items()}

    # take top quality scores:
    keys_quality_scores_dict_top = {key: max(qs, key=lambda x: x[0]) for key, qs in keys_quality_scores_dict.items()}
    quality_scores_dict_top = {pairings_dict[key]: qs for key, qs in keys_quality_scores_dict_top.items()}

    # readjust if option adjust is set:
    keys_quality_scores_dict_top = {key: reverse_adjust_qs_for_linker_length([val], tolerance=1, strength=args.short_linker_preference)[0] for key, val in keys_quality_scores_dict_top.items()}

    ##################### PATHSEARCH ###########################################################
    ## Compile Pairings Dictionary and DataFrame for all possible paths through all residues
    def compile_nodes_from_dict(row: dict) -> list:
        '''AAA'''
        return [f"{key}{row[key]['frag_num']}" for key in row]

    # collect nodes for paths:
    logging.info(f"Compiling paths from {len(input_dict)} initial ensembles.")
    pathdict_list = list()
    for row in input_dict:
        nodes = compile_nodes_from_dict(input_dict[row])
        paths_dict = create_paths_from_nodes(nodes)
        pathdict_list.append(paths_dict)

    # flatten pathdict_list into one dictionary:
    pathdict = {k: v for d in pathdict_list for k, v in d.items()}
    logging.info(f"Created {len(pathdict)} paths from input data.")

    # Apply predicted pairings-pLDDT to pathdict ### (list(sum(, ())) flattens a list very quickly)
    logging.info(f"Assigning precalculated quality scores and distances to paths.")
    pathdict_quality_scores = {key: list(sum([keys_quality_scores_dict_top[pairing] for pairing in pathdict[key]], ())) for key in pathdict}
    pathdict_distances = {key: [keys_pairings_distance_dict[pairing] for pairing in pathdict[key]] for key in pathdict}

    ################### Processing Path DataFrame #################################################
    def compile_path_df_cols(in_list):
        '''AAA'''
        return list(sum([[f"{i}_quality", f"{i}_linker"] for i in range(int(len(in_list)/2))], []))

    columns=compile_path_df_cols(pathdict_quality_scores[list(pathdict_quality_scores)[0]])

    # Collect scored pathdict into DataFrame
    logging.info(f"Collecting scores into DataFrame")
    path_plddt_df = pd.DataFrame.from_dict(pathdict_quality_scores, orient="index", columns=columns)
    path_distance_df = pd.DataFrame.from_dict(pathdict_distances, orient="index", columns=[f"{i}_distance" for i, _ in enumerate(list(pathdict_distances.values())[0])])
    path_plddt_df = pd.concat([path_plddt_df, path_distance_df], axis=1)

    # Calc mean_quality from quality scores over all fragments:
    path_plddt_df["mean_quality"] = path_plddt_df[[col for col in path_plddt_df.columns if col.endswith("_quality")]].mean(axis=1)

    # Read out Rotamer probability and identity from Fragments:
    logging.info(f"Reading Rotamer Probabilites into Path DataFrame.")
    path_fragment_df = pd.DataFrame.from_dict({x: x.split("-") for x in list(path_plddt_df.index)}, orient="index")
    path_identity_df = path_fragment_df.applymap(lambda x: unique_fragments_dict[x]["identity"]).add_prefix("identity_")
    path_rotprob_df = path_fragment_df.applymap(lambda x: unique_fragments_dict[x]["rot_prob"]).add_prefix("rot_prob_")
    path_rotprob_df["mean_rotprob"] = path_rotprob_df.mean(axis=1)

    # Concatenate with path_df:
    full_path_df = pd.concat([path_plddt_df, path_fragment_df.add_prefix("fragment_"), path_identity_df, path_rotprob_df], axis=1)
    
    # calculate stats for plotting:
    full_path_df["mean_distance"] = full_path_df[[col for col in full_path_df.columns if "distance" in col]].mean(axis=1)
    full_path_df["mean_linker_length"] = full_path_df[[col for col in full_path_df.columns if col.endswith("_linker")]].mean(axis=1)

    # normalize and combine rotamer probability with structure quality for filtering:
    full_path_df = normalize_col(full_path_df, "mean_rotprob")
    full_path_df = normalize_col(full_path_df, "mean_quality")
    full_path_df["rotprob_and_quality"] = full_path_df["mean_rotprob_normalized"] * args.weight_rotprob + full_path_df["mean_quality_normalized"] * args.weight_structure_quality
    full_path_df["mean_quality_adjusted"] = full_path_df["mean_quality"] + args.plddt_threshold

    # Remove Paths from Path DataFrame that have a quality score of 0, ore have linker distance and linker length above the specified threshold
    logging.debug(f"length before distance filtering: {len(full_path_df)}")
    path_plddt_df = full_path_df[full_path_df[[col for col in full_path_df.columns if col.endswith("distance")]].max(axis=1) < args.max_linker_distance]
    logging.debug(f"length after distance filtering: {len(full_path_df)}")
    logging.info(f"Filtered {len(full_path_df) - len(path_plddt_df)} paths with linker distance higher than {args.max_linker_distance}")
    top_path_df = filter_paths(path_plddt_df, min_quality=0.0000000000001, max_linker_length=args.max_linker_length)
    logging.debug(f"length after quality score filtering: {len(full_path_df)}")
    logging.info(f"Filtered {len(path_plddt_df)-len(top_path_df)} paths with quality scores of ~0")

    # Extract top (args.max_num) rows from the filtered Path DataFrame.
    selected_path_df = sample_from_path_df(top_path_df, n=args.max_num, random_sample_subset_fraction=args.sample_from_subset_fraction)
    logging.debug(f"length after sampling: {len(full_path_df)}")
    logging.info(f"Selected {len(selected_path_df)} paths from Path DataFrame.")

    ##################### Plotting #########################################################
    if not os.path.isdir((plotdir := f"{args.output_dir}/plots")): os.makedirs(plotdir, exist_ok=True)

    # plot predicted plddts and predicted RMSDs:
    dat = [preds_rescaled, rmsds_rescaled]
    titles = ["Predicted pLDDTs", "Predicted RMSDs"]
    labels = ["pLDDT [AU]", "RMSD [\u00C5]"]
    dims = [(0,1), (0,1)] 
    plotpath = f"{plotdir}/predictions.png"
    logging.info(f"Plotting predicted plddts and rmsds at {plotpath}")
    plots.violinplot_multiple_lists(lists=dat, titles=titles, y_labels=labels, dims=dims, out_path=plotpath)

    # plot all paths vs. paths with quality score >0 vs. selected paths:
    plotpath = f"{plotdir}/filter_stats.png"
    logging.info(f"Plotting statistics of filtered paths at {plotpath}")
    dfs = [full_path_df, top_path_df, selected_path_df]
    df_names = ["All Paths", "Paths with Quality > 0", "Selected Paths"]
    cols = ["mean_quality", "mean_rotprob", "mean_distance", "mean_linker_length"]
    col_names = ["Structure Quality", "Rotamer Probability", "Average Distance", "Average Linker Length"]
    y_labels = ["Quality Score [AU]", "Probability [%]", "Distance [\u00C5]", "Number of Residues"]
    dims = [(0, 1), (0, 100), None, None]
    _ = plots.violinplot_multiple_cols_dfs(dfs, df_names=df_names, cols=cols, titles=col_names, y_labels=y_labels, dims=dims, out_path=plotpath)

    ##################### PDB-FILE Reassembly ##############################################
    # sanity
    pdb_dir = f"{args.output_dir}/pdb_in/"
    if not os.path.isdir(pdb_dir): os.makedirs(pdb_dir, exist_ok=True)

    # store Ligand in separate folder for diffusion.
    os.makedirs((lig_folder := f"{args.output_dir}/ligand/"), exist_ok=True)
    x_pdb_path = glob(f"{input_dir}/*.pdb")[0]
    x_pdb = load_structure_from_pdbfile(x_pdb_path)
    ligand = x_pdb[args.ligand_chain]
    ligand_pdbfile = utils.biopython_tools.store_pose(ligand, (lig_path:=f"{lig_folder}/LG1.pdb"))
    substrate_name = [x for x in x_pdb[args.ligand_chain].get_residues()][0].get_resname()

    # create Rosetta .params file if ligand contains more than 2 atoms. (Ligands with less atoms will not be considered explicitily for Rosetta Design!) (#TODO: create rotamer library too).
    if len(list(ligand.get_atoms())) > 2:
        # store ligand as .mol file for rosetta .molfile-to-params.py
        logging.info(f"Running 'molfile_to_params.py' to generate params file for Rosetta.")
        lig_molfile = obabel_fileconverter(input_file=lig_path, output_file=lig_path.replace(".pdb", ".mol2"), input_format="pdb", output_format=".mol2")
        run(f"python3 {script_dir}/rosetta/molfile_to_params.py -n {substrate_name} -p {lig_folder}/LG1 {lig_molfile} --keep-names --clobber --chain={args.ligand_chain}", shell=True, stdout=True, check=True, stderr=True)
        lig_path = f"{lig_folder}/LG1_0001.pdb"
    else:
        logging.info(f"Ligand at {ligand_pdbfile} contains less than 3 atoms. No Rosetta Params file can be written for it.")

    # Store pdb-files of reassembled Fragments at <out_path>
    logging.info(f"Generating PDB-files for top {len(selected_path_df)} fragment ensembles at {pdb_dir}")
    pdb_files = [assemble_pdb(selected_path_df.loc[index], out_path=pdb_dir, input_dir=input_dir, fragment_dict_dict=unique_fragments_dict, add_ligand=lig_path, add_channel=args.add_channel, auto_superimpose_channel=args.auto_superimpose_channel) for index in selected_path_df.index]

    # write contigs for inpainting
    inpaint_contigs_path = f"{args.output_dir}/inpaint_pose_opts.json" # output path to inpaint contigs file
    logging.info(f"Writing inpaint pose options to {inpaint_contigs_path}")
    contigs_dict = write_inpaint_contigs_to_json(selected_path_df, inpaint_contigs_path, fragment_dict=unique_fragments_dict, max_length=args.pdb_length)

    # compile motif_residues, fixed_residues and catres_identities for inpainting:
    logging.info(f"Compiling motif_residues, fixed_residues and catres_identities for inpainting.")
    selected_fragments = selected_path_df.index
    output_datadict = {
        "fixed_residues": {index: get_fixed_res(selected_path_df.loc[index], unique_fragments_dict) for index in selected_fragments},
        "motif_residues": {index: get_motif_res(selected_path_df.loc[index], unique_fragments_dict) for index in selected_fragments},
        "catres_identities" : {index: get_res_identity(selected_path_df.loc[index], unique_fragments_dict) for index in selected_fragments},
        "hallucination_pose_opts": {index: compile_hallucination_pose_opts(selected_path_df.loc[index], unique_fragments_dict, max_length=args.pdb_length) for index in selected_fragments},
        "rfdiffusion_pose_opts": {index: compile_rfdiffusion_pose_opts(selected_path_df.loc[index], unique_fragments_dict, input_dir=input_dir, max_length=args.pdb_length, ligand_chain=args.ligand_chain) for index in selected_fragments},
        "inpainting_pose_opts": contigs_dict,
        "covalent_bonds": {index: get_covalent_bonds(selected_path_df.loc[index], unique_fragments_dict, input_dir=input_dir) for index in selected_fragments}
    }
    
    selected_path_df = selected_path_df.join([pd.DataFrame.from_dict(output_datadict)])

    # if custom center was set, add option to DataFrame:
    if args.custom_diffusion_center:
        # when coordinates are given, just copy coordinates
        if len(args.custom_diffusion_center.split(",")) == 3:
            selected_path_df["diffusion_custom_center"] = [args.custom_diffusion_center for row in selected_path_df.index]

        # otherwise extract coordinates from specified atom by -atom_specifier-
        else:
            center_str = get_custom_center_from_theozyme(args.input_theozyme, args.custom_diffusion_center)
            selected_path_df["diffusion_custom_center"] = [center_str for row in selected_path_df.index]

    # store selected paths DataFrame
    scores_path = f"{args.output_dir}/selected_paths.json"
    logging.info(f"Storing selected path scores at {scores_path}")
    selected_path_df.to_json(scores_path)

    # Finish
    logging.info(f"Done")

if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--input_json", type=str, required=True, help="File containing the .json input file located in the same directory as the .pdb files that are linked to it. (Output Directory of Fragment Generation)")
    argparser.add_argument("--input_theozyme", type=str, required=True, help="Path to the theozyme .pdb file that was used as input for the backbone rotamer finder.")
    argparser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory where the .pdb files and options for inpainting should be written to.")
    
    # PDB Options
    argparser.add_argument("--max_num", type=int, default=50, help="Number of pdb-files that will be created by pathsearch.")
    argparser.add_argument("--pdb_length", type=int, default=200, help="Maximum length of the pdb-files that will be inpainted.")
    argparser.add_argument("--ligand_chain", type=str, default="Z", help="PDB letter for Ligand chain. The ligand will be used as substrate during diffusion.")
    argparser.add_argument("--add_channel", type=str, default="/home/mabr3112/riff_diff/utils/helix_cone_long.pdb", help="If specified, adds the structure specified to the fragment to be used as a 'substrate channel' during diffusion. IMPORTANT!!!  Channel pdb-chain name has to be 'Q' ")
    argparser.add_argument("--auto_superimpose_channel", type=str, default="True", help="Set to false, if you want to copy the channel pdb-chain from the reference file without superimposing on moitf-substrate centroid axis.")
    argparser.add_argument("--custom_diffusion_center", type=str, default=None, help="Chain name, PDB residue-number, and atom name of atom that should be used as a custom center of mass during rfdiffusion.")

    # Filter Options
    argparser.add_argument("--max_linker_length", type=int, default=10, help="Maximum length of linkers that the fragments should be connected with.")
    argparser.add_argument("--short_linker_preference", type=float, default=0.01, help="Strength for how much short linkers should be upweighted in scoring.")
    argparser.add_argument("--max_linker_distance", type=float, default=50, help="Maximum Distance that the linker should have.")
    argparser.add_argument("--weight_rotprob", type=float, default=3, help="Strength of the rotamer probability weight for filtering")
    argparser.add_argument("--weight_structure_quality", type=float, default=0.1, help="Strength of the structure quality weight for filtering")
    argparser.add_argument("--sample_from_subset_fraction", type=float, default=0.5, help="Take random sample from top <subset> rows of DataFrame.")

    # Structure Quality
    argparser.add_argument("--plddt_threshold", type=float, default=0.0, help="Threshold for filtering inpaint plddts during quality-score calculation")
    argparser.add_argument("--rmsd_threshold", type=float, default=15, help="Threshold tolerance for RMSD during quality-score calculation")
    argparser.add_argument("--rmsd_strength", type=float, default=1, help="Strength for RMSD filtering if RMSD is above the Threshold. For quality-score calculation")

    # logging:
    argparser.add_argument("--debug", default=None, help="Use Logging configuration DEBUG")

    args = argparser.parse_args()
    if args.auto_superimpose_channel in ["false", "False", 0]: args.auto_superimpose_channel = False
    print(args.auto_superimpose_channel)

    if args.sample_from_subset_fraction > 1: raise ValueError(f"random_sample_subset_fraction can only be value between 0 and 1!")
    main(args)
