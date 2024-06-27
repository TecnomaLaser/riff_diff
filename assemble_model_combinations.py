#!/home/mabr3112/anaconda3/bin/python3.9

import os
import sys
import json
import functools
import itertools
import copy
import logging
from pathlib import Path
import time
import Bio
from Bio.PDB import *
import pandas as pd
import numpy as np
from openbabel import openbabel
import shutil




# import custom modules
sys.path.append("/home/tripp/riff_diff/")
sys.path += ["/home/mabr3112/projects/iterative_refinement/"]
import utils.adrian_utils as myutils
import utils.biopython_tools
import utils.plotting as plots
#no idea why, but I cant import it
#import utils.obabel_tools
#from utils.obabel_tools import obabel_fileconverter

from iterative_refinement import *




def obabel_fileconverter(input_file: str, output_file:str=None, input_format:str="pdb", output_format:str="mol2") -> None:
    '''converts pdbfile to mol2-file.'''
    # Create an Open Babel molecule object
    mol = openbabel.OBMol()

    # Read the PDB file
    obConversion = openbabel.OBConversion()
    obConversion.SetInFormat(input_format)
    obConversion.ReadFile(mol, input_file)

    # Convert the molecule to the desired output format (Mol2 file)
    obConversion.SetOutFormat(output_format)
    obConversion.WriteFile(mol, output_file or input_file)
    return output_file or input_file


def identify_rotamer_by_bfactor_probability(entity):
    '''
    returns the residue number where bfactor > 0, since this is where the rotamer probability was saved
    '''
    residue = None
    for atom in entity.get_atoms():
        if atom.bfactor > 0:
            residue = atom.get_parent()
            break
    if not residue:
        raise RuntimeError('Could not find any rotamer in chain. Maybe rotamer probability was set to 0?')
    resnum = residue.id[1]
    return resnum

def distance_detection(entity1, entity2, bb_only:bool=True, ligand:bool=False, clash_detection_vdw_multiplier:float=1.0, database:str='database', resnum:int=None, covalent_bonds:str=None):
    '''
    checks for clashes by comparing VanderWaals radii. If clashes with ligand should be detected, set ligand to true. Ligand chain must be added as second entity.
    bb_only: only detect backbone clashes between to proteins or a protein and a ligand.
    clash_detection_vdw_multiplier: multiply Van der Waals radii with this value to set clash detection limits higher/lower
    database: path to database directory
    '''
    backbone_atoms = ['CA', 'C', 'N', 'O', 'H']
    vdw_radii = import_vdw_radii(database)
    if bb_only == True and ligand == False:
        entity1_atoms = (atom for atom in entity1.get_atoms() if atom.name in backbone_atoms)
        entity2_atoms = (atom for atom in entity2.get_atoms() if atom.name in backbone_atoms)
    elif bb_only == True and ligand == True:
        entity1_atoms = (atom for atom in entity1.get_atoms() if atom.name in backbone_atoms)
        entity2_atoms = (atom for atom in entity2.get_atoms())
    else:
        entity1_atoms = (atom for atom in entity1.get_atoms())
        entity2_atoms = (atom for atom in entity2.get_atoms())
    for atom_combination in itertools.product(entity1_atoms, entity2_atoms):
        #skip clash detection for covalent bonds
        covalent = False
        if resnum and covalent_bonds:
            for cov_bond in covalent_bonds.split(','):
                if atom_combination[0].get_parent().id[1] == resnum and atom_combination[0].name == cov_bond.split(':')[0] and atom_combination[1].name == cov_bond.split(':')[1]:
                    covalent = True
        if covalent == True:
            continue
        distance = atom_combination[0] - atom_combination[1]
        element1 = atom_combination[0].element
        element2 = atom_combination[1].element
        clash_detection_limit = clash_detection_vdw_multiplier * (vdw_radii[str(element1)] + vdw_radii[str(element2)])

        if distance < clash_detection_limit:
            return True
    return False

def extract_backbone_coordinates(residue):
    bb_atoms = [atom for atom in residue.get_atoms() if atom.id in ['N', 'CA', 'C', 'O']]
    coord_dict = {}
    for atom in bb_atoms:
        coord_dict[atom.id] = tuple(round(float(coord), 3) for coord in atom.get_coord())
    return coord_dict

def extract_chi_angles(residue):
    '''
    residue has to be converted to internal coords first! (on chain/model/structure level)
    '''
    chi1 = float('nan')
    chi2 = float('nan')
    chi3 = float('nan')
    chi4 = float('nan')
    resname = residue.get_resname()
    if resname in AAs_up_to_chi1() + AAs_up_to_chi2() + AAs_up_to_chi3() + AAs_up_to_chi4():
        chi1 = round(residue.internal_coord.get_angle("chi1"), 1)
    if resname in AAs_up_to_chi2() + AAs_up_to_chi3() + AAs_up_to_chi4():
        chi2 = round(residue.internal_coord.get_angle("chi2"), 1)
    if resname in AAs_up_to_chi3() + AAs_up_to_chi4():
        chi3 = round(residue.internal_coord.get_angle("chi3"), 1)
    if resname in AAs_up_to_chi4():
        chi4 = round(residue.internal_coord.get_angle("chi4"), 1)
    return {"chi1": chi1, "chi2": chi2, "chi3": chi3, "chi4": chi4}



def import_vdw_radii(database_dir):
    '''
    from https://en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page), accessed 30.1.2023
    '''
    vdw_radii = pd.read_csv(f'{database_dir}/vdw_radii.csv')
    vdw_radii.drop(['name', 'atomic_number', 'empirical', 'Calculated', 'Covalent(sb)', 'Covalent(tb)', 'Metallic'], axis=1, inplace=True)
    vdw_radii.dropna(subset=['VdW_radius'], inplace=True)
    vdw_radii['VdW_radius'] = vdw_radii['VdW_radius'] / 100
    vdw_radii = vdw_radii.set_index('element')['VdW_radius'].to_dict()
    return vdw_radii

def AAs_up_to_chi1():
    AAs = ['CYS', 'SER', 'THR', 'VAL']
    return AAs

def AAs_up_to_chi2():
    AAs = ['ASP', 'ASN', 'HIS', 'ILE', 'LEU', 'PHE', 'PRO', 'TRP', 'TYR']
    return AAs

def AAs_up_to_chi3():
    AAs = ['GLN', 'GLU', 'MET']
    return AAs

def AAs_up_to_chi4():
    AAs = ['ARG', 'LYS']
    return AAs

def log_and_print(string: str): 
    logging.info(string)
    print(string)
    return string

def normalize_col(df:pd.DataFrame, col:str, scale:bool=False, output_col_name:str=None) -> pd.DataFrame:
    ''''''
    median = df[col].median()
    std = df[col].std()
    if not output_col_name:
        output_col_name = f"{col}_normalized"
    if df[col].nunique() == 1:
        df[output_col_name] = 0
        return df
    df[output_col_name] = (df[col] - median) / std
    if scale == True:
        df = scale_col(df=df, col=output_col_name, inplace=True)
    return df

def scale_col(df:pd.DataFrame, col:str, inplace=False) -> pd.DataFrame:
    #scale column to values between 0 and 1
    factor = df[col].max() - df[col].min()
    df[f"{col}_scaled"] = df[col] / factor
    df[f"{col}_scaled"] = df[f"{col}_scaled"] + (1 - df[f"{col}_scaled"].max())
    if inplace == True:
        df[col] = df[f"{col}_scaled"]
        df.drop(f"{col}_scaled", axis=1, inplace=True)
    return df


def combine_normalized_scores(df: pd.DataFrame, name:str, scoreterms:list, weights:list, normalize:bool=False, scale:bool=False):
    if not len(scoreterms) == len(weights):
        raise RuntimeError(f"Number of scoreterms ({len(scoreterms)}) and weights ({len(weights)}) must be equal!")
    df[name] = sum([df[col]*weight for col, weight in zip(scoreterms, weights)]) / sum(weights)
    df[name] = df[name] / df[name].max()
    if normalize == True:
        df = normalize_col(df, name, False)
        df.drop(name, axis=1, inplace=True)
        df.rename(columns={f'{name}_normalized': name}, inplace=True)
    if scale == True:
        df = scale_col(df, name, True)
    return df

def run_masterv2(poses, output_dir, chains, rmsdCut:float, topN:int=None, minN:int=None, gapLen:int=None, outType:str='match', master_dir:str="/home/tripp/MASTER-v2-masterlib/bin/", database:str="/home/tripp/MASTER-v2-masterlib/master_db/list", max_array_size:int=320):

    """
    for each input ensembles, identifies how many times this ensemble occurs in the the database (= there is a substructure below the rmsd cutoff). if no rmsd cutoff is provided, will automatically set one according to ensemble size)
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/pds_files", exist_ok=True)
    os.makedirs(f"{output_dir}/matches", exist_ok=True)

    scorefile = os.path.join(output_dir, f'master_scorefile.json')
    if os.path.isfile(scorefile):
        log_and_print(f"Found existing scorefile at {scorefile}. Skipping step.")
        return pd.read_json(scorefile)

    sbatch_options = ["-c1", f'-e {output_dir}/create_pds.err -o {output_dir}/create_pds.out']

    cmds = [f"{master_dir}createPDS --type query --pdb {pose} --pds {output_dir}/pds_files/{Path(pose).stem}.pds" for pose in poses]
    top_cmds = [cmds[x:x+200] for x in range(0, len(cmds), 200)]
    top_cmds = ["; ".join(cmds) for cmds in top_cmds]
    
    sbatch_array_jobstarter(cmds=top_cmds, sbatch_options=sbatch_options, jobname="create_pds", max_array_size=max_array_size, wait=True, remove_cmdfile=False, cmdfile_dir=output_dir)

    pdsfiles = [f"{output_dir}/pds_files/{Path(pose).stem}.pds" for pose in poses]

    cmds = [f"{master_dir}/master --query {pds} --targetList {database} --outType {outType} --rmsdCut {rmsdCut} --matchOut {output_dir}/matches/{Path(pds).stem}.match" for pds in pdsfiles]
    top_cmds = [cmds[x:x+5] for x in range(0, len(cmds), 5)]
    top_cmds = ["; ".join(cmds) for cmds in top_cmds]

    matchfiles = [f"{output_dir}/matches/{Path(pds).stem}.match" for pds in pdsfiles]
    
    if topN:
        cmds = [cmd + f" --max {topN}" for cmd in cmds]
    if minN:
        cmds = [cmd + f" --min {minN}" for cmd in cmds]
    if gapLen:
        cmds = [cmd + f" --gapConst {gapLen}" for cmd in cmds]

    sbatch_options = ["-c1", f'-e {output_dir}/MASTER.err -o {output_dir}/MASTER.out']

    sbatch_array_jobstarter(cmds=top_cmds, sbatch_options=sbatch_options, jobname="MASTER", max_array_size=max_array_size, wait=True, remove_cmdfile=False, cmdfile_dir=output_dir)

    combinations = [''.join(perm) for perm in itertools.permutations(chains)]
    match_dict = {}
    out_df = []
    for match in matchfiles:
        for i in combinations:
            match_dict[i] = {'path_num_matches': 0, 'rmsds': []}
            ensemble_rmsds = []
        with open(match, 'r') as m:
            lines = m.readlines()
            for line in lines:
                order = assign_chain_letters(line)
                if not order == None:
                    match_dict[order]['path_num_matches'] += 1
                    rmsd = float(line.split()[0])
                    match_dict[order]['rmsds'].append(rmsd)
                    ensemble_rmsds.append(rmsd)

        ensemble_matches = sum([match_dict[i]['path_num_matches'] for i in combinations])
        if ensemble_matches > 0:
            mean_ensemble_rmsd = sum(ensemble_rmsds) / len(ensemble_rmsds)
            min_ensemble_rmsd = min(ensemble_rmsds)
        else:
            mean_ensemble_rmsd = None
            min_ensemble_rmsd = None

        df = pd.DataFrame({'description': [Path(match).stem for i in combinations], 'path': combinations, 'ensemble_num_matches': [ensemble_matches for i in combinations], 'path_num_matches': [match_dict[i]['path_num_matches'] for i in combinations], 'mean_match_rmsd': [mean_ensemble_rmsd for i in combinations], 'min_match_rmsd': [min_ensemble_rmsd for i in combinations]})
        out_df.append(df)
    
    out_df = pd.concat(out_df).reset_index(drop=True)
    out_df.to_json(scorefile)
    return out_df

def add_terminal_coordinates_to_df(df, Ntermres, Ctermres):
    bb_atoms = ['N', 'CA', 'C', 'O']

    for atom in bb_atoms:
        df[f'Nterm_{atom}_x'] = round(float(Ntermres[atom].get_coord()[0]), 3)
        df[f'Nterm_{atom}_y'] = round(float(Ntermres[atom].get_coord()[1]), 3)
        df[f'Nterm_{atom}_z'] = round(float(Ntermres[atom].get_coord()[2]), 3)
        df[f'Cterm_{atom}_x'] = round(float(Ctermres[atom].get_coord()[0]), 3)
        df[f'Cterm_{atom}_y'] = round(float(Ctermres[atom].get_coord()[1]), 3)
        df[f'Cterm_{atom}_z'] = round(float(Ctermres[atom].get_coord()[2]), 3)
    return df

def create_bbcoord_dict(series):
    bb_atoms = ['N', 'CA', 'C', 'O']
    bb_dict = {'Nterm': {}, 'Cterm': {}}
    for atom in bb_atoms:
        bb_dict['Nterm'][atom] = (series[f'Nterm_{atom}_x'], series[f'Nterm_{atom}_y'], series[f'Nterm_{atom}_z'])
        bb_dict['Cterm'][atom] = (series[f'Cterm_{atom}_x'], series[f'Cterm_{atom}_y'], series[f'Cterm_{atom}_z'])
    return bb_dict


def run_clash_detection(combinations, num_combs, max_array_size, directory, bb_multiplier, sc_multiplier, database, script_path):
    '''
    combinations: iterator object that contains pd.Series
    max_num: maximum number of ensembles per slurm task
    directory: output directory
    bb_multiplier: multiplier for clash detection only considering backbone clashes
    sc_multiplier: multiplier for clash detection, considering sc-sc and bb-sc clashes
    database: directory of riffdiff database
    script_path: path to clash_detection.py script 
    '''
    ens_json_dir = os.path.join(directory, 'ensemble_pkls')
    scores_json_dir = os.path.join(directory, 'scores')
    os.makedirs(ens_json_dir, exist_ok=True)
    out_pkl = os.path.join(directory, 'clash_detection_scores.pkl')
    if os.path.isfile(out_pkl):
        log_and_print(f'Found existing scorefile at {out_pkl}. Skipping step!')

        out_df = pd.read_pickle(out_pkl)
        return out_df

    max_num = int(num_combs / max_array_size)
    if max_num < 10000:
        max_num = 10000

    ensemble_num = 0
    ensemble_nums_toplist = []
    ensemble_names = []
    score_names = []
    ensembles_toplist = []
    ensembles_list = []

    for comb in combinations:
        if ensemble_num % max_num == 0:
            ensembles_list = []
            ensembles_toplist.append(ensembles_list)
            ensemble_nums = []
            ensemble_nums_toplist.append(ensemble_nums)
        for series in comb:
            ensemble_nums.append(ensemble_num)
            ensembles_list.append(series)
        ensemble_num += 1
        

    in_df = []
    count = 0
    log_and_print(f'Writing pickles...')
    for ensembles_list, ensemble_nums in zip(ensembles_toplist, ensemble_nums_toplist):
        df = pd.DataFrame(ensembles_list).reset_index(drop=True)
        df['ensemble_num'] = ensemble_nums
        in_name = os.path.join(ens_json_dir, f'ensembles_{count}.pkl')
        out_name = os.path.join(scores_json_dir, f'ensembles_{count}.pkl')
        ensemble_names.append(in_name)
        score_names.append(out_name)
        df[['ensemble_num', 'poses', 'model_num', 'chain_id']].to_pickle(in_name)
        in_df.append(df)
        count += 1
    log_and_print(f'Done writing pickles!')
    
    in_df = pd.concat(in_df).reset_index(drop=True)
    
    sbatch_options = ["-c1", f'-e {directory}/clash_detection.err -o {directory}/clash_detection.out']
    cmds = [f"{script_path} --pkl {json} --working_dir {directory} --bb_multiplier {bb_multiplier} --sc_multiplier {sc_multiplier} --output_prefix {str(index)} --database_dir {database}" for index, json in enumerate(ensemble_names)]
    
    log_and_print(f'Distributing clash detection to cluster...')
    sbatch_array_jobstarter(cmds=cmds, sbatch_options=sbatch_options, jobname="clash_detection", max_array_size=320, wait=True, remove_cmdfile=False, cmdfile_dir=directory)

    log_and_print(f'Reading in clash pickles...')    
    out_df = []
    for file in score_names:
        out_df.append(pd.read_pickle(file))
    
    
    #delete input pkls because this folder probably takes a lot of space
    shutil.rmtree(ens_json_dir)
    shutil.rmtree(scores_json_dir)

    out_df = pd.concat(out_df)
    log_and_print(f'Merging with original dataframe...')
    out_df = out_df.merge(in_df, on=['ensemble_num', 'poses', 'model_num', 'chain_id']).reset_index(drop=True)
    log_and_print(f'Writing output pickle...')
    out_df.to_pickle(out_pkl)
    log_and_print(f'Clash check completed.')

    return out_df

def auto_determine_rmsd_cutoff(ensemble_size):
    '''
    calcutes rmsd cutoff based on ensemble size. equation is set so that ensemble size of 10 residues returns cutoff of 1.4 A and ensemble size of 26 residues returns cutoff of 2 A. rmsd cutoff cannot be higher than 2 A (to prevent excessive runtimes)
    parameters determined for 4 disjointed fragments of equal length, most likely lower cutoffs can be used when using 3 or less disjointed fragments
    '''
    rmsd_cutoff = 0.0375 * ensemble_size + 1.025
    if rmsd_cutoff > 2:
        rmsd_cutoff = 2
    return round(rmsd_cutoff, 2)


def create_path_name(df, order):

    name = []
    for chain in order:
        model_num = df[df['chain_id'] == chain]['model_num'].to_list()[0]
        name.append(f"{chain}{model_num}")
    name = "-".join(name)
    return name

def assign_chain_letters(line):
    line_sep = line.split()
    cleaned_list = [int(s.replace('[', '').replace(',', '').replace(']', '')) for s in line_sep[2:]]
    sorted_list = sorted(cleaned_list)
    new_list = [sorted_list.index(value) for value in cleaned_list]

    result = ''.join([chr(65 + index) for index in new_list])
    return result

def create_covalent_bonds(df:pd.DataFrame):
    covalent_bonds = []
    for index, series in df.iterrows():
        if not series['covalent_bond'] == None:
            covalent_bond = series['covalent_bond'].split(':')
            covalent_bond = f"{series['rotamer_pos']}{series['chain_id']}_{series['catres_identities']}_{covalent_bond[0]}:1Z_{series['ligand_name']}_{covalent_bond[1]}"
            covalent_bonds.append(covalent_bond)
    if len(covalent_bonds) >= 1:
        covalent_bonds = ",".join(covalent_bonds)
    else:
        covalent_bonds = ""
    return covalent_bonds


def compile_contig_string(series:pd.Series):
    nterm = 20
    cterm = 20
    total_length = 200
    num_gaps = len([chain for chain in series['path_order']]) - 1
    frag_length = sum([len(series['motif_residues'][chain]) for chain in series['path_order']])
    #gap_length = int((total_length - nterm - cterm - frag_length) / num_gaps)
    gap_length = 10
    contigs = [f"{chain}{series['motif_residues'][chain][0]}-{series['motif_residues'][chain][-1]}" for chain in sorted(series['path_order'])]
    contigs = "/".join([f"{contig}/{gap_length}" for contig in contigs[:-1]]) + f"/{contigs[-1]}"
    contig_string = f"'contigmap.contigs=[{nterm}/{contigs}/{cterm}]'"
    return contig_string

def compile_inpaint_string(series:pd.Series):
    chains = sorted(series['path_order'])
    inpaint = "/".join([f"{chain}{pos}" for chain in chains for pos in series['motif_residues'][chain] if not pos == series['fixed_residues'][chain][0]])
    inpaint_string = f"'contigmap.inpaint_seq=[{inpaint}]'"
    return inpaint_string

def compile_rfdiff_pose_opts(series:pd.Series):
    #TODO: at the moment only the first ligand is included in the rfdiffusion run, it crashes when both are added :(
    #pose_opts = f"{compile_contig_string(series)} {compile_inpaint_string(series)} potentials.substrate=[{series['ligand_name'].split(',')[0]}]"
    if len(series['ligand_name'].split(',')) > 1:
        pose_opts = f"{compile_contig_string(series)} {compile_inpaint_string(series)} potentials.substrate=LIG"
    else:
        pose_opts = f"{compile_contig_string(series)} {compile_inpaint_string(series)} potentials.substrate={series['ligand_name']}"
    return pose_opts

def create_path_pdb(series:pd.Series, output_dir, ligand, channel_exists, add_channel, auto_superimpose_channel):
    #chains_models = series['path_name'].split('-')
    chains = sorted(series['path_order'])
    struct = Structure.Structure('out')
    model = Model.Model(0)
    frag_chains = [myutils.import_structure_from_pdb(series['pose_paths'][chain])[series['model_num'][chain]][series['original_chain_id'][chain]] for chain in chains]
    for fragment, chain in zip(frag_chains, chains):
        fragment.id = chain
        model.add(fragment)
    if len([res for res in ligand.get_residues()]) > 1:
        ligands = copy.deepcopy(ligand)
        for lig in ligands.get_residues():
            lig.resname = 'LIG'
        model.add(ligands)
    else:
        model.add(ligand)
    struct.add(model)
    # adding a channel only works if a ligand is present:
    if channel_exists == True and auto_superimpose_channel == True:
        model = utils.biopython_tools.add_polyala_to_pose(model, polyala_path=add_channel, polyala_chain="Q", ligand_chain='Z')
    elif channel_exists == True:
        model.add(myutils.import_structure_from_pdb(add_channel)[0]["Q"])
    output_path = os.path.join(output_dir, f"{series['path_name']}.pdb")
    myutils.write_multimodel_structure_to_pdb(struct, output_path)
    return output_path
    
def combine_ligands(ligand_chain):
    for lig in ligand_chain.get_residues():
        lig.name = 'LIG'
    return ligand_chain
    


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def create_path_series(path_df, path_order, ligands, ligand_name, pdb_dir, channel_exists, add_channel, auto_superimpose_channel):
    path_dict = {'fixed_residues': {}, 'motif_residues': {}, 'catres_identities': {}, 'pose_paths': {}, 'original_chain_id': {}, 'model_num': {}}
    for index, row in path_df.iterrows():
        #TODO: this has to be done because subsequent scripts assume that first fragment is chain A, ... otherwise superpositioning does not work anymore
        new_chain = chain_alphabet()[row['path'].index(row['chain_id'])]
        row['original_chain_id'] = row['chain_id']
        row['chain_id'] = new_chain
        path_dict['original_chain_id'][row['chain_id']] = row['original_chain_id']
        path_dict['model_num'][row['chain_id']] = row['model_num']
        path_dict['pose_paths'][row['chain_id']] = row['poses']
        path_dict['fixed_residues'][row['chain_id']] = [row['rotamer_pos']]
        path_dict['motif_residues'][row['chain_id']] = [_ for _ in range(1, row['frag_length'] + 1)]
        path_dict['catres_identities'][f"{row['chain_id']}{row['rotamer_pos']}"] = row['AAs'][row['rotamer_pos'] - 1]
    mean_cols = ['path_score', 'ensemble_score', 'fragment_score', 'backbone_score', 'path_num_matches', 'match_score', 'rotamer_probability', 'rotamer_score']
    series_dict = {}
    for col in mean_cols:
        series_dict[col] = path_df[col].mean()
    for col in path_dict:
        series_dict[col] = path_dict[col]
    
    series = pd.Series(series_dict)
    series['path_order'] = path_order
    series['path_name'] = create_path_name(path_df, path_order)
    series['ligand_name'] = ligand_name
    series['ligand_chain'] = 'Z'
    series['rfdiffusion_pose_opts'] = compile_rfdiff_pose_opts(series)
    series['pose'] = create_path_pdb(series, pdb_dir, ligands, channel_exists, add_channel, auto_superimpose_channel)
    series['covalent_bonds'] = create_covalent_bonds(path_df)
    return series

def chain_alphabet():
    return ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

def main(args):
    '''
    Combines every model from each input pdb with every model from other input pdbs. Input pdbs must only contain chain A and (optional) a ligand in chain Z.
    Checks if any combination contains clashes, and removes them.
    Writes the coordinates of N, CA, C of the central atom as well as the rotamer probability and other infos to a json file.
    '''
    if os.environ.get('SLURM_SUBMIT_DIR'):
        script_dir = "/home/mabr3112/riff_diff/"
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))

    if not args.json_files and not args.json_prefix:
        raise RuntimeError('Either --json_files or --json_prefix must be specified!')

    auto_superimpose_channel = str2bool(args.auto_superimpose_channel)
    if args.add_channel:
        if os.path.isfile(args.add_channel) == True:
            channel_exists = True
        elif args.add_channel in ['False', 'false', '0', 'no']:
            channel_exists = False
        else:
            raise RuntimeError(f'<add_channel> should be either path to pdb containing channel in chain Q or set to False!')

    
    out_json = os.path.join(args.working_dir, f'{args.output_prefix}_assembly.json') 
    os.makedirs(args.working_dir, exist_ok=True)
    if os.path.exists(out_json):
        raise RuntimeError(f'Output file already exists at {out_json}!')
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=os.path.join(args.working_dir, f"{args.output_prefix}_assembly.log"))
    cmd = ''
    for key, value in vars(args).items():
        cmd += f'--{key} {value} '
    cmd = f'{sys.argv[0]} {cmd}'
    logging.info(cmd)

    #import json files
    input_jsons = []
    if args.json_files:
        input_jsons += [os.path.abspath(json) for json in args.json_files]
    if args.json_prefix:
        json_prefix = os.path.abspath(args.json_prefix)
        path, prefix = os.path.split(json_prefix)
        for file in os.listdir(path):
            if file.endswith('.json') and file.startswith(prefix):
                input_jsons.append(os.path.join(path, file))

    input_jsons = sorted(list(set(input_jsons)))

    inputs = []
    column_names = ['model_num', 'rotamer_pos', 'AAs', 'backbone_score', 'fragment_score', 'rotamer_probability', 'covalent_bond', 'ligand_chain', 'poses', 'poses_description']
    for file in input_jsons:
        df = pd.read_json(file)
        if not all(column in df.columns for column in column_names):
            raise RuntimeError(f'{file} is not a correct fragment json file!')
        inputs.append(df)
    in_df = pd.concat(inputs).reset_index(drop=True)


    database = myutils.path_ends_with_slash(args.database_dir)

    clash_dir = os.path.join(args.working_dir, f'{args.output_prefix}_clash_check')
    os.makedirs(clash_dir, exist_ok=True)

    grouped_df = in_df.groupby('poses', sort=False)
    counter = 0

    df_list = []
    structdict = {}
    ensemble_size = grouped_df.mean(numeric_only=True)['frag_length'].sum()
    
    for pose, pose_df in grouped_df:
        channel_clashes = 0
        log_and_print(f'Working on {pose}...')
        pose_df['input_poses'] = pose_df['poses']
        pose_df['chain_id'] = chain_alphabet()[counter]
        struct = myutils.import_structure_from_pdb(pose)
        model_dfs = []
        for index, series in pose_df.iterrows():
            chain = struct[series['model_num']]['A']
            if channel_exists == True and auto_superimpose_channel == False:
                if distance_detection(chain, myutils.import_structure_from_pdb(args.add_channel)[0]["Q"], True, False, args.channel_clash_detection_vdw_multiplier, database=database) == True:
                    channel_clashes += 1
                    continue
            chain.id = chain_alphabet()[counter]
            model_dfs.append(series)
        if channel_exists == True and auto_superimpose_channel == False:
            log_and_print(f'Removed {channel_clashes} models that were clashing with channel chain found in {args.add_channel}.')
        if len(model_dfs) == 0:
            raise RuntimeError(f'Could not find any models that are not clashing with channel chain for {pose}. Adjust clash detection parameters or move channel!')
        pose_df = pd.DataFrame(model_dfs)
        structdict[struct.id] = struct
        filename = os.path.join(clash_dir, f'{args.output_prefix}_{struct.id}_rechained.pdb')
        struct.id = filename
        myutils.write_multimodel_structure_to_pdb(struct, filename)
        pose_df['poses'] = os.path.abspath(filename)
        counter += 1
        #TODO: no idea if this is still required
        if 'covalent_bond' in pose_df.columns:
            pose_df['covalent_bond'].replace(np.nan, None, inplace=True)
        else:
            pose_df['covalent_bond'] = None
        df_list.append(pose_df)

    ligands = struct[0]['Z']
    lig_names = ','.join([res.get_resname() for res in ligands.get_residues()])
    for res in ligands.get_residues():
        res.id = ("H", res.id[1], res.id[2])


    
    
    grouped_df = pd.concat(df_list).groupby('poses', sort=False)
    
    # generate every possible combination of input models
    num_models = [len(df.index) for group, df in grouped_df]
    num_combs = 1
    for i in num_models:
        num_combs *= i
    log_and_print(f'Generating {num_combs} possible combinations...')

    init = time.time()

    combinations = itertools.product(*[[row for index, row in pose_df.iterrows()] for pose, pose_df in grouped_df])

    clash_dir = os.path.join(args.working_dir, f'{args.output_prefix}_clash_check')
    os.makedirs(clash_dir, exist_ok=True)

    log_and_print(f'Performing pairwise clash detection...')
    ensemble_dfs = run_clash_detection(combinations, num_combs, 320, clash_dir, args.fragment_backbone_clash_detection_vdw_multiplier, args.fragment_fragment_clash_detection_vdw_multiplier, database, "/home/tripp/riffdiff2/riff_diff/utils/clash_detection.py")
    
    #calculate scores
    score_df = ensemble_dfs.groupby('ensemble_num', sort=False).mean(numeric_only=True)
    score_df = normalize_col(score_df, 'fragment_score', scale=True, output_col_name='ensemble_score')['ensemble_score']
    ensemble_dfs = ensemble_dfs.merge(score_df, left_on='ensemble_num', right_index=True).sort_values('ensemble_num').reset_index(drop=True)


    
    #remove all clashing ensembles
    log_and_print(f'Filtering clashing ensembles...')
    post_clash = ensemble_dfs[ensemble_dfs['clash_check'] == False].reset_index(drop=True)
    log_and_print(f'Completed clash check in {round(time.time() - init, 0)} s.')
    if len(post_clash) == 0:
        log_and_print(f'No ensembles found! Try adjusting VdW multipliers or pick different fragments!')
        raise RuntimeError(f'No ensembles found! Try adjusting VdW multipliers or pick different fragments!')

    passed = int(len(post_clash.index) / len(input_jsons))
    log_and_print(f'Found {passed} non-clashing ensembles.')

    dfs = [post_clash, ensemble_dfs]
    df_names = ['filtered', 'unfiltered']
    cols = ['ensemble_score', 'fragment_score', 'backbone_score', 'rotamer_score']
    col_names = ['ensemble score', 'fragment score', 'backbone score', 'rotamer score']
    y_labels = ['score [AU]', 'score [AU]', 'score [AU]', 'score [AU]']
    dims = [(-0.05, 1.05), (-0.05, 1.05), (-0.05, 1.05), (-0.05, 1.05)]
    plotpath = os.path.join(args.working_dir, f"{args.output_prefix}_clash_filter.png")
    _ = plots.violinplot_multiple_cols_dfs(dfs, df_names=df_names, cols=cols, titles=col_names, y_labels=y_labels, dims=dims, out_path=plotpath)



    #### sort ensembles by fragment score ####
    log_and_print(f'Sorting and filtering ensembles by fragment score...')
    score_df_list = []
    for ensemble, df in post_clash.groupby('ensemble_num', sort=False):
        score_df_list.append((df['fragment_score'].mean(), df))

    score_df_list.sort(key=lambda x: x[0], reverse=True)
    #only accept top ensembles as master input
    score_df_list = score_df_list[0:args.max_master_input]

    score_sorted_list = []
    ensemble_num = 0
    for score_df in score_df_list:
        df = score_df[1]
        df['ensemble_num'] = ensemble_num
        df['ensemble_name'] = 'ensemble_' + str(ensemble_num)
        score_sorted_list.append(df)
        ensemble_num += 1
    master_input = pd.concat(score_sorted_list)

    dfs = [master_input, post_clash]
    df_names = ['master input', 'non-clashing ensembles']
    cols = ['ensemble_score', 'fragment_score', 'backbone_score', 'rotamer_score']
    col_names = ['ensemble score', 'fragment score', 'backbone score', 'rotamer score']
    y_labels = ['score [AU]', 'score [AU]', 'score [AU]', 'score [AU]']
    dims = [(-0.05, 1.05), (-0.05, 1.05), (-0.05, 1.05), (-0.05, 1.05)]
    plotpath = os.path.join(args.working_dir, f"{args.output_prefix}_master_input_filter.png")
    _ = plots.violinplot_multiple_cols_dfs(dfs, df_names=df_names, cols=cols, titles=col_names, y_labels=y_labels, dims=dims, out_path=plotpath)


    #### check for matches in database for top ensembles ####
    
    log_and_print(f'Writing input ensembles to disk...')
    master_dir = os.path.join(args.working_dir, f'{args.output_prefix}_master')
    os.makedirs(master_dir, exist_ok=True)
    os.makedirs(f"{master_dir}/pdbs/", exist_ok=True)
    filenames = []
    for index, ensemble in master_input.groupby('ensemble_num', sort=False):
        ens = Structure.Structure(f'ensemble_{index}.pdb')
        ens.add(Model.Model(0))
        for num, row in ensemble.iterrows():
            pose = row['poses_description']
            ens_chain = structdict[pose][row['model_num']][row['chain_id']]
            ens_chain.id = row['chain_id']
            ens[0].add(ens_chain)
        filename = os.path.join(f"{master_dir}/pdbs/", ens.id)
        if not os.path.isfile(filename):
            myutils.write_multimodel_structure_to_pdb(ens, filename)
        filenames.append(filename)

    if args.master_rmsd_cutoff == "auto":
        rmsd_cutoff = auto_determine_rmsd_cutoff(ensemble_size)
    else:
        rmsd_cutoff = args.master_rmsd_cutoff

    log_and_print(f'Checking for matches in database below {rmsd_cutoff} A.')

    df = run_masterv2(poses=filenames, output_dir=f"{master_dir}/output", chains= [chain_alphabet()[i] for i in range(0, counter)], rmsdCut=rmsd_cutoff, master_dir=args.master_dir, database=args.master_db, max_array_size=args.max_array_size)
    #df['combined_matches'] = df['ensemble_num_matches'] + df['path_num_matches']
    df = normalize_col(df, 'path_num_matches', True)
    df = normalize_col(df, 'ensemble_num_matches', True)
    df = combine_normalized_scores(df, 'match_score', ['path_num_matches_normalized', 'ensemble_num_matches_normalized'], [args.path_match_weight, args.ensemble_match_score], False, False)
    post_match = master_input.merge(df, left_on='ensemble_name', right_on='description').drop('description', axis=1)
    post_match = combine_normalized_scores(post_match, 'path_score', ['ensemble_score', 'match_score'], [args.fragment_score_weight, args.match_score_weight], False, False)
    _ = plots.violinplot_multiple_cols(post_match, cols=['match_score', 'path_num_matches', 'ensemble_num_matches'], titles=['match score', f'path matches\n< {rmsd_cutoff}', f'ensemble matches\n< {rmsd_cutoff}'], y_labels=['AU', '#', '#'], dims=[(-0.05, 1.05), (post_match['path_num_matches'].max() * -0.05, post_match['path_num_matches'].max() * 1.05 ), (post_match['ensemble_num_matches'].max() * -0.05, post_match['ensemble_num_matches'].max() * 1.05 )], out_path=os.path.join(args.working_dir, f"{args.output_prefix}_master_matches_<_{rmsd_cutoff}.png"))

    
    path_dfs = post_match.copy()
    if args.match_cutoff:
        passed = int(len(path_dfs.index) / len(input_jsons))
        path_dfs = path_dfs[path_dfs['ensemble_num_matches'] >= args.match_cutoff]
        filtered = int(len(path_dfs.index) / len(input_jsons))
        log_and_print(f'Removed {passed - filtered} paths with less than {args.match_cutoff} ensemble matches below {rmsd_cutoff} A. Remaining paths: {filtered}.')
        dfs = [path_dfs, post_match]
        df_names = ['filtered', 'unfiltered']
        cols = ['path_score', 'match_score', 'ensemble_score', 'fragment_score', 'backbone_score', 'rotamer_score']
        col_names = ['path score', 'master score', 'ensemble score', 'fragment score', 'backbone score', 'rotamer score']
        y_labels = ['score [AU]', 'score [AU]', 'score [AU]', 'score [AU]', 'score [AU]', 'score [AU]']
        dims = [(-0.05, 1.05), (-0.05, 1.05), (-0.05, 1.05), (-0.05, 1.05), (-0.05, 1.05), (-0.05, 1.05)]
        plotpath = os.path.join(args.working_dir, f"{args.output_prefix}_num_matches_<_{args.match_cutoff}_filter.png")
        _ = plots.violinplot_multiple_cols(path_dfs, cols=['match_score', 'path_num_matches', 'ensemble_num_matches', 'mean_match_rmsd', 'min_match_rmsd'], titles=['match score', f'path\nmatches < {rmsd_cutoff}', f'ensemble\nmatches < {rmsd_cutoff}', 'ensemble\nmean match rmsd', 'ensemble\nminimum match rmsd'], y_labels=['AU', '#', '#', 'A', 'A'], dims=[(-0.05, 1.05), (path_dfs['path_num_matches'].min() - (path_dfs['path_num_matches'].max() - path_dfs['path_num_matches'].min()) * 0.05, (path_dfs['path_num_matches'].max() + (path_dfs['path_num_matches'].max() - path_dfs['path_num_matches'].min()) * 0.05)), (args.match_cutoff - (path_dfs['path_num_matches'].max() - args.match_cutoff) * 0.05, (path_dfs['ensemble_num_matches'].max() + (path_dfs['ensemble_num_matches'].max() - args.match_cutoff) * 0.05)), (rmsd_cutoff * -0.05, rmsd_cutoff * 1.05), (rmsd_cutoff * -0.05, rmsd_cutoff * 1.05)], out_path=os.path.join(args.working_dir, f"{args.output_prefix}_master_matches_<_{rmsd_cutoff}_matchfilter.png"))
        _ = plots.violinplot_multiple_cols_dfs(dfs, df_names=df_names, cols=cols, titles=col_names, y_labels=y_labels, dims=dims, out_path=plotpath)
    

    ens_num = 0
    pdb_dir = os.path.join(args.working_dir, "pdb_in")
    os.makedirs(pdb_dir, exist_ok=True)

    ensemble_grouped_path_df = path_dfs.copy()
    ensemble_filtered_path_df = []
    for ensemble, df in ensemble_grouped_path_df.groupby('ensemble_num', sort=False):
        ensemble_filtered_path_df.append(df.sort_values(['path_score', 'path'], ascending=False).head(args.max_paths_per_ensemble * len(input_jsons)))
    ensemble_filtered_path_df = pd.concat(ensemble_filtered_path_df)
    log_and_print(f'Removed {int((len(path_dfs.index) - len(ensemble_filtered_path_df.index)) / len(input_jsons))} paths because of {args.max_paths_per_ensemble} paths per ensemble cutoff.\nTotal passed paths: {int(len(ensemble_filtered_path_df.index) / len(input_jsons))}.')

    top_path_dfs = ensemble_filtered_path_df.sort_values(['path_score', 'ensemble_num', 'path'], ascending=False).head(args.max_out * len(input_jsons))
    log_and_print(f'Selecting top {args.max_out} paths...')

    selected_paths = []
    for ensemble_num, ensemble_df in top_path_dfs.groupby('ensemble_num', sort=False):
        ensemble_df['ensemble_num'] = ens_num
        ens_num += 1
        for path_order, path_df in ensemble_df.groupby('path', sort=False):
            path_series = create_path_series(path_df, path_order, ligands, lig_names, pdb_dir, channel_exists, args.add_channel, auto_superimpose_channel) 
            selected_paths.append(path_series)

    selected_paths = pd.DataFrame(selected_paths).sort_values('path_score', ascending=False).reset_index(drop=True).drop(['path_order', 'pose_paths'], axis=1)

    selected_paths.set_index('path_name', inplace=True)
    selected_paths.to_json(os.path.join(args.working_dir, "selected_paths.json"))

    ligand_dir = os.path.join(args.working_dir, 'ligand')
    os.makedirs(ligand_dir, exist_ok=True)
    for index, ligand in enumerate(ligands.get_residues()):
        ligand_pdbfile = utils.biopython_tools.store_pose(ligand, (lig_path:=os.path.join(ligand_dir, f"LG{index+1}.pdb")))
        lig_name = ligand.get_resname()
        if len(list(ligand.get_atoms())) > 2:
            # store ligand as .mol file for rosetta .molfile-to-params.py
            log_and_print(f"Running 'molfile_to_params.py' to generate params file for Rosetta.")
            lig_molfile = obabel_fileconverter(input_file=lig_path, output_file=lig_path.replace(".pdb", ".mol2"), input_format="pdb", output_format=".mol2")
            run(f"python3 {script_dir}/rosetta/molfile_to_params.py -n {lig_name} -p {ligand_dir}/LG{index+1} {lig_molfile} --keep-names --clobber --chain=Z", shell=True, stdout=True, check=True, stderr=True)
            lig_path = f"{ligand_dir}/LG{index+1}_0001.pdb"
        else:
            log_and_print(f"Ligand at {ligand_pdbfile} contains less than 3 atoms. No Rosetta Params file can be written for it.")

    _ = plots.violinplot_multiple_cols(selected_paths, cols=['path_score', 'match_score', 'path_num_matches', 'ensemble_score', 'fragment_score', 'backbone_score', 'rotamer_score', 'rotamer_probability'], titles=['path score', 'master score', f'matches < {rmsd_cutoff} A', 'ensemble score', 'mean fragment score', 'mean backbone score', 'mean rotamer score', 'mean rotamer\nprobability'], y_labels=['AU', 'AU', '#', 'AU', 'AU', 'AU', 'AU', 'probability'], dims=[(-0.05, 1.05), (-0.05, 1.05), (0 - selected_paths['path_num_matches'].max() * 0.05, selected_paths['path_num_matches'].max() * 1.05), (-0.05, 1.05), (-0.05, 1.05), (-0.05, 1.05), (-0.05, 1.05), (-0.05, 1.05)], out_path=os.path.join(args.working_dir, f"{args.output_prefix}_selected_paths_info.png"))


    dfs = [top_path_dfs, path_dfs]
    df_names = ['selected paths', '> match cutoff']
    cols = ['path_score', 'match_score', 'ensemble_score', 'fragment_score', 'backbone_score', 'rotamer_score']
    col_names = ['path score', 'master score', 'ensemble score', 'fragment score', 'backbone score', 'rotamer score']
    y_labels = ['score [AU]', 'score [AU]', 'score [AU]', 'score [AU]', 'score [AU]', 'score [AU]']
    dims = [(-0.05, 1.05), (-0.05, 1.05), (-0.05, 1.05), (-0.05, 1.05), (-0.05, 1.05), (-0.05, 1.05)]
    plotpath = os.path.join(args.working_dir, f"{args.output_prefix}_top_paths_filter.png")
    _ = plots.violinplot_multiple_cols_dfs(dfs, df_names=df_names, cols=cols, titles=col_names, y_labels=y_labels, dims=dims, out_path=plotpath)


    log_and_print(f'Done!')

if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # mandatory input
    argparser.add_argument("--database_dir", type=str, default="/home/tripp/riffdiff2/riff_diff/database/", help="Path to folder containing rotamer libraries, fragment library, etc.")
    argparser.add_argument("--json_prefix", type=str, default=None, nargs='?', help="Prefix for all json files that should be combined (including path, e.g. './output/mo6_'). Alternative to --json_files")
    argparser.add_argument("--json_files", default=None, nargs='*', help="List of json files that contain fragment information. Alternative to --json_prefix.")
    argparser.add_argument("--output_prefix", type=str, required=True, help="Prefix for output.")
    argparser.add_argument("--working_dir", type=str, required=True, help="Path to working directory. Has to contain the input pdb files, otherwise run_ensemble_evaluator.py will not work!")

    # stuff you might want to adjust
    argparser.add_argument("--channel_clash_detection_vdw_multiplier", type=float, default=0.9, help="Multiplier for VanderWaals radii for clash detection between backbone fragments and channel placeholder. Clash is detected if distance_between_atoms < (VdW_radius_atom1 + VdW_radius_atom2)*multiplier")
    argparser.add_argument("--fragment_backbone_clash_detection_vdw_multiplier", type=float, default=1.0, help="Multiplier for VanderWaals radii for clash detection inbetween backbone fragments. Clash is detected if distance_between_atoms < (VdW_radius_atom1 + VdW_radius_atom2)*multiplier")
    argparser.add_argument("--backbone_ligand_clash_detection_vdw_multiplier", type=float, default=1.0, help="Multiplier for VanderWaals radii for clash detection between fragment backbones and ligand. Set None if no ligand is present. Clash is detected if distance_between_atoms < (VdW_radius_atom1 + VdW_radius_atom2)*multiplier")
    argparser.add_argument("--rotamer_ligand_clash_detection_vdw_multiplier", type=float, default=0.75, help="Multiplier for VanderWaals radii for clash detection between rotamer sidechain and ligand. Clash is detected if distance_between_atoms < (VdW_radius_atom1 + VdW_radius_atom2)*multiplier")
    argparser.add_argument("--fragment_fragment_clash_detection_vdw_multiplier", type=float, default=0.85, help="Multiplier for VanderWaals radii for clash detection inbetween fragments (including sidechains!). Effectively detects clashes between rotamer of one fragment and the other fragment (including the other rotamer) if multiplier is lower than <fragment_backbone_clash_detection_vdw_multiplier>. Clash is detected if distance_between_atoms < (VdW_radius_atom1 + VdW_radius_atom2)*multiplier")
    argparser.add_argument("--ligand_chain", type=str, default="Z", help="Name of ligand chain.")
    argparser.add_argument("--master_rmsd_cutoff", default='auto', help="Detects how many structures have segments with RMSD below this cutoff for each ensemble. Higher cutoff increases runtime tremendously!")
    argparser.add_argument("--master_dir", type=str, default="/home/tripp/MASTER-v2-masterlib/bin/", help="Path to master executable")
    argparser.add_argument("--master_db", type=str, default="/home/tripp/MASTER-v2-masterlib/master_db/list", help="Path to Master database")
    argparser.add_argument("--max_master_input", type=int, default=20000, help="Maximum number of ensembles that should be fed into master, sorted by fragment score")
    argparser.add_argument("--match_score_weight", type=float, default=1, help="Maximum number of cpus to run on")
    argparser.add_argument("--fragment_score_weight", type=float, default=1, help="Maximum number of cpus to run on")
    argparser.add_argument("--match_cutoff", type=int, default=1, help="Remove all ensembles that have less matches than <match_cutoff> below <master_rmsd_cutoff>")
    argparser.add_argument("--max_out", type=int, default=200, help="Maximum number of output paths")
    argparser.add_argument("--add_channel", type=str, default="/home/mabr3112/riff_diff/utils/helix_cone_long.pdb", help="If specified, adds the structure specified to the fragment to be used as a 'substrate channel' during diffusion. IMPORTANT!!!  Channel pdb-chain name has to be 'Q' ")
    argparser.add_argument("--auto_superimpose_channel", type=str, default="True", help="Set to false, if you want to copy the channel pdb-chain from the reference file without superimposing on moitf-substrate centroid axis.")
    argparser.add_argument("--path_match_weight", type=float, default=1, help="Weight of the path-specific number of matches for calculating match score")
    argparser.add_argument("--ensemble_match_score", type=float, default=1, help="Weight of the number of matches for all paths within the ensemble for calculating match score")
    argparser.add_argument("--max_paths_per_ensemble", type=int, default=5, help="Maximum number of paths per ensemble (=same fragments but in different order)")

    argparser.add_argument("--max_array_size", type=int, default=320, help="Maximum number of cpus to run on")


    args = argparser.parse_args()

    main(args)
