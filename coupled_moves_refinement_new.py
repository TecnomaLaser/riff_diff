#!/home/mabr3112/anaconda3/bin/python3.9

import os
import argparse
import subprocess
import math
import time
import pandas as pd
import shutil
import Bio
from glob import glob
from subprocess import run
import json
import copy
import itertools
from Bio.PDB import *
import logging


import sys
sys.path += ["/home/mabr3112/projects/iterative_refinement/"]
#sys.path += ["/home/tripp/riffdiff2/riff_diff/it_test/"]
sys.path.append("/home/mabr3112/riff_diff")

import utils.adrian_utils as my_utils
#import rfdiffusion_and_refinement as diffrf
# import custom modules
from iterative_refinement import *
import utils.plotting as plots
import utils.biopython_tools
import utils.pymol_tools
from utils.plotting import PlottingTrajectory
import utils.metrics as metrics
import superimposition_tools as si_tools
from protocols.composite_protocols import rosetta_scripts_and_mean


def aa_three_or_one_letter_code(AA):
    letter_codes = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
        "A": "ALA",
        "R": "ARG",
        "N": "ASN",
        "D": "ASP",
        "C": "CYS",
        "Q": "GLN",
        "E": "GLU",
        "G": "GLY",
        "H": "HIS",
        "I": "ILE",
        "L": "LEU",
        "K": "LYS",
        "M": "MET",
        "F": "PHE",
        "P": "PRO",
        "S": "SER",
        "T": "THR",
        "W": "TRP",
        "Y": "TYR",
        "V": "VAL"
    }

    return(letter_codes[AA])

def calc_ligand_stats(input_df: pd.DataFrame, ref_frags_col:str, ref_motif_col:str, poses_motif_col:str, prefix:str, ligand_chain:str="Z", save_path_list=None) -> None:
    '''
    Superimposes the poses onto reference fragments in input_df[ref_frags_col] by specified motifs in input_df. Then calculates statistics over ligands. (if it is clashing and the number of contacts).
    '''
    if f"{prefix}_ligand_clash" in input_df.columns:
        print('Ligand stats found in dataframe. Skipping step.')
        return input_df
    # superimpose reference frags onto poses to make sure ligand calculation works in the same coordinate frame:
    if save_path_list:
        poses = [superimposition_tools.superimpose_pdb_by_motif(ref_frag, pose, fixed_motif=ref_motif, mobile_motif=pose_motif, atoms=["CA"], save_path=path) for pose, ref_frag, pose_motif, ref_motif, path in zip(input_df["poses"].to_list(), input_df[ref_frags_col].to_list(), input_df[poses_motif_col].to_list(), input_df[ref_motif_col].to_list(), save_path_list)]
    else:
        poses = [superimposition_tools.superimpose_pdb_by_motif(ref_frag, pose, fixed_motif=ref_motif, mobile_motif=pose_motif, atoms=["CA"]) for pose, ref_frag, pose_motif, ref_motif in zip(input_df["poses"].to_list(), input_df[ref_frags_col].to_list(), input_df[poses_motif_col].to_list(), input_df[ref_motif_col].to_list())]

    # calculate statistics of ligands:
    input_df[f"{prefix}_ligand_clash"] = [utils.metrics.check_for_ligand_clash_of_pdb(pose, ligand_chain=ligand_chain, ligand_pdb_path=ref_pose, dist=1.4, ignore_atoms=["H"]) for pose, ref_pose in zip(input_df["poses"].to_list(), input_df[ref_frags_col].to_list())]
    #input_df[f"{prefix}_peratom_ligand_contacts"] = [utils.metrics.calc_ligand_contacts_of_pdb(pose, ligand_chain=ligand_chain, ligand_pdb_path=ref_pose, d_0=3.5, r_0=3.5, ignore_atoms=["H"]) for pose, ref_pose in zip(input_df["poses"].to_list(), input_df[ref_frags_col].to_list())]

    return input_df



def get_protein_sequence(pdb_file):
  # Create a parser object
  parser = Bio.PDB.PDBParser(QUIET=True)
  # Parse the pdb file and get the structure object
  structure = parser.get_structure("protein", pdb_file)
  # Create an empty list to store the sequence
  sequence = []
  # Loop over all the residues in the first model and first chain of the structure
  for residue in structure.get_residues():
    # Check if the residue is standard amino acid
    if Bio.PDB.is_aa(residue):
      # Get the one-letter code of the residue and append it to the sequence list
      sequence.append(aa_three_or_one_letter_code(residue.get_resname()))
  # Join the sequence list into a string and return it
  return "".join(sequence)

def create_output_dir_change_filename(output_dir, filename):
    if output_dir:
        output_dir = my_utils.path_ends_with_slash(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        filename = output_dir + filename
    else:
        filename = filename
    return filename


def groups_from_unique_input(scoretable, label):
    all_elements = scoretable[label].values.tolist()
    unique = set(all_elements)
    unique = sorted(list(unique))
    return unique

def extract_designpositions_from_resfile(resfile):
    design_positions = []
    with open(resfile, "r") as r:
        for line in r:
            if not line.startswith("NAT") and not line.startswith("start"):
                design_positions.append(int(line.split()[0]))
    #filter for unique elements in list
    #design_positions = sorted(list(set(design_positions)))
    return(design_positions)

def statsfile_to_df(statsfile: str):
    #reads in the .stats file output from a coupled-moves run and converts it to a dataframe
    df = pd.read_csv(statsfile, sep=None, engine='python', header=None, keep_default_na=False)
    df_scores = df[4].str.split(expand=True)
    columnheaders = df_scores[df_scores.columns[0::2]]
    columnheaders = columnheaders.loc[0, :].values.tolist()
    columnheaders = [i.replace(':','') for i in columnheaders]
    df_scores = df_scores[df_scores.columns[1::2]]
    df_scores.columns = columnheaders
    df_scores = df_scores.astype(float)
    df_scores["total_score"] = df_scores.sum(axis=1)
    df_scores["sequence"] = df[3]
    return(df_scores)

def statsfiles_to_json(input_dir: str, description:str, filename):

    if os.path.isfile(filename):
        with open(filename) as json_file:
            print(f"Read structdict from file {filename}")
            structdict = json.load(json_file)
            return(structdict)
    #gathers all coupled-moves statsfiles and converts to a single dictionary
    statsfiles = []
    resfiles = []
    for file in os.listdir(input_dir):
        if file.endswith(".stats") and file[6:-11] == description:
            statsfiles.append(input_dir + file)
        elif file.endswith(".resfile") and file[6:-13] == description:
            resfiles.append(input_dir + file)

    statsfiles = sorted(statsfiles)
    resfiles = sorted(resfiles)


    df = pd.DataFrame()


    stats_df_list = []

    for stats, res in zip(statsfiles, resfiles):
        statsdf = statsfile_to_df(stats)
        #statsdf['total_score'] = statsdf['total_score'] - statsdf['res_type_constraint']
        seqlist = statsdf["sequence"].tolist()
        design_positions = extract_designpositions_from_resfile(res)
        design_positions = [design_positions for i in range(0, len(statsdf))]
        statsdf['design_positions'] = design_positions
        for index, row in statsdf.iterrows():
            for mut, pos in zip(row['sequence'], row['design_positions']):
                mut_row = row.copy()
                mut_row['mutation'] = mut
                mut_row['position'] = pos
                stats_df_list.append(mut_row)

    statsdf = pd.DataFrame(stats_df_list)
    statsdf['total_score'] = statsdf['total_score'] - statsdf['res_type_constraint']

    structdict = {}
    for pos, df in statsdf.groupby('position'):
        posdict = {}
        for AA, pos_df in df.groupby('mutation'):
            posdict[AA] = {"pos": pos, "identity": AA, "count": len(pos_df), "ratio": len(pos_df)/len(df), "total_score": [round(score, 2) for score in pos_df['total_score'].to_list()], "total_score_average": round(pos_df['total_score'].mean(), 2), "coordinate_constraint": [round(score, 2) for score in pos_df['coordinate_constraint'].to_list()], "coordinate_constraint_average": round(pos_df['coordinate_constraint'].mean(), 2)}
        structdict[pos] = posdict

    with open(filename, "w") as outfile:
        json.dump(structdict, outfile)
    return(structdict)


def import_json_to_dict(jsonpath: str):
    with open(jsonpath) as jsonfile:
        data = json.load(jsonfile)
    return(data)


def generate_mutations_dict(datadict, occurence_cutoff):
    '''
    only accepts mutations that show up in at least <occurence_cutoff> of coupled moves runs. if no mutation is above 30%, picks the most common one.
    '''
    mutations = {}
    for pos in datadict:
        df = pd.DataFrame(datadict[pos]).transpose().sort_values('ratio')
        df_filtered = df[df['ratio'] >= occurence_cutoff]
        if df_filtered.empty:
            df_filtered = df[df['ratio'] >= 0.1]
            if df_filtered.empty:
                df_filtered = df
            df_filtered = df_filtered.sort_values('coordinate_constraint_average', ascending=True).head(int(1 + len(df_filtered) / 2))
            df_filtered = df_filtered.sort_values('total_score_average', ascending=True).head(1)
        mutations[pos] = df_filtered['identity'].to_list()

    return mutations


def generate_variants(mutation_dict, pdb):
    mutlist = []
    poslist = []
    for pos in mutation_dict:
        poslist.append(pos)
        mutlist.append(mutation_dict[pos])
    combs = list(itertools.product(*mutlist))
    seq = list(get_protein_sequence(pdb))
    variants = []
    for comb in combs:
        var = copy.deepcopy(seq)
        for index, AA in enumerate(comb):
            var[int(poslist[index]) - 1] = AA
        variants.append(''.join(var))
    return variants

def chainresdict_to_str(in_dict:dict):
    cat_res = []
    for chain in in_dict:
        for residue in in_dict[chain]:
            cat_res.append(f'{str(residue)+chain}')
    return ",".join(cat_res)



def extract_residue_atoms_from_structure(structure, chaindict):
    res = []
    for chain in chaindict:
        for resnum in chaindict[chain]:
            res.append(structure[0][chain][resnum])
    atoms = []
    for residue in res:
        for atom in residue.get_atoms():
            if atom.element in ['C', 'N', 'O', 'S']:
                atoms.append(atom)
    return atoms

def prepare_coupled_moves_relax_mpnn(output_dir, working_dir, cm_resultsdir, poses_df, occurence_cutoff, cycle_num):

    variants_list = []
    os.makedirs(output_dir, exist_ok=True)

    for index, row in poses_df.iterrows():
        statsdict = statsfiles_to_json(working_dir, row['poses_description'], f"{cm_resultsdir + row['poses_description']}.json")
        mutations = generate_mutations_dict(statsdict, occurence_cutoff)
        variants_df = pd.DataFrame(generate_variants(mutations, row['poses']), columns=[f'cycle_{cycle_num}_sequence'])
        print(f"Generated {len(variants_df.index)} variants for pose {row['poses_description']}.")
        for seqnum, var in variants_df.iterrows():
            #TODO: hardcoded chain, make flexible later
            new_pose_description = f"{row['poses_description']}_{seqnum+1:04d}"
            new_pose = os.path.join(output_dir, f"{new_pose_description}.pdb")
            shutil.copy(row['poses'], new_pose)
            fixed_pos = sorted(row['mutations_fixed_residues']['A'] + [int(pos) for pos in mutations])
            var[f'cycle_{cycle_num}_cm_fixed_positions'] = {'A': fixed_pos}
            var[f'poses'] = new_pose
            var[f'cycle_{cycle_num}_old_poses_description'] = row['poses_description']
            var[f'cycle_{cycle_num}_cm_poses_description'] = new_pose_description
            cm_relax_opts = f"-parser:script_vars cat_res={chainresdict_to_str(row['fixed_residues'])} motif_res={chainresdict_to_str(row['motif_residues'])} seq={var[f'cycle_{cycle_num}_sequence']} -in:file:native {row['updated_reference_frags_location']}"
            if 'params_file_path' in poses_df.columns:
                cm_relax_opts = cm_relax_opts + f" -extra_res_fa {os.path.abspath(row['params_file_path'])}"
            var[f'cycle_{cycle_num}_cm_relax_opts'] = cm_relax_opts
            variants_list.append(var)

    variants = pd.DataFrame(variants_list)
    print(f"Generated {len(variants.index)} variants in total.")
    poses_df = variants.merge(poses_df.drop('poses', axis=1), how='left', left_on=f'cycle_{cycle_num}_old_poses_description', right_on='poses_description').drop('poses_description', axis=1)
    poses_df['poses_description'] = poses_df[f'cycle_{cycle_num}_cm_poses_description']

    return poses_df.drop(f'cycle_{cycle_num}_cm_poses_description', axis=1)

def convert_af2_perresidue_plddt_to_list(df, perresidue_plddt_column):
    perresidue_plddts = df[perresidue_plddt_column].to_list()
    print(type(perresidue_plddts[0]))
    print(perresidue_plddts[0])
    perresidue_plddts = [list(i.values()) for i in perresidue_plddts]
    return perresidue_plddts

def filter_input_poses(poses, perresidue_plddt_column, ligand_chain, sitescore_cutoff, max_input_per_backbone, output_dir, database, bb_clash_vdw_multiplier):
    poses = clash_detection(poses=poses, ref_frags_col="updated_reference_frags_location", ref_motif_col="fixed_residues", poses_motif_col="fixed_residues", prefix="pre_cm", ligand_chain=ligand_chain, database_dir=database, bb_clash_vdw_multiplier=bb_clash_vdw_multiplier, save_path_list=None)
    poses.poses_df = poses.poses_df[poses.poses_df["pre_cm_ligand_clash"] == False]

    poses.calc_motif_bb_rmsd_df('updated_reference_frags_location', 'motif_residues', 'motif_residues', 'pre_cm_bb', ['N', 'CA', 'C', 'O'])
    poses.calc_motif_heavy_rmsd_df('updated_reference_frags_location', 'fixed_residues', 'fixed_residues', 'pre_cm_catres')

    if 'af2' in perresidue_plddt_column:
        #af2 perresidue plddts are dictionaries --> convert them to list
        perresidue_plddts = convert_af2_perresidue_plddt_to_list(poses.poses_df, perresidue_plddt_column)
    else:
        perresidue_plddts = poses.poses_df[perresidue_plddt_column].to_list()
    poses.add_site_score('pre_cm_bb_motif_rmsd', 'motif_residues', perresidue_plddts, 'pre_cm_motif')
    poses.add_site_score('pre_cm_catres_motif_heavy_rmsd', 'fixed_residues', perresidue_plddts, 'pre_cm_catres')
    if sitescore_cutoff:
        poses.filter_poses_by_score(max_input_per_backbone, 'pre_cm_motif_site_score', remove_layers=1, ascending=False)
        logging.info(f'{len(poses.poses_df.index)} input poses passed sitescore cutoff of {sitescore_cutoff}.')

    return(poses)

def distance_detection(entity1, entity2, vdw_radii:dict, bb_only:bool=True, ligand:bool=False, clash_detection_vdw_multiplier:float=1.0, covalent_bond:str=None, ignore_func_groups:bool=True):
    '''
    checks for clashes by comparing VanderWaals radii. If clashes with ligand should be detected, set ligand to true. Ligand chain must be added as second entity.
    bb_only: only detect backbone clashes between to proteins or a protein and a ligand.
    clash_detection_vdw_multiplier: multiply Van der Waals radii with this value to set clash detection limits higher/lower
    database: path to database directory
    '''
    backbone_atoms = ['CA', 'C', 'N', 'O', 'H']
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
        if covalent_bond:
            for cov_bond in covalent_bond.split(','):
                resnum, chain = split_pdb_numbering(cov_bond.split('_')[0])
                if atom_combination[0].get_parent().id[1] == resnum and atom_combination[0].get_parent().get_parent().id == chain and atom_combination[0].name == cov_bond.split(':')[0].split('_')[-1] and atom_combination[1].name == cov_bond.split(':')[1].split('_')[-1]:
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

def split_pdb_numbering(pdbnum):
    resnum = ""
    chain = ""
    for char in pdbnum:
        if char.isdigit():
            resnum += char
        else:
            chain += char
    resnum = int(resnum)
    if not chain:
        chain = "A"
    return [resnum, chain]


def clash_detection(poses, ref_frags_col:str, ref_motif_col:str, poses_motif_col:str, prefix:str, ligand_chain:str="Z", database_dir="database", bb_clash_vdw_multiplier=0.9, save_path_list=None) -> None:
    '''
    Superimposes the poses onto reference fragments in input_df[ref_frags_col] by specified motifs in input_df. Then calculates statistics over ligands. (if it is clashing and the number of contacts).
    '''
    scorefilepath = os.path.join(poses.scores_dir, f"{prefix}_ligand_clash.json")
    if f"{prefix}_ligand_clash" in poses.poses_df.columns:
        print('Ligand stats found in dataframe. Skipping step.')
        return poses
    if os.path.isfile(scorefilepath):
        print(f'Ligand stats found at {scorefilepath}. Skipping step. ')
        clash_df = pd.read_json(scorefilepath)
        poses.poses_df = poses.poses_df.merge(clash_df, on="poses_description")
        return poses
    # superimpose reference frags onto poses to make sure ligand calculation works in the same coordinate frame:
    if save_path_list:
        poses_path = [superimposition_tools.superimpose_pdb_by_motif(ref_frag, pose, fixed_motif=ref_motif, mobile_motif=pose_motif, atoms=["CA"], save_path=path) for pose, ref_frag, pose_motif, ref_motif, path in zip(poses.poses_df["poses"].to_list(), poses.poses_df[ref_frags_col].to_list(), poses.poses_df[poses_motif_col].to_list(), poses.poses_df[ref_motif_col].to_list(), save_path_list)]
        poses.poses_df['poses'] = save_path_list
    else:
        poses_path = [superimposition_tools.superimpose_pdb_by_motif(ref_frag, pose, fixed_motif=ref_motif, mobile_motif=pose_motif, atoms=["CA"]) for pose, ref_frag, pose_motif, ref_motif in zip(poses.poses_df["poses"].to_list(), poses.poses_df[ref_frags_col].to_list(), poses.poses_df[poses_motif_col].to_list(), poses.poses_df[ref_motif_col].to_list())]

    structs = []
    ligands = []
    covalent_bonds = []
    for index, row in poses.poses_df.iterrows():
        structs.append(my_utils.import_structure_from_pdb(row['poses']))
        ligands.append(my_utils.import_structure_from_pdb(row[ref_frags_col])[0][ligand_chain])
        try:
            covalent_bonds.append(row['covalent_bonds'])
        except:
            covalent_bonds.append(None)

    vdw_radii = import_vdw_radii(database_dir)

    # calculate statistics of ligands:
    poses.poses_df[f"{prefix}_ligand_clash"] = [distance_detection(struct, ligand, vdw_radii, True, True, bb_clash_vdw_multiplier, covalent_bond) for struct, ligand, covalent_bond in zip(structs, ligands, covalent_bonds)]
    poses.poses_df[['poses_description', f"{prefix}_ligand_clash"]].to_json(scorefilepath)
    return poses

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

def prepare_attnpacker_input_dirs(file_list, output_dir, files_per_folder):
    target_dir = f'{output_dir}/input_pdbs'

    input_dirs = []
    for i, file in enumerate(file_list):
        dir_name = f'{target_dir}/in_{str(int(1+i/files_per_folder)).zfill(4)}'
        os.makedirs(dir_name, exist_ok=True)
        file_name = os.path.basename(file)
        file_path = os.path.join(dir_name, file_name)
        shutil.copy(file_list[i], file_path)
        input_dirs.append(dir_name)

    return sorted(list(set(input_dirs)))

def run_attn(poses, prefix, attn_script:str="/home/tripp/riffdiff2/riff_diff/utils/run_attnpacker.py", gpu="auto"):

    working_dir = f"{poses.dir}/{prefix}"
    output_dir = f"{working_dir}/packed"

    scorefilepath = f'{working_dir}/att_repacker_scores.json'

    if os.path.isfile((scorefilepath)):
        out_df = pd.read_json(scorefilepath)
        out_df.rename(columns={'sc_plddts': f"{prefix}_sc_plddts"}, inplace=True)
        poses.poses_df = poses.poses_df.drop(['poses'], axis=1).merge(out_df, on="poses_description")
        #no idea if this is necessary
        poses.poses = list(poses.poses_df['poses'])
        return poses

    os.makedirs(working_dir, exist_ok=True)

    sbatch_options = ["-c1", f'-e {working_dir}/attnpacker.err -o {working_dir}/attnpacker.out']

    if gpu == "auto":
        if len(poses.poses_df['poses'].index) > poses.max_rosetta_cpus * 5:
            gpu = True
        else:
            gpu = False
    if gpu == True:
        files_per_folder = 100
        sbatch_options.append('--gpus-per-node 1')
        max_array_size = 10
    else:
        files_per_folder = 5
        max_array_size = poses.max_rosetta_cpus

    input_dirs = prepare_attnpacker_input_dirs(poses.poses_df['poses'].to_list(), working_dir, files_per_folder)

    cmds = [f"{attn_script} --input_dir {dir} --output_dir {output_dir} --scorefile {output_dir}/attn_repacker_{str(i+1).zfill(4)}_scores.json" for i, dir in enumerate(input_dirs)]

    sbatch_array_jobstarter(cmds=cmds, sbatch_options=sbatch_options, jobname="attnpacker", max_array_size=max_array_size, wait=True, remove_cmdfile=False, cmdfile_dir=working_dir)

    out_df = pd.DataFrame()
    for jsonfile in [file for file in os.listdir(output_dir) if file.startswith('attn_repacker_') and file.endswith('_scores.json')]:
        out_df = pd.concat([out_df, pd.read_json(os.path.join(output_dir, jsonfile))])
    out_df.reset_index(drop=True, inplace=True)

    out_df.to_json(scorefilepath)
    out_df.rename(columns={'sc_plddts': f"{prefix}_sc_plddts"}, inplace=True)
    poses.poses_df = poses.poses_df.drop(['poses'], axis=1).merge(out_df, on="poses_description")
    poses.poses = list(poses.poses_df['poses'])

    return poses

def write_fasta(pose, seq):
    # write fasta-files
    fasta_name = pose.replace(".pdb", ".fa")
    description = os.path.splitext(os.path.basename(pose))[0]
    with open(fasta_name, 'w') as f:
        f.write(f">{description}\n{seq}")

    return fasta_name

def create_reduced_motif(fixed_res:dict, motif_res:dict):
    reduced_dict = {}
    for chain in fixed_res:
        res = []
        reduced_motif = []
        for residue in fixed_res[chain]:
            res.append(residue -1)
            res.append(residue)
            res.append(residue + 1)
        for i in res:
            if i in motif_res[chain]:
                reduced_motif.append(i)
        reduced_dict[chain] = reduced_motif
    return reduced_dict

def update_sitescore_with_bb_plddts(poses, sitescore_column, bb_plddt_column, motif_residue_column, prefix):

    pose_motif_list = parse_pose_options(motif_residue_column, poses.poses_df)
    perresidue_plddt_list = parse_pose_options(bb_plddt_column, poses.poses_df)

    metric_name = f"{prefix}_sc_bb_site_score"

    if os.path.isfile((scorefile := f"{poses.scores_dir}/{metric_name}_scores.json")):
        print(f"Site score found at {scorefile} Reading scores directly from file.")
        site_score_df = pd.read_json(scorefile)
        poses.poses_df = poses.poses_df.merge(site_score_df, on="poses_description")
        if len(poses.poses_df) == 0: raise ValueError("ERROR: Length of DataFrame = 0. DataFrame merging failed!")
        site_score = list(poses.poses_df[metric_name])
    else:
        cat_res_pos = []
        for resdict in pose_motif_list:
            cat_res_pos.append(list(resdict.values())[0])
        cat_res_av_plddt = []
        for resposlist, plddtlist in zip(cat_res_pos, perresidue_plddt_list):
            plddts = [plddtlist[resnum - 1] for resnum in resposlist]
            cat_res_av_plddt.append(sum(plddts) / len(plddts))
        site_score = [score * av_residue_plddt / 100 for av_residue_plddt, score in zip(cat_res_av_plddt, poses.poses_df[sitescore_column].to_list())]

    pd.DataFrame({"poses_description": list(poses.poses_df["poses_description"]), metric_name: site_score}).to_json(scorefile)
    poses.poses_df.loc[:, metric_name] = site_score

    poses.poses_df.to_json(poses.scorefile)

    return poses

def save_top_poses(poses, cycle_num, output_dir):


    top_df = poses.poses_df[["poses", "poses_description", "input_description", f"cycle_{cycle_num}_attn_sc_bb_site_score", f"cycle_{cycle_num}_post_cm_esm_motif_site_score", f'cycle_{cycle_num}_post_cm_esm_bb_motif_rmsd', f'cycle_{cycle_num}_post_cm_attn_catres_motif_heavy_rmsd', 'updated_reference_frags_location', 'motif_residues', 'fixed_residues']].copy()
    if 'covalent_bonds' in poses.poses_df.columns:
        top_df['covalent_bonds'] = poses.poses_df['covalent_bonds'].copy()
    
    if 'params_file_path' in poses.poses_df.columns:
        top_df['params_file_path'] = poses.poses_df['params_file_path'].copy()

    top_df['cycle'] = cycle_num
    top_df.rename(columns={f"cycle_{cycle_num}_attn_sc_bb_site_score": "esm_catres_site_score", f"cycle_{cycle_num}_post_cm_esm_motif_site_score": "esm_motif_site_score", f'cycle_{cycle_num}_post_cm_esm_bb_motif_rmsd': "esm_bb_motif_rmsd", f'cycle_{cycle_num}_post_cm_attn_catres_motif_heavy_rmsd': "esm_catres_rmsd"}, inplace=True)
    top_df.to_json(os.path.join(output_dir, f"cycle_{cycle_num}_top.json"))


    return top_df

def create_mutations_resfiles(poses_df, mutations_column, output_dir, fixed_residues_column, prefix):
    #reads in mutations, writes resfiles, creates a new fixed residues column so that mutated residues can be kept during mpnn

    if f"{prefix}_fixed_residues" in poses_df.columns:
        logging.info(f"Found {prefix}_fixed_residues in poses_df! Skipping!")
        print(f"Found {prefix}_fixed_residues in poses_df! Skipping!")
        return(poses_df)
    resfiledir = os.path.join(output_dir, f"{prefix}_resfiles")
    os.makedirs(resfiledir, exist_ok=True)
    rows = []
    for index, row in poses_df.iterrows():
        mutations = row[mutations_column]
        resfilename = os.path.join(resfiledir, f"{prefix}_{row['poses_description']}")
        resfilepath_cm, resfilepath_rx, mutated_positions = create_resfile(mutations, resfilename)
        row[f"{prefix}_resfilepath_cm"] = resfilepath_cm
        row[f"{prefix}_resfilepath_rx"] = resfilepath_rx
        row[f"{prefix}_fixed_residues"] = copy.deepcopy(row[fixed_residues_column])
        if len(mutated_positions) > 0:
            for mut in mutated_positions:
                if mut not in row[f"{prefix}_fixed_residues"]['A']:
                    row[f"{prefix}_fixed_residues"]['A'].append(mut)

        rows.append(row)

    poses_df = pd.DataFrame(rows)

    return poses_df



def create_resfile(mutations, resfile_description):

    positions = []
    resfile_content = []
    if isinstance(mutations, str):
        resfile_content.append("start\n")
        for mutation in mutations.split(","):
            position, residues = mutation.split(":")
            if residues.startswith('-'):
                residues = residues[1:]
                resfileline = f"{position} A NOTAA {residues}\n"
            else:
                resfileline = f"{position} A PIKAA {residues}\n"
            resfile_content.append(resfileline)
            positions.append(int(position))
    resfile_cm = "".join(resfile_content)
    resfile_content = ["NATAA\n"] + resfile_content
    resfile_rx = "".join(resfile_content)
    resfilepath_cm =f"{resfile_description}_cm.resfile"
    resfilepath_rx = f"{resfile_description}_rx.resfile"
    with open(resfilepath_cm, 'w') as resfile:
        resfile.write(resfile_cm)
    with open(resfilepath_rx, 'w') as resfile:
        resfile.write(resfile_rx)
    return resfilepath_cm, resfilepath_rx, positions



def main(args):

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=f"{args.output_dir}/coupled_moves.log")
    cmd = ''
    for key, value in vars(args).items():
        cmd += f'--{key} {value} '
    cmd = f'{sys.argv[0]} {cmd}'
    logging.info(cmd)

    ######################## PREPARE INPUT ########################

    #absolute path for xml is necessary because coupled moves output can only be controlled by cd'ing into output directory before starting the run
    xml = os.path.abspath(args.cm_protocol)
    rx_xml = os.path.abspath(args.relax_protocol)
    pre_cm_rx_xml = os.path.abspath(args.pre_cm_relax_protocol)

    df = pd.read_json(args.json)
    #drop everything that is not needed to make df less cluttered
    for column in df.columns:
        if column not in ['poses', 'poses_description', 'updated_reference_frags_location', 'fixed_residues', 'motif_residues', 'af2_top_plddt', 'af2_top_plddt_list', 'input_poses', 'template_motif', 'template_fixedres', 'params_file_path', 'covalent_bonds']:
            df.drop(column, inplace=True, axis=1)

    coupled_moves = Poses(args.output_dir, df['poses'].to_list())

    input_dir = f'{args.output_dir}/input_pdbs/'
    input_dir_filtered = f'{args.output_dir}/input_pdbs/selected/'
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(input_dir_filtered, exist_ok=True)

    #set maximum number of cpus for coupled moves & relax runs
    coupled_moves.max_rosetta_cpus = args.max_cpus
    #merge pose dataframes
    coupled_moves.poses_df = coupled_moves.poses_df.drop(['input_poses', 'poses'], axis=1).merge(df, on='poses_description', how='left')

    coupled_moves.poses_df['input_description'] = coupled_moves.poses_df['poses_description'].copy()

    if args.use_reduced_motif in ['True', 'true', 'TRUE', '1', 'yes', 'YES', 'Yes']:
        logging.info('Using reduced motif!')
        print('Using reduced motif!')
        coupled_moves.poses_df['original_motif'] = coupled_moves.poses_df['motif_residues'].copy()
        coupled_moves.poses_df['motif_residues'] = coupled_moves.poses_df.apply(lambda row: create_reduced_motif(row['fixed_residues'], row['motif_residues']), axis=1)



    if args.mutations_csv:
        #read in desired mutations
        mutations_df = pd.read_csv(args.mutations_csv)
        coupled_moves.poses_df = coupled_moves.poses_df.merge(mutations_df, on='poses_description')
        #drop everything marked with 'x'
        coupled_moves.poses_df = coupled_moves.poses_df[~coupled_moves.poses_df['mutations'].isin(['X', 'x'])]
        #write resfiles
        coupled_moves.poses_df = create_mutations_resfiles(coupled_moves.poses_df, "mutations", args.output_dir, 'fixed_residues', 'mutations')
    else:
        resfiledir = os.path.join(args.output_dir, "mutations_resfiles")
        os.makedirs(resfiledir, exist_ok=True)
        resfilename = os.path.join(resfiledir, "no_mutations")
        resfilepath_cm, resfilepath_rx, positions = create_resfile(float('nan'), resfilename)
        coupled_moves.poses_df['mutations_resfilepath_cm'] = resfilepath_cm
        coupled_moves.poses_df['mutations_resfilepath_rx'] = resfilepath_rx
        coupled_moves.poses_df['mutations_fixed_residues'] = copy.deepcopy(coupled_moves.poses_df['fixed_residues'])

    #original_poses = copy.deepcopy(coupled_moves)

    #Use alphafold plddt of final prediction for site score
    coupled_moves = filter_input_poses(coupled_moves, f'af2_top_plddt_list', args.ligand_chain, args.sitescore_cutoff, args.max_input_per_backbone, input_dir, args.database_dir, args.bb_clash_vdw_multiplier)
    logging.info(f'{len(coupled_moves.poses_df.index)} input poses passed ligand clash filter.')
    print(f'{len(coupled_moves.poses_df.index)} input poses passed ligand clash filter.')



    options_list = []
    pose_list = []
    pre_cm_rx_options_list = []
    #iterate over dataframe, create coupled moves options
    for index, row in coupled_moves.poses_df.iterrows():
        #set filename for input pdbs
        pose_path = os.path.abspath(input_dir_filtered + row['poses_description'] + '.pdb')
        #add ligand to input pdbs
        si_tools.superimpose_add_chain_by_motif(row['updated_reference_frags_location'], row['poses'], args.ligand_chain, row['fixed_residues'], row['fixed_residues'], pose_path, ['N', 'CA', 'C', 'O'])
        #identify catalytic residues, format them for rosettascripts input
        cat_res = chainresdict_to_str(row['fixed_residues'])
        motif_res = chainresdict_to_str(row['motif_residues'])
        fixed_res = chainresdict_to_str(row['mutations_fixed_residues'])
        cm_script_vars = f"-parser:script_vars cat_res={cat_res} motif_res={motif_res} fixed_res={fixed_res} resfilepath={os.path.abspath(row['mutations_resfilepath_cm'])} cut1={args.cm_design_shell[0]} cut2={args.cm_design_shell[1]} cut3={args.cm_design_shell[2]} cut4={args.cm_design_shell[3]} favor_native_weight={args.cm_favor_native_weight}"
        pre_cm_rx_script_vars = f"-parser:script_vars cat_res={cat_res} motif_res={motif_res} resfilepath={os.path.abspath(row['mutations_resfilepath_rx'])}"
        if args.omit_AAs:
            cm_script_vars = cm_script_vars + f" prohibited_residues={','.join([aa_three_or_one_letter_code(AA) for AA in args.omit_AAs])}"
            pre_cm_rx_script_vars = pre_cm_rx_script_vars + f" prohibited_residues={','.join([aa_three_or_one_letter_code(AA) for AA in args.omit_AAs])}"
        pre_cm_rx_options = f"{pre_cm_rx_script_vars} -in:file:native {row['updated_reference_frags_location']}"
        cm_options =  f"{cm_script_vars} -in:file:native {row['updated_reference_frags_location']}"
        if 'params_file_path' in coupled_moves.poses_df.columns:
            cm_options = cm_options + f" -extra_res_fa {os.path.abspath(row['params_file_path'])}"
            pre_cm_rx_options = pre_cm_rx_options + f" -extra_res_fa {os.path.abspath(row['params_file_path'])}"

        options_list.append(cm_options)
        pose_list.append(pose_path)
        pre_cm_rx_options_list.append(pre_cm_rx_options)

    #update path to input pdbs, add coupled_moves options
    coupled_moves.poses_df['coupled_moves_options'] = options_list
    coupled_moves.poses_df['pre_cm_rx_opts'] = pre_cm_rx_options_list
    coupled_moves.poses_df['poses'] = pose_list


    cols = [f"af2_top_plddt", "pre_cm_bb_motif_rmsd", "pre_cm_catres_motif_heavy_rmsd", "pre_cm_motif_site_score", "pre_cm_catres_site_score"]
    titles = ["AF2-pLDDT", "AF2 Motif bb RMSD", "AF2 Catres\nSidechain RMSD", "motif site score", "catres site score"]
    y_labels = ["pLDDT", "RMSD [\u00C5]", "RMSD [\u00C5]", "AU", "AU"]
    dims = [(0,100), (0,5), (0,5), (0,1), (0,1)]
    _ = plots.violinplot_multiple_cols(coupled_moves.poses_df, cols=cols, titles=titles, y_labels=y_labels, dims=dims, out_path=f"{args.output_dir}/plots/pre_cm.png")

    cm_opts = f"-parser:protocol {xml} -coupled_moves:ligand_mode true -coupled_moves:ligand_weight {args.cm_ligand_weight}"
    if args.cm_options:
        cm_opts = cm_opts + ' ' + args.cm_options

    rx_opts = f"-parser:protocol {rx_xml}"
    pre_cm_rx_opts = f"-parser:protocol {pre_cm_rx_xml}"
    if args.relax_options:
        rx_opts = rx_opts + ' ' + args.relax_options
        pre_cm_rx_opts = pre_cm_rx_opts + ' ' + args.relax_options

    best_per_cycle_dir = os.path.join(args.output_dir, "best_per_cycle")
    os.makedirs(best_per_cycle_dir, exist_ok=True)




    ######################## COUPLED MOVES, MPNN, ESMFOLD, ATTN REFINEMENT ########################


    for cycle_num in range(1, args.cycles + 1):


        #relax all poses to optimize rmsds, introduce mutations
        if 'covalent_bonds' in coupled_moves.poses_df.columns:
            logging.info('Covalent bonds present! Adding LINK records to poses...')
            print('Covalent bonds present! Adding LINK records to poses...')
            coupled_moves.add_LINK_to_poses('covalent_bonds', f'cycle_{cycle_num}_pre_cm')
        coupled_moves.rosetta("rosetta_scripts.default.linuxgccrelease", options=pre_cm_rx_opts, pose_options=coupled_moves.poses_df['pre_cm_rx_opts'].to_list(), n=3, prefix=f'cycle_{cycle_num}_pre_cm_relax')
        coupled_moves.poses_df.to_json(coupled_moves.scorefile)
        #identify relaxed structure with lowest total score for each input
        coupled_moves.filter_poses_by_score(1, f'cycle_{cycle_num}_pre_cm_relax_total_score', remove_layers=1, ascending=True)
        logging.info(f'{len(coupled_moves.poses_df.index)} poses selected after relax!')

        if cycle_num == 1:
            original_poses = copy.deepcopy(coupled_moves)

        #create working directory for coupled moves, cd into it (because cm output is always generated in starting directory)
        starting_dir = os.getcwd()
        working_dir = f'{args.output_dir}/cycle_{cycle_num}_cm_working_dir/'
        os.makedirs(working_dir, exist_ok=True)
        os.chdir(working_dir)

        #create copy of poses, because they will be overwritten after running rosettascripts
        old_poses = copy.deepcopy(coupled_moves)

        #run coupled moves
        coupled_moves.rosetta("rosetta_scripts.default.linuxgccrelease", options=cm_opts, pose_options=coupled_moves.poses_df['coupled_moves_options'].to_list(), n=args.cm_nstruct, prefix=f'cycle_{cycle_num}_coupled_moves')

        #return to starting directory
        os.chdir(starting_dir)

        coupled_moves.poses_df.to_json(coupled_moves.scorefile)

        #create results dir for coupled moves
        cm_resultsdir = f'{args.output_dir}/cycle_{cycle_num}_cm_results/'
        os.makedirs(cm_resultsdir, exist_ok=True)

        #restore old poses
        coupled_moves = old_poses

        #refine coupled moves output
        relax_input_dir = f'{args.output_dir}/cycle_{cycle_num}_cm_relax_input/'
        coupled_moves.poses_df = prepare_coupled_moves_relax_mpnn(relax_input_dir, working_dir, cm_resultsdir, coupled_moves.poses_df, args.cm_occurence_cutoff, cycle_num)


        #relax all variants suggested by coupled moves
        if 'covalent_bonds' in coupled_moves.poses_df.columns:
            logging.info('Covalent bonds present! Adding LINK records to poses...')
            print('Covalent bonds present! Adding LINK records to poses...')
            coupled_moves.add_LINK_to_poses('covalent_bonds', 'rm_rx')
        coupled_moves.rosetta("rosetta_scripts.default.linuxgccrelease", options=rx_opts, pose_options=coupled_moves.poses_df[f'cycle_{cycle_num}_cm_relax_opts'].to_list(), n=args.relax_nstruct, prefix=f'cycle_{cycle_num}_cm_relax')
        #identify relaxed structure with lowest total score for each variant
        coupled_moves.filter_poses_by_score(1, f'cycle_{cycle_num}_cm_relax_total_score', remove_layers=1, ascending=True)
        logging.info(f'{len(coupled_moves.poses_df.index)} poses selected after relax!')

        #filter out top variants
        coupled_moves.filter_poses_by_score(args.mpnn_max_input, f'cycle_{cycle_num}_cm_relax_total_score', remove_layers=2, ascending=True)
        print(f'{len(coupled_moves.poses_df.index)} variants selected after relax!')
        logging.info(f'{len(coupled_moves.poses_df.index)} variants selected after relax!')
        #preserve pre-mpnn sequences
        if cycle_num == 1:
            preserved_poses = copy.deepcopy(coupled_moves)
            preserved_path = os.path.join(args.output_dir, "cm_out_preserved.json")
            logging.info(f'Writing preserved poses to {preserved_path}!')
            coupled_moves.poses_df.to_json(preserved_path)


        #run mpnn on relaxed variants, keep coupled-moves suggested positions fixed
        mpnn_opts = f"--num_seq_per_target={args.mpnn_nstruct} --sampling_temp={args.mpnn_temp}"
        if args.omit_AAs:
            mpnn_opts = mpnn_opts + f" --omit_AAs {args.omit_AAs}"
        coupled_moves.mpnn_design(mpnn_options=mpnn_opts, prefix=f"cycle_{cycle_num}_cm_mpnn", fixed_positions_col=f"cycle_{cycle_num}_cm_fixed_positions", use_soluble_model=True)
        coupled_moves.filter_poses_by_score(args.mpnn_max_output, f"cycle_{cycle_num}_cm_mpnn_global_score", remove_layers=1, prefix=f"cycle_{cycle_num}_cm_mpnn_seqfilter")

        #predict mpnn sequences, analyze output
        coupled_moves.predict_sequences(run_ESMFold, prefix=f"cycle_{cycle_num}_cm_predictions_esm")
        coupled_moves.calc_motif_bb_rmsd_df('updated_reference_frags_location', 'motif_residues', 'motif_residues', f'cycle_{cycle_num}_post_cm_esm_bb', ['N', 'CA', 'C'])
        coupled_moves.add_site_score(f'cycle_{cycle_num}_post_cm_esm_bb_motif_rmsd', 'motif_residues', f"cycle_{cycle_num}_cm_predictions_esm_perresidue_plddt", f'cycle_{cycle_num}_post_cm_esm_motif')
        coupled_moves.calc_motif_heavy_rmsd_df('updated_reference_frags_location', 'fixed_residues', 'fixed_residues', f'cycle_{cycle_num}_post_cm_esm_catres')
        coupled_moves.add_site_score(f'cycle_{cycle_num}_post_cm_esm_catres_motif_heavy_rmsd', 'fixed_residues', f"cycle_{cycle_num}_cm_predictions_esm_perresidue_plddt", f'cycle_{cycle_num}_post_cm_esm_catres')
        coupled_moves = clash_detection(poses=coupled_moves, ref_frags_col="updated_reference_frags_location", ref_motif_col="fixed_residues", poses_motif_col="fixed_residues", prefix=f"cycle_{cycle_num}_post_cm_esm", ligand_chain=args.ligand_chain, database_dir=args.database_dir, bb_clash_vdw_multiplier=args.bb_clash_vdw_multiplier, save_path_list=coupled_moves.poses_df['poses'].to_list())
        coupled_moves.poses_df.to_json(coupled_moves.scorefile)

        #filter predictions below plddt cutoff & clashing predictions
        print(f'Filtering {len(coupled_moves.poses_df.index)} poses...')
        logging.info(f'Filtering {len(coupled_moves.poses_df.index)} poses...')
        coupled_moves.poses_df = coupled_moves.poses_df[coupled_moves.poses_df[f"cycle_{cycle_num}_cm_predictions_esm_plddt"] > args.plddt_cutoff]
        print(f'{len(coupled_moves.poses_df.index)} passed plddt cutoff of {args.plddt_cutoff}.')
        logging.info(f'{len(coupled_moves.poses_df.index)} passed plddt cutoff of {args.plddt_cutoff}.')
        coupled_moves.poses_df = coupled_moves.poses_df[coupled_moves.poses_df[f"cycle_{cycle_num}_post_cm_esm_ligand_clash"] == False]
        print(f'{len(coupled_moves.poses_df.index)} passed clash detection.')
        logging.info(f'{len(coupled_moves.poses_df.index)} passed clash detection.')

        #run attn, filter by sitescore and top poses per backbone
        coupled_moves = run_attn(coupled_moves, prefix=f"cycle_{cycle_num}_attn")
        coupled_moves.calc_motif_heavy_rmsd_df('updated_reference_frags_location', 'fixed_residues', 'fixed_residues', f'cycle_{cycle_num}_post_cm_attn_catres')
        coupled_moves.add_site_score(f'cycle_{cycle_num}_post_cm_attn_catres_motif_heavy_rmsd', 'fixed_residues', f"cycle_{cycle_num}_attn_sc_plddts", f'cycle_{cycle_num}_post_cm_attn_catres')
        coupled_moves = update_sitescore_with_bb_plddts(coupled_moves, f"cycle_{cycle_num}_post_cm_attn_catres_site_score", f"cycle_{cycle_num}_cm_predictions_esm_perresidue_plddt", 'fixed_residues', f"cycle_{cycle_num}_attn")
        coupled_moves.filter_poses_by_score(args.max_output_per_backbone, f"cycle_{cycle_num}_attn_sc_bb_site_score", remove_layers=4*cycle_num, ascending=False)
        logging.info(f'{len(coupled_moves.poses_df.index)} poses were selected for next round!')
        print(f'{len(coupled_moves.poses_df.index)} poses were selected for next round!')
        coupled_moves.poses_df.to_json(coupled_moves.scorefile)


        #create output directory
        cycle_resultsdir = os.path.join(args.output_dir, f'cycle_{cycle_num}_results/')
        os.makedirs(cycle_resultsdir, exist_ok=True)
        ref_frags = [shutil.copy(ref_pose, cycle_resultsdir) for ref_pose in coupled_moves.poses_df["input_poses"].to_list()]
        cycle_results = [shutil.copy(pose, cycle_resultsdir) for pose in coupled_moves.poses_df["poses"].to_list()]

        #create alignments and plots
        utils.pymol_tools.pymol_alignment_scriptwriter(df=coupled_moves.poses_df, scoreterm=f"cycle_{cycle_num}_attn_sc_bb_site_score", top_n=len(coupled_moves.poses_df.index), path_to_script=f'{cycle_resultsdir}align.pml', ascending=False, pose_col='poses_description', ref_pose_col='input_poses', motif_res_col="motif_residues", fixed_res_col="fixed_residues", ref_motif_res_col="template_motif", ref_fixed_res_col="template_fixedres")
        cols = [f'cycle_{cycle_num}_cm_predictions_esm_plddt', f'cycle_{cycle_num}_post_cm_esm_bb_motif_rmsd', f'cycle_{cycle_num}_post_cm_attn_catres_motif_heavy_rmsd', f'cycle_{cycle_num}_post_cm_esm_motif_site_score', f'cycle_{cycle_num}_post_cm_attn_catres_site_score']
        titles = ["ESM-pLDDT", "ESM Motif RMSD", "ATTN Catres\nSidechain RMSD", "motif site score", "catres site score"]
        y_labels = ["pLDDT", "RMSD [\u00C5]", "RMSD [\u00C5]", "AU", "AU"]
        dims = [(0,100), (0,5), (0,5), (0,1), (0,1)]
        _ = plots.violinplot_multiple_cols(coupled_moves.poses_df, cols=cols, titles=titles, y_labels=y_labels, dims=dims, out_path=f"{args.output_dir}/plots/cycle_{cycle_num}_results.png")

        esm_ligated_dir = f'{args.output_dir}/cycle_{cycle_num}_esm_ligated/'
        os.makedirs(esm_ligated_dir, exist_ok=True)
        pose_list = []
        for index, row in coupled_moves.poses_df.iterrows():
            #use absolute paths for next round of coupled moves
            pose_path = os.path.abspath(f"{esm_ligated_dir}/{row['poses_description']}.pdb")
            si_tools.superimpose_add_chain_by_motif(row['updated_reference_frags_location'], row['poses'], args.ligand_chain, row['fixed_residues'], row['fixed_residues'], pose_path, ['N', 'CA', 'C', 'O'])
            pose_list.append(pose_path)

        coupled_moves.poses_df['poses'] = pose_list
        coupled_moves.poses_df.to_json(coupled_moves.scorefile)

        top_df = save_top_poses(coupled_moves, cycle_num, best_per_cycle_dir)


    ######################## RUN PREDICTION ON PRESERVED SEQUENCES WITH/WITHOUT CM-OPTIMIZATION ONLY (NO MPNN) ########################


    coupled_moves = preserved_poses
    cycle_num = 0

    #predict coupled moves sequences, analyze output
    coupled_moves.predict_sequences(run_ESMFold, prefix=f"cycle_{cycle_num}_cm_predictions_esm")
    coupled_moves.calc_motif_bb_rmsd_df('updated_reference_frags_location', 'motif_residues', 'motif_residues', f'cycle_{cycle_num}_post_cm_esm_bb', ['N', 'CA', 'C'])
    coupled_moves.add_site_score(f'cycle_{cycle_num}_post_cm_esm_bb_motif_rmsd', 'motif_residues', f"cycle_{cycle_num}_cm_predictions_esm_perresidue_plddt", f'cycle_{cycle_num}_post_cm_esm_motif')
    coupled_moves.calc_motif_heavy_rmsd_df('updated_reference_frags_location', 'fixed_residues', 'fixed_residues', f'cycle_{cycle_num}_post_cm_esm_catres')
    coupled_moves.add_site_score(f'cycle_{cycle_num}_post_cm_esm_catres_motif_heavy_rmsd', 'fixed_residues', f"cycle_{cycle_num}_cm_predictions_esm_perresidue_plddt", f'cycle_{cycle_num}_post_cm_esm_catres')
    coupled_moves = clash_detection(poses=coupled_moves, ref_frags_col="updated_reference_frags_location", ref_motif_col="fixed_residues", poses_motif_col="fixed_residues", prefix=f"cycle_{cycle_num}_post_cm_esm", ligand_chain=args.ligand_chain, database_dir=args.database_dir, bb_clash_vdw_multiplier=args.bb_clash_vdw_multiplier, save_path_list=coupled_moves.poses_df['poses'].to_list())

    #filter predictions below plddt cutoff & clashing predictions
    print(f'Filtering {len(coupled_moves.poses_df.index)} poses...')
    logging.info(f'Filtering {len(coupled_moves.poses_df.index)} poses...')
    coupled_moves.poses_df = coupled_moves.poses_df[coupled_moves.poses_df[f"cycle_{cycle_num}_cm_predictions_esm_plddt"] > args.plddt_cutoff]
    print(f'{len(coupled_moves.poses_df.index)} passed plddt cutoff of {args.plddt_cutoff}.')
    logging.info(f'{len(coupled_moves.poses_df.index)} passed plddt cutoff of {args.plddt_cutoff}.')
    coupled_moves.poses_df = coupled_moves.poses_df[coupled_moves.poses_df[f"cycle_{cycle_num}_post_cm_esm_ligand_clash"] == False]
    print(f'{len(coupled_moves.poses_df.index)} passed clash detection.')
    logging.info(f'{len(coupled_moves.poses_df.index)} passed clash detection.')

    #run attn, filter by sitescore and top poses per backbone
    coupled_moves = run_attn(coupled_moves, prefix=f"cycle_{cycle_num}_attn")
    coupled_moves.calc_motif_heavy_rmsd_df('updated_reference_frags_location', 'fixed_residues', 'fixed_residues', f'cycle_{cycle_num}_post_cm_attn_catres')
    coupled_moves.add_site_score(f'cycle_{cycle_num}_post_cm_attn_catres_motif_heavy_rmsd', 'fixed_residues', f"cycle_{cycle_num}_attn_sc_plddts", f'cycle_{cycle_num}_post_cm_attn_catres')
    coupled_moves = update_sitescore_with_bb_plddts(coupled_moves, f"cycle_{cycle_num}_post_cm_attn_catres_site_score", f"cycle_{cycle_num}_cm_predictions_esm_perresidue_plddt", 'fixed_residues', f"cycle_{cycle_num}_attn")
    coupled_moves.filter_poses_by_score(args.max_output_per_backbone, f"cycle_{cycle_num}_attn_sc_bb_site_score", remove_layers=3, ascending=False)

    top_df = save_top_poses(coupled_moves, cycle_num, best_per_cycle_dir)

    ## repredict original input to get comparable data ##
    cycle_num = -1
    coupled_moves = original_poses
    coupled_moves.predict_sequences(run_ESMFold, prefix=f"cycle_{cycle_num}_cm_predictions_esm")
    coupled_moves.calc_motif_bb_rmsd_df('updated_reference_frags_location', 'motif_residues', 'motif_residues', f'cycle_{cycle_num}_post_cm_esm_bb', ['N', 'CA', 'C'])
    coupled_moves.add_site_score(f'cycle_{cycle_num}_post_cm_esm_bb_motif_rmsd', 'motif_residues', f"cycle_{cycle_num}_cm_predictions_esm_perresidue_plddt", f'cycle_{cycle_num}_post_cm_esm_motif')
    coupled_moves.calc_motif_heavy_rmsd_df('updated_reference_frags_location', 'fixed_residues', 'fixed_residues', f'cycle_{cycle_num}_post_cm_esm_catres')
    coupled_moves.add_site_score(f'cycle_{cycle_num}_post_cm_esm_catres_motif_heavy_rmsd', 'fixed_residues', f"cycle_{cycle_num}_cm_predictions_esm_perresidue_plddt", f'cycle_{cycle_num}_post_cm_esm_catres')
    coupled_moves = clash_detection(poses=coupled_moves, ref_frags_col="updated_reference_frags_location", ref_motif_col="fixed_residues", poses_motif_col="fixed_residues", prefix=f"cycle_{cycle_num}_post_cm_esm", ligand_chain=args.ligand_chain, database_dir=args.database_dir, bb_clash_vdw_multiplier=args.bb_clash_vdw_multiplier, save_path_list=coupled_moves.poses_df['poses'].to_list())

    #filter predictions below plddt cutoff & clashing predictions
    print(f'Filtering {len(coupled_moves.poses_df.index)} poses...')
    logging.info(f'Filtering {len(coupled_moves.poses_df.index)} poses...')
    coupled_moves.poses_df = coupled_moves.poses_df[coupled_moves.poses_df[f"cycle_{cycle_num}_cm_predictions_esm_plddt"] > args.plddt_cutoff]
    print(f'{len(coupled_moves.poses_df.index)} passed plddt cutoff of {args.plddt_cutoff}.')
    logging.info(f'{len(coupled_moves.poses_df.index)} passed plddt cutoff of {args.plddt_cutoff}.')
    coupled_moves.poses_df = coupled_moves.poses_df[coupled_moves.poses_df[f"cycle_{cycle_num}_post_cm_esm_ligand_clash"] == False]
    print(f'{len(coupled_moves.poses_df.index)} passed clash detection.')
    logging.info(f'{len(coupled_moves.poses_df.index)} passed clash detection.')

    #run attn, filter by sitescore and top poses per backbone
    coupled_moves = run_attn(coupled_moves, prefix=f"cycle_{cycle_num}_attn")
    coupled_moves.calc_motif_heavy_rmsd_df('updated_reference_frags_location', 'fixed_residues', 'fixed_residues', f'cycle_{cycle_num}_post_cm_attn_catres')
    coupled_moves.add_site_score(f'cycle_{cycle_num}_post_cm_attn_catres_motif_heavy_rmsd', 'fixed_residues', f"cycle_{cycle_num}_attn_sc_plddts", f'cycle_{cycle_num}_post_cm_attn_catres')
    coupled_moves = update_sitescore_with_bb_plddts(coupled_moves, f"cycle_{cycle_num}_post_cm_attn_catres_site_score", f"cycle_{cycle_num}_cm_predictions_esm_perresidue_plddt", 'fixed_residues', f"cycle_{cycle_num}_attn")

    top_df = save_top_poses(coupled_moves, cycle_num, best_per_cycle_dir)

    ######################## ALPHAFOLD2 & ATTN ########################

    #identify best outputs from all cycles
    top_df = pd.concat([pd.read_json(os.path.join(best_per_cycle_dir, json)) for json in [file for file in os.listdir(best_per_cycle_dir) if file.startswith("cycle_") and file.endswith("_top.json")]]).reset_index(drop=True)
    top_df = pd.concat([df.sort_values("esm_catres_site_score", ascending=False).head(args.max_output_per_backbone * 3) for input_pdb, df in top_df.groupby("input_description")]).reset_index(drop=True)

    out_paths = []
    for index, row in top_df.iterrows():
        out_path = os.path.join(best_per_cycle_dir, f"{row['poses_description']}.pdb")
        shutil.copy(row['poses'], out_path)
        #I don't know why, but automatic conversion to fasta crashes --> manually converted them here
        fasta_path = write_fasta(out_path, get_protein_sequence(row['poses']))
        out_paths.append(fasta_path)
    top_df['poses'] = out_paths
    top_df.to_json(os.path.join(best_per_cycle_dir, "top.json"))

    coupled_moves.poses_df = top_df

    logging.info(f'{len(coupled_moves.poses_df.index)} top poses of all cycles selected for prediction with alphafold!')

    #run alphafold on best structures
    coupled_moves.poses_df['post_cm_esm_location'] = coupled_moves.poses_df['poses']
    coupled_moves.predict_sequences(run_AlphaFold2, options="--msa-mode single_sequence ", prefix="cm_predictions_af2")
    #coupled_moves.calc_bb_rmsd_df(ref_pdb='post_cm_esm_location', metric_prefix="post_cm_af2_esm") TODO: this crashes for some reason
    coupled_moves.calc_motif_bb_rmsd_df(ref_pdb="updated_reference_frags_location", ref_motif="motif_residues", target_motif="motif_residues", metric_prefix="post_cm_af2_bb", atoms=['N', 'CA', 'C', 'O'])
    coupled_moves.add_site_score('post_cm_af2_bb_motif_rmsd', 'motif_residues', convert_af2_perresidue_plddt_to_list(coupled_moves.poses_df, 'cm_predictions_af2_top_plddt_list'), 'post_cm_af2_motif')
    coupled_moves = clash_detection(poses=coupled_moves, ref_frags_col="updated_reference_frags_location", ref_motif_col="fixed_residues", poses_motif_col="fixed_residues", prefix="post_cm_af2", ligand_chain=args.ligand_chain, database_dir=args.database_dir, bb_clash_vdw_multiplier=args.bb_clash_vdw_multiplier, save_path_list=coupled_moves.poses_df['poses'].to_list())
    coupled_moves.poses_df.to_json(coupled_moves.scorefile)


    #filter af2 output
    initial_length = len(coupled_moves.poses_df.index)
    coupled_moves.poses_df = coupled_moves.poses_df[coupled_moves.poses_df["post_cm_af2_ligand_clash"] == False]
    logging.info(f'{len(coupled_moves.poses_df.index)} passed clash detection.')
    print(f'{len(coupled_moves.poses_df.index)} passed clash detection.')
    coupled_moves.poses_df = coupled_moves.poses_df[coupled_moves.poses_df["cm_predictions_af2_top_plddt"] > args.plddt_cutoff]
    logging.info(f'{len(coupled_moves.poses_df.index)} passed plddt cutoff of {args.plddt_cutoff}.')
    print(f'{len(coupled_moves.poses_df.index)} passed plddt cutoff of {args.plddt_cutoff}.')

    #run attn & calculate scores
    coupled_moves = run_attn(coupled_moves, prefix=f"af2_attn")
    coupled_moves.calc_motif_heavy_rmsd_df(ref_pdb="updated_reference_frags_location", ref_motif="fixed_residues", target_motif="fixed_residues", metric_prefix="post_cm_attn_catres")
    coupled_moves.add_site_score('post_cm_attn_catres_motif_heavy_rmsd', 'fixed_residues', 'af2_attn_sc_plddts', 'post_cm_attn_catres')
    coupled_moves.poses_df['af2_perresidue_plddt_list'] = convert_af2_perresidue_plddt_to_list(coupled_moves.poses_df, 'cm_predictions_af2_top_plddt_list')
    coupled_moves = update_sitescore_with_bb_plddts(coupled_moves, 'post_cm_attn_catres_site_score', 'af2_perresidue_plddt_list', 'fixed_residues', f"post_cm_attn")

    #filter output
    coupled_moves.poses_df['af2_esm_combined_catres_sitescore'] = coupled_moves.poses_df["post_cm_attn_sc_bb_site_score"] * coupled_moves.poses_df[f"esm_catres_site_score"]
    coupled_moves.poses_df = coupled_moves.poses_df[coupled_moves.poses_df['af2_esm_combined_catres_sitescore'] >= args.combined_sitescore_cutoff]
    print([row['input_description'] for i, row in coupled_moves.poses_df.iterrows()])
    coupled_moves.poses_df = pd.concat([df.sort_values("af2_esm_combined_catres_sitescore", ascending=False).head(args.max_output_per_backbone) for input_pdb, df in coupled_moves.poses_df.groupby("input_description")]).reset_index(drop=True)
    print(f'{len(coupled_moves.poses_df.index)} poses passed all filters.')

    # if option is set, dock:
    if args.run_docking.lower() == "true":
        # add ligand:
        stats_scoreterms = ["final_dock_dG", "final_dock_total_score", "final_dock_lig_rms", f"final_dock_interface_delta_{args.ligand_chain}", "final_dock_fa_atr"]

        lig_poses = coupled_moves.add_ligand_from_ref(ref_col="updated_reference_frags_location", ref_motif="motif_residues", target_motif="motif_residues", lig_chain=args.ligand_chain, prefix="docking_lig_poses")
        docking_options = f"-parser:protocol {args.docking_protocol}"
        docking_pose_opts = [f"-parser:script_vars native={pose} ligchain={args.ligand_chain}" for pose in coupled_moves.poses_df["poses"].to_list()]
        docked_poses = rosetta_scripts_and_mean(coupled_moves, prefix="final_dock", n=50, options=docking_options, pose_options=docking_pose_opts, filter_scoreterm="final_dock_lig_rms", scoreterms=stats_scoreterms, min_scoreterms=stats_scoreterms, std_scoreterms=stats_scoreterms)

    ################# Reindex poses before output #################################
    coupled_moves.reindex_poses(out_dir="cm_reindexed", remove_layers=2, keep_layers=True)

    # if specified, calculate protparams:
    if args.calc_protparams.lower() == "true": coupled_moves.calc_protparams(prefix="cm")

    #create output directory
    resultsdir = os.path.join(args.output_dir, 'results/')
    os.makedirs(resultsdir, exist_ok=True)

    coupled_moves.poses_df[["poses_description", "cm_predictions_af2_top_plddt", "post_cm_af2_bb_motif_rmsd", "post_cm_attn_catres_motif_heavy_rmsd", "post_cm_af2_motif_site_score", "post_cm_attn_catres_site_score", f"esm_catres_site_score", f"esm_motif_site_score", f'esm_bb_motif_rmsd', f'esm_catres_rmsd', 'af2_esm_combined_catres_sitescore']].sort_values('af2_esm_combined_catres_sitescore', ascending=False).to_csv(f"{resultsdir}/af2_results.csv")
    #"post_cm_af2_esm_bb_ca_rmsd"

    ref_frags = [shutil.copy(ref_pose, resultsdir) for ref_pose in coupled_moves.poses_df["updated_reference_frags_location"].to_list()]
    cm_af2_results = [shutil.copy(af2_pose, resultsdir) for af2_pose in coupled_moves.poses_df["poses"].to_list()]


    utils.pymol_tools.pymol_alignment_scriptwriter(df=coupled_moves.poses_df, scoreterm='af2_esm_combined_catres_sitescore', top_n=len(coupled_moves.poses_df.index), path_to_script=f'{resultsdir}align.pml', ascending=False, pose_col='poses_description', ref_pose_col='updated_reference_frags_location', motif_res_col="motif_residues", fixed_res_col="fixed_residues", ref_motif_res_col="motif_residues", ref_fixed_res_col="fixed_residues")

    cols = [f"cm_predictions_af2_top_plddt", "post_cm_af2_bb_motif_rmsd", "post_cm_attn_catres_motif_heavy_rmsd", "post_cm_af2_motif_site_score", "post_cm_attn_catres_site_score"]
    titles = ["AF2-pLDDT", "AF2 Motif RMSD", "ATTN Catres\nSidechain RMSD", "motif site score", "catres site score"]
    y_labels = ["pLDDT", "RMSD [\u00C5]", "RMSD [\u00C5]", "AU", "AU"]
    dims = [(0,100), (0,5), (0,5), (0,1), (0,1)]
    _ = plots.violinplot_multiple_cols(coupled_moves.poses_df, cols=cols, titles=titles, y_labels=y_labels, dims=dims, out_path=f"{args.output_dir}/plots/post_cm.png")

    # store sequences
    coupled_moves.store_sequences(prefix="final_sequence")
    coupled_moves.poses_df.to_json(coupled_moves.scorefile)

    logging.info("Done!")


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #general options
    argparser.add_argument("--output_dir", type=str, required=True, help="working directory")
    argparser.add_argument("--json", type=str, required=True, help="path to jsonfile containing information about input pdbs (catres etc)")
    argparser.add_argument("--ligand_chain", type=str, default='Z', help="should always be Z in current workflow")
    argparser.add_argument("--max_input_per_backbone", type=int, default=5, help="maximum number of input pdbs coming from identical rfdiffusion backbones")
    argparser.add_argument("--max_output_per_backbone", type=int, default=5, help="maximum number of output pdbs coming from identical starting structures (= number of structures predicted with alphafold)")
    argparser.add_argument("--max_cpus", type=int, default=320, help="maximum number of cpus for coupled_moves and relax runs")
    argparser.add_argument("--cycles", type=int, default=3, help="maximum number of cycles the refinement should be run")
    argparser.add_argument("--database_dir", type=str, default="/home/tripp/riffdiff2/riff_diff/database/", help="Path to folder containing rotamer libraries, fragment library, etc.")
    argparser.add_argument("--use_reduced_motif", type=str, default=True, help="Only fix residues +- 1 around catalytic residues during trajectory.")
    argparser.add_argument("--mutations_csv", type=str, default=None, help="Read in information about mutations (e.g. to keep channel open). Format should be 'resnumA:resids,resnumB:resids' etc. E.g. '35:AGS' only allows Ala, Gly or Ser at position 35. Residue IDs can be prohibited by using 35:-KR --> excludes Lys and Arg at pos 35. A small 'x' indicates this pose should not be passed on for the cm-pipeline. Mutations should be separated by a comma and put into quotation marks")
    argparser.add_argument("--omit_AAs", type=str, default="C", help="Prevent these residues from being built during mpnn/coupled moves. Use 1-letter code: 'CH' if no cysteines and histidines should be incorporated.")
    argparser.add_argument("--calc_protparams", type=str, default="False", help="Calculate Protparams for final output structures?")

    #coupled moves options
    argparser.add_argument("--cm_protocol", type=str, default='/home/mabr3112/riff_diff/rosetta/coupled_moves.xml', help="path to xmlfile that should be used for coupled moves")
    argparser.add_argument("--cm_nstruct", type=int, default=50, help="coupled_moves runs per input pdb")
    argparser.add_argument("--cm_design_shell", default=[6, 8, 10, 12], nargs=4, help="Design shells around ligand in Angstrom. All below cut1 is set to designable, all within cut2 is set to designable if Calpha Cbeta vector points to ligand, all within cut3 is set to repack, all within cut4 is set to repack if pointing to ligand.")
    argparser.add_argument("--cm_favor_native_weight", type=float, default=1.0, help="Weight for favoring input (mpnn) derived sequence. Recommended values 0 to 2")
    argparser.add_argument("--cm_options", type=str, default=None, help="additional coupled moves cmd-line arguments in string format. Use absolute paths!")
    argparser.add_argument("--cm_occurence_cutoff", type=float, default=0.25, help="Set how common a mutation has to be to be accepted")
    argparser.add_argument("--cm_ligand_weight", type=float, default=1, help="Weight for protein-ligand interactions during coupled moves. Recommended values 1 to 2.")

    #relax options
    argparser.add_argument("--relax_protocol", type=str, default='/home/mabr3112/riff_diff/rosetta/coupled_moves_relax_new.xml', help="path to xmlfile that should be used for coupled moves relax runs")
    argparser.add_argument("--pre_cm_relax_protocol", type=str, default='/home/mabr3112/riff_diff/rosetta/relax.xml', help="path to xmlfile that should be used for relax runs before running coupled moves")
    argparser.add_argument("--relax_nstruct", type=int, default=3, help="relax runs per coupled moves variant")
    argparser.add_argument("--relax_options", type=str, default=None, help="additional relax cmd-line arguments in string format.")

    #mpnn options
    argparser.add_argument("--mpnn_nstruct", type=int, default=100, help="number of mpnn output sequences per relaxed variant")
    argparser.add_argument("--mpnn_temp", type=float, default=0.1, help="temperature for mpnn runs")

    #filter options
    argparser.add_argument("--sitescore_cutoff", type=float, default=None, help="cutoff for site score (1/e**(motif_rmsd) * motif_plddt / 100) for filtering input pdbs. Recommended values are 0.2 to 0.5")
    argparser.add_argument("--bb_clash_vdw_multiplier", type=float, default=0.7, help="Multiplier for VanderWaals radii for clash detection between backbone and ligand. Clash is detected if distance_between_atoms < (VdW_radius_atom1 + VdW_radius_atom2)*multiplier")
    argparser.add_argument("--mpnn_max_input", type=int, default=5, help="maximum number of variants that should be passed to mpnn (ranked by total score of relaxed structure)")
    argparser.add_argument("--mpnn_max_output", type=float, default=30, help="maximum number of mpnn sequences per relaxed variant that should be passed on for structure prediction (ranked by global score)")
    argparser.add_argument("--plddt_cutoff", type=float, default=80, help="plddt cutoff for predictions that should be passed to the next round of coupled moves refinement")
    argparser.add_argument("--combined_sitescore_cutoff", type=float, default=0.05, help="Combined ESM & AF2 sitescore cutoff for output PDBs.")

    # docking options
    argparser.add_argument("--run_docking", default="False", type=str, help="Do you want to start a docking run with your final coupled moves outputs?")
    argparser.add_argument("--docking_protocol", default="/home/mabr3112/riff_diff/rosetta/GA_dock.xml", type=str, help="Docking Protocol at the end of Coupled Moves.")

    args = argparser.parse_args()

    main(args)
