#!/home/tripp/anaconda3/envs/riffdiff/bin/python3.11

import logging
import os
import sys
import copy

# import dependencies
import warnings
import Bio
from Bio.PDB import *
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import time
import itertools
sys.path.append("/home/mabr3112/riff_diff")


# import custom modules
#sys.path.append("/home/tripp/riff_diff/")

import utils.adrian_utils as utils
import utils.plotting as plots

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

def tip_symmetric_residues():
    symres = ["ARG", "ASP", "GLU", "LEU", "PHE", "TYR", "VAL"]
    return symres

def return_residue_rotamer_library(library_folder:str, residue_identity:str):
    '''
    finds the correct library for a given amino acid and drops not needed chi angles
    '''
    library_folder = utils.path_ends_with_slash(library_folder)
    prefix = residue_identity.lower()
    rotlib = pd.read_csv(f'{library_folder}{prefix}.bbdep.rotamers.lib')
    if residue_identity in AAs_up_to_chi3():
        rotlib.drop(['chi4', 'chi4sig'], axis=1, inplace=True)
    elif residue_identity in AAs_up_to_chi2():
        rotlib.drop(['chi3', 'chi3sig', 'chi4', 'chi4sig'], axis=1, inplace=True)
    elif residue_identity in AAs_up_to_chi1():
        rotlib.drop(['chi2', 'chi2sig', 'chi3', 'chi3sig', 'chi4', 'chi4sig'], axis=1, inplace=True)

    return rotlib

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

def num_chis_for_residue_id(res_id):
    if res_id in AAs_up_to_chi4():
        return 4
    if res_id in AAs_up_to_chi3():
        return 3
    elif res_id in AAs_up_to_chi2():
        return 2
    elif res_id in AAs_up_to_chi1():
        return 1
    else:
        return 0



def rama_plot_old(df, x, y, color_column, size_column, filename):
    '''
    plots phi and psi angles
    '''
    # Normalize the values in the color & size column
    norm_color = Normalize(vmin=0, vmax=data[color_column].max())
    norm_size = Normalize(vmin=0, vmax=data[size_column].max())
    # Create a colormap with white to blue gradient
    cmap = plt.get_cmap("Blues")
    # Map the normalized values to the colormap
    colors = cmap(norm_color(data[color_column]))
    # Set the size of dots according to the values in the size column
    data['norm_size'] = norm_size(data[size_column])
    size = data['norm_size'] * 3000
    plt.figure()
    plt.scatter(data[x], data[y], c=colors, s=size, marker=".")

    scalar_mappable = plt.cm.ScalarMappable(norm=norm_color, cmap=cmap)
    scalar_mappable.set_array(data[color_column])
    cb = plt.colorbar(mappable=scalar_mappable, label=color_column)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    plt.xticks(range(-180, 181, 60))
    plt.yticks(range(-180, 181, 60))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(f'{filename}.png', dpi=300)
    plt.show()
    return

def rama_plot(df, x_col, y_col, color_col, size_col, save_path=None):
    df_list = []
    for phi_psi, df in df.groupby([x_col, y_col]):
        top = df.sort_values(color_col, ascending=False).head(1)
        df_list.append(top)
    df = pd.concat(df_list)
    df = df[df[size_col] > 0]
    fig, ax = plt.subplots()
    norm_color = plt.Normalize(0, df[color_col].max())
    cmap = plt.cm.Blues
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_color)
    sm.set_array([])
    norm_size = plt.Normalize(0, df[size_col].max())
    ax.scatter(df[x_col], df[y_col], c=df[color_col], cmap=cmap, s=df[size_col], norm=norm_color)
    fig.colorbar(sm, label="probability", ax=ax)
    ax.set_xlabel("phi [degrees]")
    ax.set_ylabel("psi [degrees]")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-180, 181, 60))
    fig.gca().set_aspect('equal', adjustable='box')
    if save_path:
        plt.savefig(save_path, dpi=300)

def identify_backbone_angles_suitable_for_rotamer(residue_identity:str, rotlib:pd.DataFrame(), output_prefix:str=None, output_dir:str=None, limit_sec_struct:str=None, occurrence_cutoff:int=5, max_output:int=None, rotamer_diff_to_best=0.05, rotamer_chi_binsize=None, rotamer_phipsi_binsize=None, prob_cutoff=None):
    '''
    finds phi/psi angles most common for a given set of chi angles from a rotamer library
    chiX_bin multiplies the chiXsigma that is used to check if a rotamer fits to the given set of chi angles --> increasing leads to higher number of hits, decreasing leads to rotamers that more closely resemble input chis. Default=1
    if score_cutoff is provided, returns only phi/psi angles above score_cutoff
    if fraction is provided, returns only the top rotamer fraction ranked by score
    if max_output is provided, returns only max_output phi/psi combinations
    '''

    filename = utils.create_output_dir_change_filename(output_dir, output_prefix + f'_{residue_identity}_rama_pre_filtering')
    df_list = []

    rama_plot(rotlib, 'phi', 'psi', 'probability', 'phi_psi_occurrence', filename)


    if limit_sec_struct:
        rotlib = filter_rotamers_by_sec_struct(rotlib, limit_sec_struct)

    if occurrence_cutoff:
        rotlib = rotlib.loc[rotlib['phi_psi_occurrence'] > occurrence_cutoff / 100]
        if rotlib.empty:
            log_and_print(f"No rotamers passed occurrence cutoff of {occurrence_cutoff}")
            logging.error(f"No rotamers passed occurrence cutoff of {occurrence_cutoff}")

    if prob_cutoff:
        rotlib = rotlib[rotlib['probability'] >= prob_cutoff]
        if rotlib.empty:
            log_and_print(f"No rotamers passed probability cutoff of {prob_cutoff}")
            logging.error(f"No rotamers passed probability cutoff of {prob_cutoff}")

    rotlib = rotlib.sort_values('rotamer_score', ascending=False)

    if rotamer_chi_binsize and rotamer_phipsi_binsize:
        rotlib = filter_rotlib_for_rotamer_diversity(rotlib, rotamer_chi_binsize, rotamer_phipsi_binsize)

    if max_output:
        rotlib = rotlib.head(max_output)

    if rotamer_diff_to_best:
        rotlib = rotlib[rotlib['probability'] >= rotlib['probability'].max() * (1 - rotamer_diff_to_best)]

    if rotlib.empty:
        raise RuntimeError('Could not find any rotamers that fit. Try setting different filter values!')

    rotlib.reset_index(drop=True, inplace=True)

    filename = utils.create_output_dir_change_filename(output_dir, output_prefix + f'_{residue_identity}_rama_post_filtering')
    rama_plot(rotlib, 'phi', 'psi', 'probability', 'phi_psi_occurrence', filename)

    return rotlib

def angle_difference(angle1, angle2):

    return min([abs(angle1 - angle2), abs(angle1 - angle2 + 360), abs(angle1 - angle2 - 360)])

def filter_rotlib_for_rotamer_diversity(rotlib, rotamer_chi_binsize, rotamer_phipsi_binsize):
    accepted_rotamers = []
    chi_columns = [column for column in rotlib.columns if column.startswith('chi') and not column.endswith('sig')]

    for index, row in rotlib.iterrows():
        accept_list = []
        if len(accepted_rotamers) == 0:
            accepted_rotamers.append(row)
            continue
        for accepted_rot in accepted_rotamers:
            column_accept_list = []
            phipsi_difference = sum([angle_difference(row['phi'], accepted_rot['phi']), angle_difference(row['psi'], accepted_rot['psi'])])
            for column in chi_columns:
                #only accept rotamers that are different from already accepted ones
                if angle_difference(row[column], accepted_rot[column]) >= rotamer_chi_binsize / 2:
                    column_accept_list.append(True)
                else:
                    #still accept rotamers that are similar if their backbone angles are different enough
                    if phipsi_difference >= rotamer_phipsi_binsize:
                        column_accept_list.append(True)
                    #kick out everything were both is similar
                    else:
                        column_accept_list.append(False)
            if True in column_accept_list:
                accept_list.append(True)
            else:
                accept_list.append(False)
        if set(accept_list) == {True}:
            accepted_rotamers.append(row)
    rotlib = pd.DataFrame(accepted_rotamers)
    return rotlib


def filter_rotamers_by_sec_struct(rotlib:pd.DataFrame, secondary_structure:str):
    filtered_list = []
    sec_structs = [*secondary_structure]
    #phi and psi angle range was determined from fragment library
    if "-" in sec_structs:
        phi_range = [x for x in range(-170, -39, 10)] + [x for x in range(60, 81, 10)]
        psi_range = [x for x in range(-180, -159, 10)] + [x for x in range(-40, 181, 10)]
        filtered = rotlib.loc[(rotlib['phi'].isin(phi_range)) & (rotlib['psi'].isin(psi_range))]
        filtered_list.append(filtered)
    if "B" in sec_structs:
        phi_range = [x for x in range(-170, -49, 10)]
        psi_range = [x for x in range(-180, -169, 10)] + [x for x in range(80, 181, 10)]
        filtered = rotlib.loc[(rotlib['phi'].isin(phi_range)) & (rotlib['psi'].isin(psi_range))]
        filtered_list.append(filtered)
    if "E" in sec_structs:
        phi_range = [x for x in range(-170, -59, 10)]
        psi_range = [x for x in range(-180, -169, 10)] + [x for x in range(90, 181, 10)]
        filtered = rotlib.loc[(rotlib['phi'].isin(phi_range)) & (rotlib['psi'].isin(psi_range))]
        filtered_list.append(filtered)
    if "G" in sec_structs:
        phi_range = [x for x in range(-130, -39, 10)] + [x for x in range(50, 71, 10)]
        psi_range = [x for x in range(-50, 41, 10)]
        filtered = rotlib.loc[(rotlib['phi'].isin(phi_range)) & (rotlib['psi'].isin(psi_range))]
        filtered_list.append(copy.deepcopy(filtered))
    if "H" in sec_structs:
        phi_range = [x for x in range(-100, -39, 10)]
        psi_range = [x for x in range(-60, 1, 10)]
        filtered = rotlib.loc[(rotlib['phi'].isin(phi_range)) & (rotlib['psi'].isin(psi_range))]
        filtered_list.append(filtered)
    if "I" in sec_structs:
        phi_range = [x for x in range(-140, -49, 10)]
        psi_range = [x for x in range(-80, 1, 10)]
        filtered = rotlib.loc[(rotlib['phi'].isin(phi_range)) & (rotlib['psi'].isin(psi_range))]
        filtered_list.append(filtered)
    if "S" in sec_structs:
        phi_range = [x for x in range(-170, -49, 10)] + [x for x in range(50, 111, 10)]
        psi_range = [x for x in range(-180, -149, 10)] + [x for x in range(-60, 181, 10)]
        filtered = rotlib.loc[(rotlib['phi'].isin(phi_range)) & (rotlib['psi'].isin(psi_range))]
        filtered_list.append(filtered)
    if "T" in sec_structs:
        phi_range = [x for x in range(-130, -40, 10)] + [x for x in range(40, 111, 10)]
        psi_range = [x for x in range(-60, 61, 10)] + [x for x in range(120, 151, 10)]
        filtered = rotlib.loc[(rotlib['phi'].isin(phi_range)) & (rotlib['psi'].isin(psi_range))]
        filtered_list.append(filtered)
    rotlib = pd.concat(filtered_list)
    if rotlib.empty:
        log_and_print(f"No rotamers passed secondary structure filtering for secondary structure {secondary_structure}.")
        logging.error(f"No rotamers passed secondary structure filtering for secondary structure {secondary_structure}.")
    return rotlib

def import_fragment_library(library_path:str):
    '''
    reads in a fragment library
    '''
    library = pd.read_pickle(library_path)
    #library.drop(library.columns[[0]], axis=1, inplace=True)
    return library

def is_unique(df_column):
    '''
    determines if all values in column are the same. quicker than nunique according to some guy on stackoverflow
    '''
    a = df_column.to_numpy()
    return (a[0] == a).all()

def check_for_chainbreaks(df, columname, fragsize):
    '''
    returns true if dataframe column is consistently numbered
    '''
    if df[columname].diff().sum() + 1 == fragsize:
        return True
    else:
        return False

def filter_frags_df_by_secondary_structure_content(frags_df, frag_sec_struct_fraction):

    frags_df_list = []
    for frag_num, df in frags_df.groupby('frag_num', sort=False):
        for sec_struct in frag_sec_struct_fraction:
            if df['ss'].str.contains(sec_struct).sum() / len(df.index) >= frag_sec_struct_fraction[sec_struct]:
                frags_df_list.append(df)
                break
    if len(frags_df_list) > 0:
        frags_df = pd.concat(frags_df_list)
        return frags_df
    else:
        return pd.DataFrame()

def filter_frags_df_by_score(frags_df, score_cutoff, scoretype, mode):

    frags_df_list = []
    for frag_num, df in frags_df.groupby('frag_num', sort=False):
        #only accepts fragments where mean value is above threshold
        if mode == 'mean_max_cutoff':
            if df[scoretype].mean() < score_cutoff:
                frags_df_list.append(df)
        if mode == 'mean_min_cutoff':
            if df[scoretype].mean() > score_cutoff:
                frags_df_list.append(df)
        #only accepts fragments if none of the residues is above threshold
        elif mode == 'max_cutoff':
            if df[scoretype].max() < score_cutoff:
                frags_df_list.append(df)
        elif mode == 'min_cutoff':
            if df[scoretype].min() > score_cutoff:
                frags_df_list.append(df)

    if len(frags_df_list) > 0:
        frags_df = pd.concat(frags_df_list)
        return frags_df
    else:
        return pd.DataFrame()

def add_frag_to_structure(frag, structure):
    frag_num = len([model for model in structure.get_models()])
    model = Model.Model(frag_num)
    model.add(frag)
    structure.add(model)

def check_fragment(frag, frag_list, frag_df, df_list, ligand, channel, vdw_radii, rotamer_position, covalent_bond, rmsd_cutoff, backbone_ligand_clash_detection_vdw_multiplier, rotamer_ligand_clash_detection_vdw_multiplier, channel_fragment_clash_detection_vdw_multiplier, num_channel_clash, num_bb_clash, num_sc_clash, num_rmsd_fails):
    frag_df['frag_num'] = len(frag_list)
    clash_check = False
    if channel:
        clash_check = distance_detection(frag, channel, vdw_radii, True, False, channel_fragment_clash_detection_vdw_multiplier, rotamer_position, None)
        if clash_check == True:
            num_channel_clash += 1
    if ligand and clash_check == False:
        #check for backbone clashes
        clash_check = distance_detection(frag, ligand, vdw_radii, True, True, backbone_ligand_clash_detection_vdw_multiplier, rotamer_position, covalent_bond)
        if clash_check == True:
            num_bb_clash += 1
        if clash_check == False:
            #check for rotamer clashes
            clash_check = distance_detection(frag[rotamer_position], ligand, vdw_radii, False, True, rotamer_ligand_clash_detection_vdw_multiplier, rotamer_position, covalent_bond, True)
            if clash_check == True:
                num_sc_clash += 1
    #add the first encountered fragment without rmsd checking
    if clash_check == False and len(frag_list) == 0:
        frag_list.append(frag)
        df_list.append(frag_df)
        return frag_list, df_list, num_channel_clash, num_bb_clash, num_sc_clash, num_rmsd_fails
    #calculate rmsds for all already accepted fragments
    if clash_check == False and len(frag_list) > 0:
        rmsdlist = [calculate_rmsd_bb(picked_frag, frag) for picked_frag in frag_list]
        #if the lowest rmsd compared to all other fragments is higher than the set cutoff, add it to the filtered dataframe
        if min(rmsdlist) >= rmsd_cutoff:
            frag_list.append(frag)
            df_list.append(frag_df)
        else:
            num_rmsd_fails += 1

    return frag_list, df_list, num_channel_clash, num_bb_clash, num_sc_clash, num_rmsd_fails

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

def distance_detection(entity1, entity2, vdw_radii:dict, bb_only:bool=True, ligand:bool=False, clash_detection_vdw_multiplier:float=1.0, resnum:int=None, covalent_bond:str=None, ignore_func_groups:bool=True):
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
        #exclude backbone atoms because they have been checked in previous step
        entity1_atoms = (atom for atom in entity1.get_atoms() if not atom.name in backbone_atoms)
        entity2_atoms = (atom for atom in entity2.get_atoms())

    for atom_combination in itertools.product(entity1_atoms, entity2_atoms):
        #skip clash detection for covalent bonds
        covalent = False
        if ignore_func_groups == True and atom_combination[0].name in atoms_of_functional_groups():
            covalent = True
        if resnum and covalent_bond:
            for cov_bond in covalent_bond.split(','):
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

def atoms_of_functional_groups():
    return ["NH1", "NH2", "OD1", "OD2", "ND2", "NE", "SG", "OE1", "OE2", "NE2", "ND1", "NZ", "SD", "OG", "OG1", "NE1", "OH"]


def sort_frags_df_by_score(fragment_df, scoretype):
    score_df_list = []
    init = time.time()
    fragment_df['phi_rounded'] = fragment_df['phi'].round(-1)
    fragment_df['psi_rounded'] = fragment_df['psi'].round(-1)
    for fragment, df in fragment_df.groupby('frag_num', sort=False):
        #rotamer score has to be in here because otherwise fragments with same phi/psi, but different chis would get filtered
        df['phi_psi_prob'] = [df['phi_rounded'].to_list() + df['psi_rounded'].to_list() + df['rotamer_score'].to_list() + df['rotamer_pos'].to_list() for i in range(0, len(df.index))]
        df['phi_psi_prob'] = df['phi_psi_prob'].astype(str)
        score_df_list.append((df[scoretype].mean(), df))

    #sort fragments by average score
    if len(score_df_list) > 1:
        score_df_list.sort(key=lambda x: x[0], reverse=True)

    #assemble dataframe
    score_sorted_list = []
    count = 0
    for score_df in score_df_list:
        df = score_df[1]
        score_sorted_list.append(df)
        count += 1
    fragment_df = pd.concat(score_sorted_list)

    #remove all fragments with same phi/psi/chi angles
    filtered = fragment_df.drop_duplicates('phi_psi_prob', keep="first")[['frag_num', 'phi_psi_prob']]
    log_and_print(f"Removed {count - len(filtered.index)} fragments with similar backbone angles.")
    fragment_df = fragment_df.drop('phi_psi_prob', axis=1).merge(filtered, how='inner', on='frag_num').reset_index(drop=True).drop(['phi_psi_prob', 'phi_rounded', 'psi_rounded'], axis=1)


    return fragment_df

def calculate_rmsd_bb(entity1, entity2):
    '''
    calculates rmsd of 2 structures considering CA atoms. does no superposition!
    '''
    sup = Superimposer()
    bb_atoms = ["N"]
    entity1_atoms = [atom for atom in entity1.get_atoms() if atom.id in bb_atoms]
    entity2_atoms = [atom for atom in entity2.get_atoms() if atom.id in bb_atoms]

    rmsd = math.sqrt(sum([(atom1 - atom2) ** 2 for atom1, atom2 in zip(entity1_atoms, entity2_atoms)]) / len(entity1_atoms))

    return rmsd

def create_fragment_from_df(df:pd.DataFrame()):
    '''
    creates a biopython chain from dataframe containing angles and coordinates
    '''
    chain = Chain.Chain('A')

    serial_num = 1
    resnum = 1
    for index, row in df.iterrows():
        res = Residue.Residue((' ', resnum, ' '), row['AA'], ' ')
        resnum += 1
        for atom in ["N", "CA", "C", "O"]:
            coords = np.array([row[f'{atom}_x'], row[f'{atom}_y'], row[f'{atom}_z']])
            bfactor = 0 if math.isnan(row['probability']) else round(row['probability'] * 100, 2)
            bb_atom = Atom.Atom(name=atom, coord=coords, bfactor=bfactor, occupancy=1.0, altloc=' ', fullname=f' {atom} ', serial_number=serial_num, element=atom[0])
            serial_num += 1
            res.add(bb_atom)
        chain.add(res)

    return chain

def check_if_angle_in_bin(df, phi, psi, phi_psi_bin):

    df['phi_difference'] = df.apply(lambda row: angle_difference(row['phi'], phi), axis=1)
    df['psi_difference'] = df.apply(lambda row: angle_difference(row['psi'], psi), axis=1)

    df = df[(df['psi_difference'] < phi_psi_bin / 2) & (df['phi_difference'] < phi_psi_bin / 2)]
    df = df.drop(['phi_difference', 'psi_difference'], axis=1)
    return df


def identify_positions_for_rotamer_insertion(fraglib, rotlib, rot_sec_struct, limit_frags_to_chi, limit_frags_to_res_id, phi_psi_bin) -> pd.DataFrame:

    #convert string to list
    if rot_sec_struct:
        rot_sec_struct = [*rot_sec_struct]

    #add 360 to all negative angles
    angles = ['chi1', 'chi2', 'chi3', 'chi4']
    columns = list(rotlib.columns)
    rotamer_positions_list = []
    for index, row in rotlib.iterrows():
        #filter based on difference & amino acid identity
        rotamer_positions = check_if_angle_in_bin(fraglib, row['phi'], row['psi'], phi_psi_bin)
        if rotamer_positions.empty:
            continue
        if limit_frags_to_res_id == True:
            rotamer_positions = rotamer_positions[rotamer_positions['AA'] == row['identity']]
        if rotamer_positions.empty:
            continue
        if rot_sec_struct:
            rotamer_positions = rotamer_positions[rotamer_positions['ss'].isin(rot_sec_struct)]
        if rotamer_positions.empty:
            continue
        
        if num_chis_for_residue_id(row['identity']) > 0:
            for chi_angle in range(1, num_chis_for_residue_id(row['identity']) + 1):
                chi = f"chi{chi_angle}"
                if limit_frags_to_chi == True:
                    #filter for positions that have chi angles within 2 stdev from input chis
                    rotamer_positions[f'{chi}_difference'] = rotamer_positions.apply(lambda line: angle_difference(line[chi], row[chi]), axis=1)
                    rotamer_positions = rotamer_positions[rotamer_positions[f'{chi}_difference'] < row[f'{chi}sig']]
                    rotamer_positions.drop(f'{chi}_difference', axis=1, inplace=True)
                #change chi angles to mean value from bin
                rotamer_positions[chi] = row[chi]
        rotamer_positions['probability'] = row['probability']
        rotamer_positions['phi_psi_occurrence'] = row['phi_psi_occurrence']
        rotamer_positions['rotamer_score'] = row['rotamer_score']
        rotamer_positions['log_occurrence'] = row['log_occurrence']
        log_and_print(f"Found {len(rotamer_positions.index)} positions for {row['identity']} rotamer {index}")
        rotamer_positions_list.append(rotamer_positions)

    if not rotamer_positions_list:
        raise RuntimeError('Could not find any fragment positions that fit criteria. Try adjusting rotamer secondary structure or use different rotamers!')
    elif len(rotamer_positions_list) > 1:
        rotamer_positions = pd.concat(rotamer_positions_list)
    else:
        rotamer_positions = rotamer_positions_list[0]
    
    return rotamer_positions

def normalize_col(df:pd.DataFrame, col:str, scale:bool=False) -> pd.DataFrame:
    ''''''
    median = df[col].median()
    std = df[col].std()
    if std == 0 or len(set(df[col].to_list())) == 1:
        df[f"{col}_normalized"] = 0
        return df
    df[f"{col}_normalized"] = (df[col] - median) / std
    if scale == True:
        #scale everything so that values range from 0 to 1
        factor = df[f"{col}_normalized"].max() - df[f"{col}_normalized"].min()
        df[f"{col}_normalized"] = df[f"{col}_normalized"] / factor
        df[f"{col}_normalized"] = df[f"{col}_normalized"] + (1 - df[f"{col}_normalized"].max())
        df[f"{col}_normalized"]
    return df

def extract_fragments(rotamer_positions_df, fraglib, frag_pos_to_replace, fragsize):
    '''
    finds fragments based on phi/psi combination and residue identity
    frag_pos_to_replace: the position in the fragment the future rotamer should be inserted. central position recommended.
    residue_identity: only accept fragments with the correct residue identity at that position (recommended)
    rotamer_secondary_structure: accepts a string describing secondary structure (B: isolated beta bridge residue, E: strand, G: 3-10 helix, H: alpha helix, I: pi helix, T: turn, S: bend, -: none (not in the sense of no filter --> use None instead!)). e.g. provide EH if central atom in fragment should be a helix or strand.
    bfactor_cutoff: only return fragments with bfactor below this threshold
    rmsd_cutoff: compare fragments by fragments, only accept fragments that have rmsd above this threshold.
    '''

    #choose fragments from fragment library that contain the positions selected above
    fragnum = 0
    frag_dict = {}
    for pos in frag_pos_to_replace:
        frag_dict[pos] = []
    for index, row in rotamer_positions_df.iterrows():
        for pos in frag_pos_to_replace:
            upper = index + fragsize - pos
            lower = index - pos + 1
            df = fraglib.loc[lower:upper, fraglib.columns]
            #only choose fragments with correct length etc
            if is_unique(df['pdb']) and len(df) == fragsize and check_for_chainbreaks(df, 'position', fragsize) == True:
                fragnum = fragnum + 1
                df.loc[:, 'frag_num'] = fragnum
                df.loc[:, 'rotamer_pos'] = int(pos)
                df.at[index, 'probability'] = row['probability']
                df.at[index, 'phi_psi_occurrence'] = row['phi_psi_occurrence'] * 100
                #set occurrence score to log_occurrence of the rotamer identity at rotamer position since the AA identity is fixed
                df.at[index, 'occurrence_score'] = row['log_occurrence']
                df['rotamer_score'] = row['rotamer_score']
                frag_dict[pos].append(df)

    for pos in frag_dict:
        if len(frag_dict[pos]) > 0:
            frag_dict[pos] = pd.concat(frag_dict[pos])
        else:
            frag_dict[pos] = pd.DataFrame()
        frag_dict[pos].to_csv(f'test_{pos}.csv')

    return frag_dict

def is_unique(s):
    '''
    determines if all values in column are the same. quicker than nunique according to some guy on stackoverflow
    '''
    a = s.to_numpy()
    return (a[0] == a).all()

def attach_rotamer_to_fragments(df, frag, AA_alphabet):
    rotamer_on_fragments = Structure.Structure("rotonfrags")

    rotamer = identify_rotamer_position_by_probability(df)
    columns = ['chi1', 'chi2', 'chi3', 'chi4']
    chi_angles = [None if math.isnan(rotamer[chi]) else rotamer[chi] for chi in columns]
    rot_pos = int(rotamer['rotamer_pos'])
    to_mutate = frag[rot_pos]
    resid = to_mutate.id
    backbone_angles = extract_backbone_angles(frag, rot_pos)
    backbone_bondlengths = extract_backbone_bondlengths(frag, rot_pos)
    res = generate_rotamer(AA_alphabet, rotamer['AA'], resid, backbone_angles["phi"], backbone_angles["psi"], backbone_angles["omega"], backbone_angles["carb_angle"], backbone_angles["tau"], backbone_bondlengths["N_CA"], backbone_bondlengths["CA_C"], backbone_bondlengths["C_O"], chi_angles[0], chi_angles[1], chi_angles[2], chi_angles[3], rotamer['probability'])
    delattr(res, 'internal_coord')
    rotamer_on_fragments = attach_rotamer_to_backbone(frag, to_mutate, res)

    return rotamer_on_fragments

def attach_rotamer_to_backbone(fragment, fragment_residue, rotamer):
    fragment.detach_child(fragment_residue.id)
    to_mutate_atoms = []
    res_atoms = []
    for atom in ["N", "CA", "C"]:
        to_mutate_atoms.append(fragment_residue[atom])
        res_atoms.append(rotamer[atom])
    sup = Bio.PDB.Superimposer()
    sup.set_atoms(to_mutate_atoms, res_atoms)
    sup.rotran
    sup.apply(rotamer)

    fragment.insert(rotamer.id[1]-1, rotamer)

    return fragment


def extract_backbone_angles(chain, resnum:int):
    '''
    takes a biopython chain and extracts phi/psi/omega angles of specified residue
    '''
    #convert to internal coordinates, read phi/psi angles
    chain = copy.deepcopy(chain)
    chain.atom_to_internal_coordinates()
    phi = chain[resnum].internal_coord.get_angle("phi")
    psi = chain[resnum].internal_coord.get_angle("psi")
    omega = chain[resnum].internal_coord.get_angle("omg")
    carb_angle = round(chain[resnum].internal_coord.get_angle("N:CA:C:O"), 1)
    tau = round(chain[resnum].internal_coord.get_angle("tau"), 1)
    if not phi == None:
        phi = round(phi, 1)
    if not psi == None:
        psi = round(psi, 1)
    if not omega == None:
        omega = round(omega, 1)
    return {"phi": phi, "psi": psi, "omega": omega, "carb_angle": carb_angle, "tau": tau}

def extract_backbone_bondlengths(chain, resnum:int):
    '''
    takes a biopython chain and extracts phi/psi/omega angles of specified residue
    '''
    #convert to internal coordinates, read phi/psi angles
    chain = copy.deepcopy(chain)
    chain.atom_to_internal_coordinates()
    N_CA = round(chain[resnum].internal_coord.get_length("N:CA"), 3)
    CA_C = round(chain[resnum].internal_coord.get_length("CA:C"), 3)
    C_O = round(chain[resnum].internal_coord.get_length("C:O"), 3)
    return {"N_CA": N_CA, "CA_C": CA_C, "C_O": C_O}



def generate_rotamer(AAalphabet_structure, residue_identity:str, res_id, phi:float=None, psi:float=None, omega:float=None, carb_angle:float=None, tau:float=None, N_CA_length:float=None, CA_C_length:float=None, C_O_length:float=None, chi1:float=None, chi2:float=None, chi3:float=None, chi4:float=None, rot_probability:float=None):
    '''
    builds a rotamer from residue identity, phi/psi/omega/chi angles
    '''
    alphabet = copy.deepcopy(AAalphabet_structure)
    for res in alphabet[0]["A"]:
        if res.get_resname() == residue_identity:
            #set internal coordinates
            alphabet[0]["A"].atom_to_internal_coordinates()
            #change angles to specified value
            if tau:
                res.internal_coord.set_angle("tau", tau)
            if carb_angle:
                res.internal_coord.bond_set("N:CA:C:O", carb_angle)
            if phi:
                res.internal_coord.set_angle("phi", phi)
            if psi:
                res.internal_coord.set_angle("psi", psi)
            if omega:
                res.internal_coord.set_angle("omega", omega)
            if N_CA_length:
                res.internal_coord.set_length("N:CA", N_CA_length)
            if CA_C_length:
                res.internal_coord.set_length("CA:C", CA_C_length)
            if C_O_length:
                res.internal_coord.set_length("C:O", C_O_length)

            max_chis = num_chis_for_residue_id(residue_identity)

            if max_chis > 0:
                res.internal_coord.bond_set("chi1", chi1)
            if max_chis > 1:
                res.internal_coord.bond_set("chi2", chi2)
            if max_chis > 2:
                res.internal_coord.bond_set("chi3", chi3)
            if max_chis > 3:
                res.internal_coord.set_angle("chi4", chi4)
            alphabet[0]["A"].internal_to_atom_coordinates()
            #change residue number to the one that is replaced (detaching is necessary because otherwise 2 res with same resid would exist in alphabet)
            alphabet[0]["A"].detach_child(res.id)
            res.id = res_id
            if rot_probability:
                for atom in res.get_atoms():
                    atom.bfactor = rot_probability * 100

            return res

def identify_rotamer_position_by_probability(df):
    #drop all rows that do not contain the rotamer, return a series --> might cause issues downstream if more than one rotamer on a fragment
    rotamer_pos = df.dropna(subset = ['probability']).squeeze()
    return(rotamer_pos)

def align_to_sidechain(entity, entity_residue_to_align, sidechain, flip_symmetric:bool=True):
    '''
    aligns an input structure (bb_fragment_structure, resnum_to_align) to a sidechain residue (sc_structure, resnum_to_alignto)
    '''
    sc_residue_identity = sidechain.get_resname()

    #superimpose structures based on specified atoms
    bbf_atoms = atoms_for_func_group_alignment(entity_residue_to_align)
    sc_atoms = atoms_for_func_group_alignment(sidechain)
    if flip_symmetric == True and sc_residue_identity in tip_symmetric_residues():
        order = [1, 0, 2]
        sc_atoms = [sc_atoms[i] for i in order]
    sup = Bio.PDB.Superimposer()
    sup.set_atoms(sc_atoms, bbf_atoms)
    sup.rotran
    sup.apply(entity)

    return entity

def identify_his_central_atom(histidine, ligand):
    HIS_NE2 = histidine["NE2"]
    HIS_ND1 = histidine["ND1"]
    lig_atoms =  [atom for atom in ligand.get_atoms()]
    NE2_distance = min([HIS_NE2 - atom for atom in lig_atoms])
    ND1_distance = min([HIS_ND1 - atom for atom in lig_atoms])
    if NE2_distance < ND1_distance:
        his_central_atom = "NE2"
    else:
        his_central_atom = "ND1"
    return his_central_atom

def rotate_histidine_fragment(entity, degrees, theozyme_residue, his_central_atom, ligand):
    if his_central_atom == "auto":
        if not ligand:
            raise RuntimeError(f"Ligand is required if using <flip_histidines='auto'>!")
        his_central_atom = identify_his_central_atom(theozyme_residue, ligand)
    if his_central_atom == "NE2":
        half = theozyme_residue["ND1"].coord + 0.5 * (theozyme_residue["CG"].coord - theozyme_residue["ND1"].coord)
    elif his_central_atom == "ND1":
        half = theozyme_residue["NE2"].coord + 0.5 * (theozyme_residue["CD2"].coord - theozyme_residue["NE2"].coord)
    entity = rotate_entity_around_axis(entity, theozyme_residue[his_central_atom].coord, half, degrees)
    
    return entity

def rotate_phenylalanine_fragment(entity, degrees, theozyme_residue):

    center = theozyme_residue["CZ"].coord - 0.5 * (theozyme_residue["CZ"].coord - theozyme_residue["CG"].coord)
    vector_A = theozyme_residue["CG"].coord - theozyme_residue["CZ"].coord
    vector_B = theozyme_residue["CE1"].coord - theozyme_residue["CZ"].coord
    N = np.cross(vector_A, vector_B)
    N = N / np.linalg.norm(N)
    point = center + N

    entity = rotate_entity_around_axis(entity, center, point, degrees)
    
    return entity

def rotate_entity_around_axis(entity, coords_1, coords_2, angle):
    rotation_axis = coords_1 - coords_2
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    angle_radians = np.radians(angle)
    for atom in entity.get_atoms():
        array = atom.coord - coords_1
        rotated_array = rotate_array_around_vector(array, rotation_axis, angle_radians)
        atom.coord = rotated_array + coords_1
    return entity

def rotation_matrix(V, X):
    K = np.array([[0, -V[2], V[1]],
                  [V[2], 0, -V[0]],
                  [-V[1], V[0], 0]])
    I = np.identity(3)
    R = I + np.sin(X)*K + (1-np.cos(X))*np.dot(K,K)
    return R

def rotate_array_around_vector(array, axis, angle):
    """
    Rotate a 3D NumPy array around a specified axis by a given angle.

    Parameters:
        array: A 3D NumPy array to be rotated.
        axis: A 3D NumPy array representing the rotation axis.
        angle: The angle (in radians) of rotation.

    Returns:
        The rotated 3D NumPy array.
    """
    rotation_matrix_3x3 = rotation_matrix(axis, angle)
    rotated_array = np.dot(rotation_matrix_3x3, array.T).T
    return rotated_array

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

def atoms_for_func_group_alignment(residue):
    '''
    return the atoms used for superposition via functional groups
    '''
    sc_residue_identity = residue.get_resname()

    if not sc_residue_identity in func_groups():
        raise RuntimeError(f'Unknown residue with name {sc_residue_identity}!')
    else:
        return [residue[atom] for atom in func_groups()[sc_residue_identity]]
    
def func_groups():
    func_groups = {
        "ALA": ["CB", "CA", "N"],
        "ARG": ["NH1", "NH2", "CZ"],
        "ASP": ["OD1", "OD2", "CG"],
        "ASN": ["OD1", "ND2", "CG"],
        "CYS": ["SG", "CB", "CA"],
        "GLU": ["OE1", "OE2", "CD"],
        "GLN": ["OE1", "NE2", "CD"],
        "GLY": ["CA", "N", "C"],
        "HIS": ["ND1", "NE2", "CG"],
        "ILE": ["CD1", "CG1", "CB"],
        "LEU": ["CD1", "CD2", "CG"],
        "LYS": ["NZ", "CE", "CD"],
        "MET": ["CE", "SD", "CG"],
        "PHE": ["CD1", "CD2", "CZ"],
        "PRO": ["CD", "CG", "CB"],
        "SER": ["OG", "CB", "CA"],
        "THR": ["OG1", "CG2", "CB"],
        "TRP": ["NE1", "CZ3", "CG"],
        "TYR": ["CE1", "CE2", "OH"],
        "VAL": ["CG1", "CG2", "CB"]
        }
    return func_groups


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def clean_input_backbone(entity):
    for model in entity.get_models():
        model.id = 0
    for chain in entity.get_chains():
        chain.id = "A"
    for index, residue in enumerate(entity.get_residues()):
        residue.id = (residue.id[0], index + 1, residue.id[2])
    for atom in entity.get_atoms():
        atom.bfactor = 0
    return entity[0]["A"]

def identify_residues_with_equivalent_func_groups(residue):
    '''
    checks if residues with same functional groups exist, returns a list of these residues
    '''
    resname = residue.get_resname()
    if resname in ['ASP', 'GLU']:
        return ['ASP', 'GLU']
    elif resname in ['ASN', 'GLN']:
        return ['ASN', 'GLN']
    elif resname in ['VAL', 'ILE']:
        return ['VAL', 'LEU']
    else:
        return [resname]
    
def rotamers_for_backbone(resnames, rotlib_path, phi, psi, rot_prob_cutoff:float=0.05, max_rotamers:int=70, max_stdev:float=2, level:int=2):
    rotlib_list = []
    for res in resnames:
        if res in ["ALA", "GLY"]:
            #TODO: assign proper scores for log prob and occurrence
            rotlib = pd.DataFrame([[res, phi, psi, float("nan"), 1, float("nan"), 0, 0]], columns=["AA", "phi", "psi", "count", "probability", "phi_psi_occurrence", "log_prob", "log_occurrence"])
            rotlib_list.append(rotlib)
        else:
            rotlib = return_residue_rotamer_library(rotlib_path, res)
            rotlib_list.append(identify_rotamers_suitable_for_backbone(res, phi, psi, rotlib, rot_prob_cutoff, max_rotamers, max_stdev, level))
    if len(rotlib_list) > 1:
        filtered_rotlib = pd.concat(rotlib_list)
        filtered_rotlib = filtered_rotlib.sort_values("probability", ascending=False)
        filtered_rotlib.reset_index(drop=True, inplace=True)
        return filtered_rotlib
    else:
        return rotlib_list[0]
    
def identify_rotamers_suitable_for_backbone(residue_identity:str, phi:float, psi:float, rotlib:pd.DataFrame(), prob_cutoff:float=None, max_rotamers:int=None, max_stdev:float=2, level:int=3):
    '''
    identifies suitable rotamers by filtering for phi/psi angles
    if fraction is given, returns only the top rotamer fraction ranked by probability (otherwise returns all rotamers)
    if prob_cutoff is given, returns only rotamers more common than prob_cutoff
    '''
    rotlib.rename(columns={'identity': 'AA'}, inplace=True)
    #round dihedrals to the next tens place
    if not phi == None:
        phi = round(phi, -1)
    if not psi == None:
        psi = round(psi, -1)
    #extract all rows containing specified phi/psi angles from library
    if phi and psi:
        log_and_print(f"Searching for rotamers in phi/psi bin {phi}/{psi}.")
        rotlib = rotlib.loc[(rotlib['phi'] == phi) & (rotlib['psi'] == psi)].reset_index(drop=True)
    elif not phi or not psi:
        if not phi:
            rotlib = rotlib[rotlib['psi'] == psi].reset_index(drop=True)
        elif not psi:
            rotlib = rotlib[rotlib['phi'] == phi].reset_index(drop=True)
        rotlib = rotlib.loc[rotlib['phi_psi_occurrence'] >= 1]
        rotlib = rotlib.drop_duplicates(subset=['phi', 'psi'], keep='first')
        rotlib.sort_values("probability", ascending=False)
        rotlib = rotlib.head(5)
    #filter top rotamers
    rotlib = rotlib.sort_values("probability", ascending=False)
    if prob_cutoff:
        rotlib = rotlib.loc[rotlib['probability'] > prob_cutoff]
    if level > 0:
        rotlib = diversify_chi_angles(rotlib, max_stdev, level)
        #filter again, since diversify_chi_angles produces rotamers with lower probability
        if prob_cutoff:
            rotlib = rotlib.loc[rotlib['probability'] > prob_cutoff]
    if max_rotamers:
        rotlib = rotlib.head(max_rotamers)
    return rotlib


def diversify_chi_angles(rotlib, max_stdev:float=2, level:int=3):
    '''
    adds additional chi angles based on standard deviation.
    max_stdev: defines how far to stray from mean based on stdev. chi_new = chi_orig +- stdev * max_stdev
    level: defines how many chis should be sampled within max_stdev. if level = 1, mean, mean + stdev*max_stdev, mean - stdev*max_stdev will be returned. if level = 2, mean, mean + 1/2 stdev*max_stdev, mean + stdev*max_stdev, mean - 1/2 stdev*max_stdev, mean - stdev*max_stdev will be returned
    '''
    #check which chi angles exist in rotamer library
    columns = list(rotlib.columns)
    columns = [column for column in columns if column.startswith('chi') and not 'sig' in column]
    #generate deviation parameters
    devs = [max_stdev * i / level for i in range(-level, level +1)]
    #calculate chi angles
    for chi_angle in columns:
        new_chis_list = []
        for dev in devs:
            new_chis = alter_chi(rotlib, chi_angle, f'{chi_angle}sig', dev)
            new_chis_list.append(new_chis)
        rotlib = pd.concat(new_chis_list)
        rotlib.drop([f'{chi_angle}sig'], axis=1, inplace=True)
        rotlib[chi_angle] = round(rotlib[chi_angle], 1)
    rotlib.sort_values('probability', inplace=True, ascending=False)
    rotlib.reset_index(drop=True, inplace=True)
    return rotlib

def create_df_from_fragment(backbone):

    pdbnames = ["frag" for res in backbone.get_residues()]
    resnames = [res.resname for res in backbone.get_residues()]
    pos_list = [res.id[1] for res in backbone.get_residues()]
    problist = [float("nan") for res in backbone.get_residues()]
    CA_x_coords_list = [(round(res["CA"].get_coord()[0], 3)) for res in backbone.get_residues()]
    CA_y_coords_list = [(round(res["CA"].get_coord()[1], 3)) for res in backbone.get_residues()]
    CA_z_coords_list = [(round(res["CA"].get_coord()[2], 3)) for res in backbone.get_residues()]
    C_x_coords_list = [(round(res["C"].get_coord()[0], 3)) for res in backbone.get_residues()]
    C_y_coords_list = [(round(res["C"].get_coord()[1], 3)) for res in backbone.get_residues()]
    C_z_coords_list = [(round(res["C"].get_coord()[2], 3)) for res in backbone.get_residues()]
    N_x_coords_list = [(round(res["N"].get_coord()[0], 3)) for res in backbone.get_residues()]
    N_y_coords_list = [(round(res["N"].get_coord()[1], 3)) for res in backbone.get_residues()]
    N_z_coords_list = [(round(res["N"].get_coord()[2], 3)) for res in backbone.get_residues()]
    O_x_coords_list = [(round(res["O"].get_coord()[0], 3)) for res in backbone.get_residues()]
    O_y_coords_list = [(round(res["O"].get_coord()[1], 3)) for res in backbone.get_residues()]
    O_z_coords_list = [(round(res["O"].get_coord()[2], 3)) for res in backbone.get_residues()]

    df = pd.DataFrame(list(zip(pdbnames, resnames, pos_list, CA_x_coords_list, CA_y_coords_list, CA_z_coords_list, C_x_coords_list, C_y_coords_list, C_z_coords_list, N_x_coords_list, N_y_coords_list, N_z_coords_list, O_x_coords_list, O_y_coords_list, O_z_coords_list, problist)), columns=["pdb", "AA", "position", "CA_x", "CA_y", "CA_z", "C_x", "C_y", "C_z", "N_x", "N_y", "N_z", "O_x", "O_y", "O_z", "probability"])
    df[["chi1", "chi2", "chi3", "chi4"]] = float("nan")
    return df

def normal_dist_density(x):
    '''
    calculates y value for normal distribution from distance from mean TODO: check if it actually makes sense to do it this way
    '''
    y = math.e **(-(x)**2 / 2)
    return y


def alter_chi(rotlib, chi_column, chi_sig_column, dev):
    '''
    calculate deviations from input chi angle for rotamer library
    '''
    new_chis = copy.deepcopy(rotlib)
    new_chis[chi_column] = new_chis[chi_column] + new_chis[chi_sig_column] * dev
    new_chis['probability'] = new_chis['probability'] * normal_dist_density(dev)
    new_chis['log_prob'] = np.log(new_chis['probability'])
    return new_chis

def exchange_covalent(covalent_bond):
    atom = covalent_bond.split(":")[0]
    exchange_dict = {"OE1": "OD1", "OE2": "OD2", "CD1": "CG1", "CD2": "CG2", "NE2": "ND2", "OD1": "OE1", "OD2": "OE2", "CG1": "CD1", "CG2": "CD2", "ND2": "NE2"}
    try:
        atom = exchange_dict[atom]
    except:
        atom = atom
    return atom + ":" + covalent_bond.split(":")[1]

def flip_covalent(covalent_bond, residue):
    atom = covalent_bond.split(":")[0]
    exchange_dict = {
        "GLU": {"OE1": "OE2", "OE2": "OE1"},
        "ASP": {"OD1": "OD2", "OD2": "OD1"},
        "VAL": {"CD1": "CD2", "CD2": "CD1"},
        "LEU": {"CG1": "CG2", "CG2": "CG1"},
        "ARG": {"NH1": "NH2", "NH2": "NH1"}     
        }
    try:
        atom = exchange_dict[residue][atom]
    except:
        atom = atom
    return atom + ":" + covalent_bond.split(":")[1]

def log_and_print(string: str): 
    logging.info(string)
    print(string)
    return string

def combine_normalized_scores(df: pd.DataFrame, name:str, scoreterms:list, weights:list, normalize:bool=False, scale:bool=False):
    if not len(scoreterms) == len(weights):
        raise RuntimeError(f"Number of scoreterms ({len(scoreterms)}) and weights ({len(weights)}) must be equal!")
    df[name] = sum([df[col]*weight for col, weight in zip(scoreterms, weights)]) / sum(weights)
    if normalize == True:
        df = normalize_col(df, name, scale)
        df.drop(name, axis=1, inplace=True)
        df.rename(columns={f'{name}_normalized': name}, inplace=True)
    return df


def main(args):

    start = time.time()

    limit_frags_to_res_id = str2bool(args.limit_frags_to_res_id)
    limit_frags_to_chi = str2bool(args.limit_frags_to_chi)
    rotate_histidines = str2bool(args.rotate_histidines)
    rotate_phenylalanines = str2bool(args.rotate_phenylalanines)

    flip_symmetric = str2bool(args.flip_symmetric)

    output_dir = utils.path_ends_with_slash(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=f"{output_dir}fragment_picker_{args.output_prefix}_{args.theozyme_resnum}.log")
    cmd = ''
    for key, value in vars(args).items():
        cmd += f'--{key} {value} '
    cmd = f'{sys.argv[0]} {cmd}'
    logging.info(cmd)

    #import and prepare stuff
    database_dir = utils.path_ends_with_slash(args.database_dir)
    theozyme = utils.import_structure_from_pdb(args.theozyme_pdb)
    resnum, chain = split_pdb_numbering(args.theozyme_resnum)
    theozyme_residue = theozyme[0][chain][resnum]
    AA_alphabet = utils.import_structure_from_pdb(f'{database_dir}AA_alphabet.pdb')
    vdw_radii = import_vdw_radii(database_dir)


    if len(args.frag_pos_to_replace) > 1:
        frag_pos_to_replace = [i for i in range(args.frag_pos_to_replace[0], args.frag_pos_to_replace[1]+1)]
    else:
        frag_pos_to_replace = args.frag_pos_to_replace

    #sanity check command line input
    if args.frag_sec_struct_fraction:
        sec_structs = args.frag_sec_struct_fraction.split(',')
        sec_dict = {}
        for i in sec_structs:
            sec, frac = i.split(':')
            frac = float(frac)
            if frac > 1 or frac < 0:
                logging.error(f'Fraction for secondary structure {sec} must be a value between 0 and 1, but it is {frac}!')
                raise ValueError(f'Fraction for secondary structure {sec} must be a value between 0 and 1, but it is {frac}!')
            if (args.fragsize - frac * args.fragsize) < 1 and sec != args.rot_sec_struct and args.rot_sec_struct != None:
                logging.error(f"If limiting all fragment residues to secondary structure {sec}, it is not possible that the rotamer has secondary structure {args.rot_sec_struct}!")
                raise KeyError(f"If limiting all fragment residues to secondary structure {sec}, it is not possible that the rotamer has secondary structure {args.rot_sec_struct}!")
            elif (args.fragsize - frac * args.fragsize) < 1 and args.rot_sec_struct == None and len(sec_structs) == 1:
                log_and_print(f"Setting <rot_sec_struct> to {sec} because all residues in fragment have to have secondary structure {sec}!")
                args.rot_sec_struct = sec
            sec_dict[sec] = float(frac)
    else:
        sec_dict = None

    if args.ligand_chain:
        if not args.ligand_chain in [chain.id for chain in theozyme.get_chains()]:
            raise RuntimeError(f'No ligand found in chain {args.ligand_chain}. Please make sure the theozyme pdb is correctly formatted.')
        ligand = copy.deepcopy(theozyme[0][args.ligand_chain])
        ligand.detach_parent()
        ligand.id = "Z"
        for res in ligand.get_residues():
            res.id = (res.id[0], 1, res.id[2])
        log_and_print(f"Found ligand in chain {args.ligand_chain}.")
    else:
        ligand = None
        logging.warning(f"WARNING: No ligand chain specified. Was this intentional?")
        if args.his_central_atom == "auto" and theozyme_residue.get_resname() == "HIS":
            log_and_print("Ligand is required if using <flip_histidines='auto'>!")
            logging.error("Ligand is required if using <flip_histidines='auto'>!")
            raise RuntimeError(f"Ligand is required if using <flip_histidines='auto'>!")


    if args.channel_chain:
        if not args.channel_chain in [chain.id for chain in theozyme.get_chains()]:
            raise RuntimeError(f'No channel placeholder found in chain {args.channel_chain}. Please make sure the theozyme pdb is correctly formatted.')
        channel = copy.deepcopy(theozyme[0][args.channel_chain])
        channel.detach_parent()
        channel.id = "Q"
        log_and_print(f"Found channel placeholder in chain {args.channel_chain}.")
    else:
        channel = None
        log_and_print(f"No channel placeholder chain provided. Channel placeholder will be added automatically in following steps.")


    if len(args.frag_pos_to_replace) == 1:
        frag_pos_to_replace = frag_pos_to_replace
    else:
        frag_pos_to_replace = [i for i in range(args.frag_pos_to_replace[0], args.frag_pos_to_replace[1]+1)]

    if args.covalent_bond:
        if not args.ligand_chain:
            logging.warning("WARNING: Covalent bonds are only useful if ligand is present!")
        for cov_bond in args.covalent_bond.split(','):
            if not cov_bond.split(':')[0] in [atom.name for atom in theozyme_residue.get_atoms()]:
                raise KeyError(f"Could not find atom {cov_bond.split(':')[0]} from covalent bond {cov_bond} in residue {args.theozyme_resnum}!")
            if not cov_bond.split(':')[1] in [atom.name for atom in ligand.get_atoms()]:
                raise KeyError(f"Could not find atom {cov_bond.split(':')[1]} from covalent bond {cov_bond} in ligand chain {args.ligand_chain}!")

    database = utils.path_ends_with_slash(args.database_dir)

    if args.add_equivalent_func_groups in ["True", "1", "true", "yes", "Yes", "TRUE"]:
        residue_identities = identify_residues_with_equivalent_func_groups(theozyme_residue)
        log_and_print(f"Added residues with equivalent functional groups: {residue_identities}")
    else:
        residue_identities = [theozyme_residue.get_resname()]




    if args.fragment_pdb:

        #################################### BACKBONE ROTAMER FINDER ####################################
        bbfinder = True
        backbone = utils.import_structure_from_pdb(args.fragment_pdb)
        backbone = clean_input_backbone(backbone)
        backbone_df = create_df_from_fragment(backbone)
        backbone_residues = [res.id[1] for res in backbone.get_residues()]
        rotlibs = []
        log_and_print(f"Identifying rotamers...")
        for pos in frag_pos_to_replace:
            if not pos in backbone_residues:
                raise ValueError(f'Positions for rotamer insertion {frag_pos_to_replace} do not match up with backbone fragment {backbone_residues}')
            backbone_angles = extract_backbone_angles(backbone, pos)
            log_and_print(f"Position {pos} phi/psi angles: {backbone_angles['phi']} / {backbone_angles['psi']}.")
            rotlib = rotamers_for_backbone(residue_identities, database, backbone_angles["phi"], backbone_angles["psi"], args.prob_cutoff, 100, 2, 2)
            rotlib["rotamer_position"] = pos
            log_and_print(f"Found {len(rotlib.index)} rotamers for position {pos}.")
            rotlibs.append(rotlib)
        rotlib = pd.concat(rotlibs).reset_index(drop=True)
        rotlib = normalize_col(rotlib, 'log_prob', scale=True)
        rotlib = normalize_col(rotlib, 'log_occurrence', scale=True)
        rotlib = combine_normalized_scores(rotlib, 'rotamer_score', ['log_prob_normalized', 'log_occurrence_normalized'], [args.prob_weight, args.occurrence_weight], True, True)
        rotlib = rotlib.sort_values('rotamer_score', ascending=False).reset_index(drop=True)
        #print(rotlib)

        log_and_print(f"Found {len(rotlib.index)} rotamers in total.")
        rotlibcsv = utils.create_output_dir_change_filename(f'{output_dir}rotamer_info', args.output_prefix + f'_rotamers_{args.theozyme_resnum}_combined.csv')
        log_and_print(f"Writing phi/psi combinations to {rotlibcsv}.")
        rotlib.to_csv(rotlibcsv)

        frag_dict = {}
        for pos, rotlib in rotlib.groupby('rotamer_position'):
            pos_frags = []
            for index, row in rotlib.iterrows():
                df = copy.deepcopy(backbone_df)
                df.loc[pos - 1, [column for column in rotlib.columns if column.startswith("chi") or column == "probability" or column == "AA"]] = [row[column] for column in rotlib.columns if column.startswith("chi") or column == "probability" or column == "AA"]
                df['frag_num'] = index
                df['rotamer_pos'] = pos
                df['rotamer_score'] = row['rotamer_score']
                df['combined_fragment_score'] = df['rotamer_score']
                pos_frags.append(df)
            log_and_print(f"Created {len(pos_frags)} fragments for position {pos}.")
            frag_dict[pos] = pd.concat(pos_frags)

    else:

        #################################### FRAGMENT FINDER ####################################
        
        
        bbfinder = False
        log_and_print(f"Importing fragment library from {database_dir}/fraglib_energy.pkl")
        fraglib = import_fragment_library(f'{database_dir}/fraglib_energy.pkl')


        rotamer_positions_list = []
        rotlibs = []

        for residue_identity in residue_identities:
            #find rotamer library for given amino acid
            log_and_print(f"Importing backbone dependent rotamer library for residue {residue_identity} from {args.database_dir}")
            rotlib = return_residue_rotamer_library(database, residue_identity)
            rotlib = normalize_col(rotlib, 'log_prob', scale=True)
            rotlib = normalize_col(rotlib, 'log_occurrence', scale=True)
            rotlib = combine_normalized_scores(rotlib, 'rotamer_score', ['log_prob_normalized', 'log_occurrence_normalized'], [args.prob_weight, args.occurrence_weight], True, True)
            log_and_print(f"Identifying most probable rotamers for residue {residue_identity}")
            rotlib = identify_backbone_angles_suitable_for_rotamer(residue_identity, rotlib, f'{args.output_prefix}_{args.theozyme_resnum}_', f'{output_dir}rotamer_info', args.rot_sec_struct, args.phipsi_occurrence_cutoff, int(args.max_phi_psis / len(residue_identities)), args.rotamer_diff_to_best, args.rotamer_chi_binsize, args.rotamer_phipsi_binsize, args.prob_cutoff)
            log_and_print(f"Found {len(rotlib.index)} phi/psi/chi combinations.")
            rotlibs.append(rotlib)

        rotlib = pd.concat(rotlibs).sort_values("rotamer_score", ascending=False).reset_index(drop=True)
        rotlib = normalize_col(rotlib, 'log_prob', scale=True)
        rotlib = normalize_col(rotlib, 'log_occurrence', scale=True)
        rotlib = combine_normalized_scores(rotlib, 'rotamer_score', ['log_prob_normalized', 'log_occurrence_normalized'], [args.prob_weight, args.occurrence_weight], True, True)
        rotlib = rotlib.sort_values('rotamer_score', ascending=False).reset_index(drop=True)
        rotlibcsv = utils.create_output_dir_change_filename(f'{output_dir}rotamer_info', args.output_prefix + f'_rotamers_{args.theozyme_resnum}_combined.csv')
        log_and_print(f"Writing phi/psi combinations to {rotlibcsv}.")
        rotlib.to_csv(rotlibcsv)

        log_and_print(f"Identifying positions for rotamer insertion...")
        rotamer_positions = identify_positions_for_rotamer_insertion(fraglib, rotlib, args.rot_sec_struct, limit_frags_to_chi, limit_frags_to_res_id, args.phi_psi_bin)
        log_and_print(f"Found {len(rotamer_positions.index)} fitting positions.")

        log_and_print(f"Extracting fragments from rotamer positions...")
        frag_dict = extract_fragments(rotamer_positions, fraglib, frag_pos_to_replace, args.fragsize)
        frag_num = int(sum([len(frag_dict[pos].index) for pos in frag_dict]) / args.fragsize)
        log_and_print(f'Found {frag_num} fragments.')

        #filter fragments
        for pos in frag_dict:
            frag_nums = int(len(frag_dict[pos].index) / args.fragsize)
            log_and_print(f'Found {frag_nums} fragments for position {pos}.')
            if frag_nums == 0:
                frag_dict.pop(pos)
                continue
            if sec_dict:
                frag_dict[pos] = filter_frags_df_by_secondary_structure_content(frag_dict[pos], sec_dict)
                log_and_print(f"{int(len(frag_dict[pos]) / args.fragsize)} fragments passed secondary structure filtering with filter {args.frag_sec_struct_fraction} for position {pos}.")
            if args.frag_phipsi_min_res_score_cutoff:
                frag_dict[pos] = filter_frags_df_by_score(frag_dict[pos], args.frag_phipsi_min_res_score_cutoff, args.fragment_scoretype, "min_cutoff")
                log_and_print(f"{int(len(frag_dict[pos]) / args.fragsize)} fragments passed minimum residue {args.fragment_scoretype} score filtering with cutoff {args.frag_phipsi_min_res_score_cutoff} for position {pos}.")
            if args.frag_phipsi_mean_score_cutoff:
                frag_dict[pos] = filter_frags_df_by_score(frag_dict[pos], args.frag_phipsi_mean_score_cutoff, args.fragment_scoretype, "mean_min_cutoff")
                log_and_print(f"{int(len(frag_dict[pos]) / args.fragsize)} fragments passed mean {args.fragment_scoretype} score filtering with cutoff {args.frag_phipsi_mean_score_cutoff} for position {pos}.")
            if frag_dict[pos].empty:
                frag_dict.pop(pos)
                log_and_print(f"Could not find fragments for position {pos}.")

        if len(frag_dict) == 0:
            raise RuntimeError('Could not find any fragments that fit criteria! Try adjusting filter values!')
        
        combined_list = []
        combined = pd.concat([frag_dict[pos] for pos in frag_dict])
        group_df = combined.groupby('frag_num', sort=False).mean(numeric_only=True)
        group_df = normalize_col(group_df, 'occurrence_score', scale=True)['occurrence_score_normalized']
        combined = combined.merge(group_df, left_on='frag_num', right_index=True)
        combined = combine_normalized_scores(combined, 'combined_fragment_score', ['occurrence_score_normalized', 'rotamer_score'], [args.backbone_score_weight, args.rotamer_score_weight], True, True)
        for pos, df in combined.groupby('rotamer_pos', sort=False):
            log_and_print(f"Sorting fragments by combined fragment score for position {pos} with weights (backbone: {args.backbone_score_weight}, rotamer: {args.rotamer_score_weight}).")
            frag_dict[pos] = sort_frags_df_by_score(df, "combined_fragment_score")
        combined = combined.groupby('frag_num', sort=False).mean(numeric_only=True)
        _ = plots.violinplot_multiple_cols(combined, cols=['combined_fragment_score', 'occurrence_score_normalized', 'rotamer_score'], titles=['fragment score', 'backbone score', 'rotamer score'], y_labels=['AU', 'AU', 'AU'], dims=[(0, 1.1), (0, 1.1), (0, 1.1)], out_path=utils.create_output_dir_change_filename(f"{args.output_dir}/fragment_info", f"{args.output_prefix}_{args.theozyme_resnum}_pre_clash_filter.png"))
        del combined
        
    #################################### CREATE FRAGS, ATTACH ROTAMERS, FILTER ####################################

    residual_to_max = 0
    fragments = Structure.Structure('fragments')
    frags_table = []
    frags_info = []
    frag_num = 0

    for pos in frag_dict:
        num_channel_clash, num_bb_clash, num_sc_clash, rmsd_fails = 0, 0, 0, 0
        log_and_print(f'Creating fragments, attaching rotamer, superpositioning with theozyme residue, calculating rmsd to all accepted fragments with cutoff {args.rmsd_cutoff} A for position {pos}.')
        picked_frags = []
        frag_dfs = []
        #calculate maximum number of fragments per position, add missing fragments from previous position to maximum
        max_frags = int(args.max_frags / len(frag_dict)) + residual_to_max
        #loop over fragment dataframe, create fragments
        for frag_index, frag_df in frag_dict[pos].groupby('frag_num', sort=False):
            if len(picked_frags) < max_frags:
                frag = create_fragment_from_df(frag_df)
                frag = attach_rotamer_to_fragments(frag_df, frag, AA_alphabet)
                frag = align_to_sidechain(frag, frag[pos], theozyme_residue, False)
                picked_frags, frag_dfs, num_channel_clash, num_bb_clash, num_sc_clash, rmsd_fails = check_fragment(frag, picked_frags, frag_df, frag_dfs, ligand, channel, vdw_radii, pos, args.covalent_bond, args.rmsd_cutoff, args.backbone_ligand_clash_detection_vdw_multiplier, args.rotamer_ligand_clash_detection_vdw_multiplier, args.channel_fragment_clash_detection_vdw_multiplier, num_channel_clash, num_bb_clash, num_sc_clash, rmsd_fails)
                frag_df['flipped'] = False
                frag_df['rotated degrees'] = 0
                #flip rotamer and fragment if theozyme residue is tip symmetric or a histidine
                if flip_symmetric == True and theozyme_residue.get_resname() in tip_symmetric_residues()  and len(picked_frags) < max_frags:
                    flipped_frag = copy.deepcopy(frag)
                    flipped_frag_df = frag_df.copy()
                    flipped_frag_df['flipped'] = True
                    flipped_frag = align_to_sidechain(flipped_frag, frag[pos], theozyme_residue, True)
                    picked_frags, frag_dfs, num_channel_clash, num_bb_clash, num_sc_clash, rmsd_fails = check_fragment(flipped_frag, picked_frags, flipped_frag_df, frag_dfs, ligand, channel, vdw_radii, pos, args.covalent_bond, args.rmsd_cutoff, args.backbone_ligand_clash_detection_vdw_multiplier, args.rotamer_ligand_clash_detection_vdw_multiplier, args.channel_fragment_clash_detection_vdw_multiplier, num_channel_clash, num_bb_clash, num_sc_clash, rmsd_fails)
                if rotate_histidines == True and theozyme_residue.get_resname() == "HIS":
                    for deg in range(args.rotate_histidines_deg, 360, args.rotate_histidines_deg):
                        if len(picked_frags) >= max_frags:
                            break
                        rot_frag = copy.deepcopy(frag)
                        rot_frag_df = frag_df.copy()
                        rot_frag_df['rotated degrees'] = deg
                        rot_frag = rotate_histidine_fragment(rot_frag, deg, theozyme_residue, args.his_central_atom, ligand)
                        picked_frags, frag_dfs, num_channel_clash, num_bb_clash, num_sc_clash, rmsd_fails = check_fragment(rot_frag, picked_frags, rot_frag_df, frag_dfs, ligand, channel, vdw_radii, pos, args.covalent_bond, args.rmsd_cutoff, args.backbone_ligand_clash_detection_vdw_multiplier, args.rotamer_ligand_clash_detection_vdw_multiplier, args.channel_fragment_clash_detection_vdw_multiplier, num_channel_clash, num_bb_clash, num_sc_clash, rmsd_fails)
                if rotate_phenylalanines == True and theozyme_residue.get_resname() == "PHE":
                    for deg in range(args.rotate_phenylalanines_deg, 360, args.rotate_phenylalanines_deg):
                        if len(picked_frags) >= max_frags:
                            break
                        rot_frag = copy.deepcopy(frag)
                        rot_frag_df = frag_df.copy()
                        rot_frag_df['rotated degrees'] = deg
                        rot_frag = rotate_phenylalanine_fragment(rot_frag, deg, theozyme_residue)
                        picked_frags, frag_dfs, num_channel_clash, num_bb_clash, num_sc_clash, rmsd_fails = check_fragment(rot_frag, picked_frags, rot_frag_df, frag_dfs, ligand, channel, vdw_radii, pos, args.covalent_bond, args.rmsd_cutoff, args.backbone_ligand_clash_detection_vdw_multiplier, args.rotamer_ligand_clash_detection_vdw_multiplier, args.channel_fragment_clash_detection_vdw_multiplier, num_channel_clash, num_bb_clash, num_sc_clash, rmsd_fails)
      


            else:
                break
        
        log_and_print(f"Discarded {num_channel_clash} fragments that show clashes between backbone and channel placeholder with VdW multiplier {args.channel_fragment_clash_detection_vdw_multiplier}")
        log_and_print(f"Discarded {num_bb_clash} fragments that show clashes between backbone and ligand with VdW multiplier {args.backbone_ligand_clash_detection_vdw_multiplier}")
        log_and_print(f"Discarded {num_sc_clash} fragments that show clashes between sidechain and ligand with VdW multiplier {args.rotamer_ligand_clash_detection_vdw_multiplier}")
        log_and_print(f"Discarded {rmsd_fails} fragments that did not pass RMSD cutoff of {args.rmsd_cutoff} to all other picked fragments")


        log_and_print(f"Found {len(picked_frags)} fragments for position {pos} of a maximum of {max_frags}.")
        residual_to_max = max_frags - len(picked_frags)
        for frag, df in zip(picked_frags, frag_dfs):
            
            rot = identify_rotamer_position_by_probability(df)
            covalent_bonds = args.covalent_bond
            if covalent_bonds and args.add_equivalent_func_groups in ["True", "1", "true", "yes", "Yes", "TRUE"] and theozyme_residue.get_resname() != rot['AA']:
                covalent_bonds = ",".join([exchange_covalent(covalent_bond) for covalent_bond in covalent_bonds.split(",")])
            if covalent_bonds and rot['flipped'] == True:
                covalent_bonds = ",".join([flip_covalent(covalent_bond, rot["AA"]) for covalent_bond in covalent_bonds.split(",")])
            row = pd.Series({'model_num': frag_num, 'rotamer_pos': pos, 'AAs': df['AA'].to_list(), 'backbone_score': df['occurrence_score_normalized'].mean() if 'occurrence_score_normalized' in df.columns else 0, 'fragment_score': df['combined_fragment_score'].mean() if "combined_fragment_score" in df.columns else 0, 'phi_psi_occurrence': df['phi_psi_occurrence'].to_list() if 'phi_psi_occurrence' in df.columns else None, 'secondary_structure': df['ss'].to_list() if 'ss' in df.columns else None, 'rotamer_probability': float(df.dropna(subset = ['probability']).squeeze()['probability']), 'covalent_bond': covalent_bonds, 'rotamer_score': df['rotamer_score'].mean()})
            model = Model.Model(frag_num)
            model.add(frag)
            if ligand:
                model.add(ligand)
                row['ligand_chain'] = ligand.id
            if channel:
                model.add(channel)
                row['channel_chain'] = channel.id
            fragments.add(model)
            df['frag_num'] = frag_num
            frags_table.append(df)
            frags_info.append(row)
            frag_num += 1
        del(picked_frags)

    log_and_print(f'Found {len(frags_info)} fragments that passed all filters.')

    #write fragment info to disk
    frags_table = pd.concat(frags_table)
    fragscsv = utils.create_output_dir_change_filename(f'{output_dir}fragment_info', args.output_prefix + f'_fragments_{args.theozyme_resnum}.csv')
    log_and_print(f'Writing fragment details to {fragscsv}.')
    frags_table.to_csv(fragscsv)


    #write multimodel fragment pdb to disk
    filename_pdb = utils.create_output_dir_change_filename(output_dir, args.output_prefix + f'_{args.theozyme_resnum}.pdb')
    log_and_print(f'Writing multimodel fragment pdb to {filename_pdb}.')
    utils.write_multimodel_structure_to_pdb(fragments, filename_pdb)

    #write output json to disk
    frags_info = pd.DataFrame(frags_info)
    frags_info['poses'] = os.path.abspath(filename_pdb)
    frags_info['poses_description'] = f'{args.output_prefix}_{args.theozyme_resnum}'
    filename_json = utils.create_output_dir_change_filename(output_dir, args.output_prefix + f'_{args.theozyme_resnum}.json')
    log_and_print(f'Writing output json to {filename_json}.')
    frags_info.to_json(filename_json)

    if bbfinder == False:
        combined = frags_table.groupby('frag_num', sort=False).mean(numeric_only=True)
        _ = plots.violinplot_multiple_cols(combined, cols=['combined_fragment_score', 'occurrence_score_normalized', 'rotamer_score'], titles=['fragment score', 'backbone score', 'rotamer score'], y_labels=['AU', 'AU', 'AU'], dims=[(0, 1.1), (0, 1.1), (0, 1.1)], out_path=utils.create_output_dir_change_filename(f"{args.output_dir}/fragment_info", f"{args.output_prefix}_{args.theozyme_resnum}_post_filter.png"))
    log_and_print(f"Done in {round(time.time() - start, 1)} seconds!")


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # mandatory input
    argparser.add_argument("--database_dir", type=str, default="/home/tripp/riffdiff2/riff_diff/database/", help="Path to folder containing rotamer libraries, fragment library, etc.")
    argparser.add_argument("--theozyme_pdb", type=str, required=True, help="Path to pdbfile containing theozyme, must contain all residues in chain A numbered from 1 to n, ligand must be in chain Z (if there is one).")
    argparser.add_argument("--theozyme_resnum", required=True, help="Residue number with chain information (e.g. 25A) in theozyme pdb to find fragments for.")
    argparser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    argparser.add_argument("--output_prefix", type=str, required=True, help="Prefix for all output files")
    argparser.add_argument("--ligand_chain", type=str, default=None, help="Chain name of your ligand chain. Set None if no ligand is present")
    argparser.add_argument("--fragment_pdb", type=str, default=None, help="Path to backbone fragment pdb. If set, only this fragment (with different rotamers) will be returned instead of searching the fragment library.")

    # stuff you might want to adjust
    argparser.add_argument("--frag_pos_to_replace", type=int, default=[2,6], nargs='+', help="Position in fragment the rotamer should be inserted, can either be int or a list containing first and last position (e.g. 2,6 if rotamer should be inserted at every position from 2 to 6). Recommended not to include N- and C-terminus!")
    argparser.add_argument("--fragsize", type=int, default=7, help="Size of output fragments.")
    argparser.add_argument("--rot_sec_struct", type=str, default=None, help="Limit fragments to secondary structure at rotamer position. Provide string of one-letter code of dssp secondary structure elements (B, E, G, H, I, T, S, -), e.g. 'HE' if rotamer should be in helices or beta strands.")
    argparser.add_argument("--frag_sec_struct_fraction", type=str, default=None, help="Limit to fragments containing at least fraction of residues with the provided secondary structure. If fragment should have at least 50 percent helical residues OR 60 percent beta-sheet, pass 'H:0.5,E:0.6'")
    argparser.add_argument("--max_frags", type=int, default=100, help="Maximum number of frags that should be returned. Recommended value is <max_phi_psis> * len(frag_pos_to_replace).")
    argparser.add_argument("--rmsd_cutoff", type=float, default=2.0, help="Set minimum RMSD of output fragments. Increase to get more diverse fragments, but high values might lead to very long runtime or few fragments!")
    argparser.add_argument("--frag_phipsi_mean_score_cutoff", type=float, default=None, help="Minimum mean phi/psi occurrence score of fragments")
    argparser.add_argument("--frag_phipsi_min_res_score_cutoff", type=float, default=None, help="Minimum phi/psi occurrence score of a fragment residue")
    argparser.add_argument("--phipsi_occurrence_cutoff", type=float, default=0.8, help="Limit how common the phi/psi combination of a certain rotamer has to be. Value is in percent")
    argparser.add_argument("--covalent_bond", type=str, default=None, help="Add covalent bond(s) between rotamer and ligand in the form 'RotAtomA:LigAtomA,RotAtomB:LigAtomB'. Atom names should follow PDB numbering schemes, e.g. 'NZ:C3' for a covalent bond between a Lysine nitrogen and the third carbon atom of the ligand.")
    argparser.add_argument("--channel_chain", type=str, default=None, help="If adding a channel placeholder manually, provide the chain name here (important for clash detection!)")
    argparser.add_argument("--prob_cutoff", type=float, default=0.10, help="Do not return any phi/psi combinations with chi angle probabilities below this value")
    argparser.add_argument("--add_equivalent_func_groups", type=str, default="True", help="use ASP/GLU, GLN/ASN and VAL/ILE interchangeably")
    argparser.add_argument("--fragment_scoretype", type=str, default="occurrence_score", help="score type used for ranking fragments. can be either 'rosetta' (sum of omega, rama_prepro, hbond_sr_bb, p_aa_pp Rosetta scoreterms with ref15 weights) or 'phi_psi_occurrence' (how common a set of phi/psi angles is relative to all other phi/psi combinations * - 1).")

    # stuff you probably don't want to touch
    argparser.add_argument("--rotamer_chi_binsize", type=float, default=None, help="Filter for diversifying found rotamers. Lower numbers mean more similar rotamers will be found. Similar rotamers will still be accepted if their backbone angles are different. Recommended value: 15")
    argparser.add_argument("--rotamer_phipsi_binsize", type=float, default=None, help="Filter for diversifying found rotamers. Lower numbers mean similar rotamers from more similar backbone angles will be accepted. Recommended value: 50")
    argparser.add_argument("--limit_frags_to_res_id", type=str, default="True", help="Only pick fragments that contain the rotamer residue identity at the specified position with given phi/psi angle combination")
    argparser.add_argument("--limit_frags_to_chi", type=str, default="True", help="Only pick fragments that contain the rotamer with chi angles within the bin range of the target chi angle")
    argparser.add_argument("--phi_psi_bin", type=float, default=8, help="Binsize used to identify if fragment fits to phi/psi combination. Should not be above 10!")
    argparser.add_argument("--max_phi_psis", type=int, default=15, help="maximum number of phi/psi combination that should be returned. Can be increased if not enough fragments are found downstream (e.g. because secondary structure filter was used, and there are not enough phi/psi combinations in the output that fit to the specified secondary structure.")
    argparser.add_argument("--rotamer_diff_to_best", type=float, default=0.5, help="Accept rotamers that have a probability not lower than this percentage of the most probable rotamer. 1 means all rotamers will be accepted.")
    argparser.add_argument("--his_central_atom", type=str, default="auto", help="Only important if rotamer is HIS and <rotate_histidines> is True, sets the name of the atom that should not be flipped. If auto, the histidine nitrogen closest to the ligand is the coordinating atom. Can be manually set to NE2 or ND1")
    argparser.add_argument("--flip_symmetric", type=str, default="True", help="Flip tip symmetric residues (ARG, ASP, GLU, LEU, PHE, TYR, VAL).")
    argparser.add_argument("--rotamer_ligand_clash_detection_vdw_multiplier", type=float, default=0.75, help="Multiplier for VanderWaals radii for clash detection between rotamer and ligand. Clash is detected if a distance between atoms < (VdW_radius_atom1 + VdW_radius_atom2)*multiplier.")
    argparser.add_argument("--backbone_ligand_clash_detection_vdw_multiplier", type=float, default=1.0, help="Multiplier for VanderWaals radii for clash detection between fragment backbone and ligand. Clash is detected if a distance between atoms < (VdW_radius_atom1 + VdW_radius_atom2)*multiplier.")
    argparser.add_argument("--channel_fragment_clash_detection_vdw_multiplier", type=float, default=1.0, help="Multiplier for VanderWaals radii for clash detection between fragment backbone and channel placeholder. Clash is detected if a distance between atoms < (VdW_radius_atom1 + VdW_radius_atom2)*multiplier.")
    argparser.add_argument("--rotate_histidines", type=str, default="True", help="Rotate the orientation of histidine residues in <rotate_his_deg> steps to generate more fragment orientations")
    argparser.add_argument("--rotate_histidines_deg", type=float, default=30, help="Rotate fragments with histidines as catalytic residues around central atom around 360 degrees in <rotate_histidines_deg> steps.")
    argparser.add_argument("--rotate_phenylalanines", type=str, default="True", help="Rotate the orientation of phenylalanine residues in <rotate_phenylalanines_deg> steps to generate more fragment orientations")
    argparser.add_argument("--rotate_phenylalanines_deg", type=float, default=60, help="Rotate fragments with phenylalanines as catalytic residues around center in <rotate_phenylalanines_deg> steps.")
    argparser.add_argument("--prob_weight", type=float, default=3, help="Weight for rotamer probability importance when picking rotamers.")
    argparser.add_argument("--occurrence_weight", type=float, default=1, help="Weight for phi/psi-occurrence importance when picking rotamers.")
    argparser.add_argument("--backbone_score_weight", type=float, default=1, help="Weight for importance of fragment backbone score (boltzman score of backbone angle probabilities) when sorting fragments.")
    argparser.add_argument("--rotamer_score_weight", type=float, default=1, help="Weight for importance of rotamer score (combined score of probability and occurrence) when sorting fragments.")


    args = argparser.parse_args()


    main(args)
