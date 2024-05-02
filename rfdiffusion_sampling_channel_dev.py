#!/home/mabr3112/anaconda3/bin/python3.9
import sys
sys.path.append("/home/mabr3112/riff_diff")
sys.path += ["/home/mabr3112/projects/iterative_refinement/"]
sys.path += ["/home/markus/Desktop/script_development/iterative_refinement/"]

import json
from glob import glob
import os
import pandas as pd
import shutil
import itertools

# import custom modules
from iterative_refinement import *
import utils.plotting as plots 
import utils.biopython_tools
import utils.pymol_tools
from utils.plotting import PlottingTrajectory
import utils.metrics as metrics
import superimposition_tools
from protocols.composite_protocols import calculate_fastrelax_sidechain_rmsd
from protocols.composite_protocols import rosetta_scripts_and_mean
import utils.adrian_utils

### ADRIAN'S LIGAND CLASH DETECTION ####
def split_pdb_numbering(pdbnum: str) -> list[int,str]:
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
        structs.append(utils.adrian_utils.import_structure_from_pdb(row['poses']))
        ligands.append(utils.adrian_utils.import_structure_from_pdb(row[ref_frags_col])[0][ligand_chain])
        try:
            covalent_bonds.append(row['covalent_bonds'])
        except:
            covalent_bonds.append(None)
    vdw_radii = import_vdw_radii(database_dir)
    # calculate statistics of ligands:
    poses.poses_df[f"{prefix}_ligand_clash"] = [distance_detection(struct, ligand, vdw_radii, True, True, bb_clash_vdw_multiplier, covalent_bond) for struct, ligand, covalent_bond in zip(structs, ligands, covalent_bonds)]
    poses.poses_df[['poses_description', f"{prefix}_ligand_clash"]].to_json(scorefilepath)
    return poses

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

def fr_mpnn_esmfold(poses, prefix:str, n:int, fastrelax_pose_opts="fr_pose_opts", ref_pdb_col:str=None, ref_motif_col="motif_residues", mpnn_fixedres_col:str=None, use_soluble_model:bool=False, params_file:str=None) -> Poses:
    '''AAA'''
    # run fastrelax on predicted poses
    fr_opts = f"-beta -parser:protocol {args.refinement_protocol}"
    if params_file: fr_opts += f" -extra_res_fa {params_file}"
    fr = poses.rosetta("rosetta_scripts.default.linuxgccrelease", options=fr_opts, pose_options=poses.poses_df[fastrelax_pose_opts].to_list(), n=n, prefix=f"{prefix}_refinement")
    
    # calculate RMSDs and filter
    rmsds = poses.calc_motif_bb_rmsd_df(ref_pdb=ref_pdb_col, ref_motif=ref_motif_col, target_motif="motif_residues", metric_prefix=f"{prefix}_refinement_bb_ca")
    fr_comp_score = poses.calc_composite_score(f"{prefix}_fr_comp_score", [f"{prefix}_refinement_total_score", f"{prefix}_refinement_bb_ca_motif_rmsd"], [1,1])
    fr_filter = poses.filter_poses_by_score(2, f"{prefix}_fr_comp_score", remove_layers=1, prefix=f"{prefix}_refinement_filter", plot=[f"{prefix}_refinement_total_score", f"{prefix}_refinement_bb_ca_motif_rmsd"])

    # design and predict:
    poses = mpnn_design_and_esmfold(poses, prefix=prefix, num_mpnn_seqs=48, num_esm_inputs=16, num_esm_outputs_per_input_backbone=5, motif_ref_pdb_col=ref_pdb_col, bb_rmsd_col=f"{prefix}_refinement_location", rmsd_weight=3, mpnn_fixedres_col=mpnn_fixedres_col, use_soluble_model=use_soluble_model, disfavor_alanines=1)
    return poses

def mpnn_fr(poses, prefix:str, fastrelax_pose_opts="fr_pose_opts", pdb_location_col:str=None, reference_location_col="input_poses", params_file:str=None):
    '''AAA'''
    def collapse_dict_values(in_dict: dict) -> str:
        return ",".join([str(y) for x in in_dict.values() for y in list(x)])
    def write_pose_opts(row: pd.Series, mpnn_col:str, reference_location_col:str="input_poses", resfiles_col:str=None, fd_motif_col:str=None) -> str:
        return f"-in:file:native {row[reference_location_col]} -parser:script_vars seq={row[mpnn_col]} motif_res={collapse_dict_values(row['motif_residues'])} cat_res={collapse_dict_values(row['fixed_residues'])} fd_motif={collapse_dict_values(row[fd_motif_col])} resfile={row[resfiles_col]} substrate_chain={args.ligand_chain}"

    # mpnn design on backbones
    mpnn_designs = poses.mpnn_design(mpnn_options=f"--num_seq_per_target=1 --sampling_temp=0.05", prefix=f"{prefix}_mpnn", fixed_positions_col="fixed_residues")

    # reset poses to Structures:
    fr_opts = f"-beta -parser:protocol {args.fastrelax_protocol}"
    if params_file: fr_opts += f" -extra_res_fa {params_file}"
    poses.poses_df["poses"] = poses.poses_df[pdb_location_col]
    poses.poses_df["poses_description"] = poses.poses_df["poses"].str.split("/").str[-1].str.replace(".pdb","")

    # copy ligand into poses
    ligposes = poses.add_ligand_from_ref(ref_col=reference_location_col, ref_motif="motif_residues", target_motif="motif_residues", lig_chain=args.ligand_chain, prefix=f"{prefix}_lig_poses", overwrite=True)

    # define motif around ligand:
    poses.poses_df[f"{prefix}_fd_motif"] = [get_design_residues(row, motif_res_col="motif_residues", cat_res_col="fixed_residues", lig_chain=args.ligand_chain) for index, row in poses.poses_df.iterrows()]

    # calculate MPNN probabilities and write resfiles
    probs = poses.get_mpnn_probs(f"{prefix}_probs")
    resfiles = poses.write_resfiles_from_mpnn_probs(colname=f"{prefix}_resfiles", probabilities_location_col=f"{prefix}_probs", motif_col=f"{prefix}_fd_motif", motif_chain="A", motif_threshold=0.05)
    
    # write pose_opts (because sequence changes every time!!!)
    mpnn_col = f"{prefix}_mpnn_sequence"
    poses.poses_df[fastrelax_pose_opts] = [write_pose_opts(row, mpnn_col, reference_location_col=reference_location_col, resfiles_col=f"{prefix}_resfiles", fd_motif_col=f"{prefix}_fd_motif") for index, row in poses.poses_df.iterrows()]

    # fastrelax
    fr = poses.rosetta("rosetta_scripts.default.linuxgccrelease", options=fr_opts, pose_options=poses.poses_df[fastrelax_pose_opts].to_list(), n=1, prefix=f"{prefix}_fr")
    
    # calculate motif RMSDs
    fr_motif_ca_rmsds = poses.calc_motif_bb_rmsd_df(ref_pdb=reference_location_col, ref_motif="motif_residues", target_motif="motif_residues", metric_prefix=f"{prefix}_fr_bb_ca")

    return poses, fr

def collapse_dict_values(in_dict: dict) -> str:
    return ",".join([str(y) for x in in_dict.values() for y in list(x)])

def write_fastdesign_opts(row: pd.Series, cycle: int, total_cycles: int, reference_location_col:str, designres_col: str, motif_res_col: str, cat_res_col: str, resfile_col: str) -> str:
    def collapse_dict_values(in_dict: dict) -> str:
        return ",".join([str(y) for x in in_dict.values() for y in list(x)])
    return f"-in:file:native {row[reference_location_col]} -parser:script_vars motif_res={collapse_dict_values(row[motif_res_col])} cat_res={collapse_dict_values(row[cat_res_col])} input_res={collapse_dict_values(row[designres_col])} substrate_chain={args.ligand_chain} sd={0.8 - (0.4 * cycle/total_cycles)} resfile={row[resfile_col]}"

def mpnn_design_and_esmfold(poses, prefix:str, num_mpnn_seqs:int=20, num_esm_inputs:int=8, num_esm_outputs_per_input_backbone:int=1, motif_ref_pdb_col:str=None, bb_rmsd_col:str=None, rmsd_weight:float=1, mpnn_fixedres_col:str=None, use_soluble_model=False, ref_motif_col:str="motif_residues", motif_col:str="motif_residues", ref_catres_motif_col:str="fixed_residues", catres_motif_col:str="fixed_residues", disfavor_alanines:int=1, calc_perplexity=False):
    '''AAA'''
    # Run MPNN and filter (by half)
    mpnn_designs = poses.mpnn_design(mpnn_options=f"--num_seq_per_target={num_mpnn_seqs} --sampling_temp=0.1 --omit_AAs=C", prefix=f"{prefix}_mpnn", fixed_positions_col=mpnn_fixedres_col or "fixed_residues", use_soluble_model=use_soluble_model)
    poses.poses_df[f"{prefix}_alanine_content"] = poses.poses_df[f"{prefix}_mpnn_sequence"].str.count("A") / poses.poses_df[f"{prefix}_mpnn_sequence"].str.len()
    alanine_content_weight = (poses.poses_df[f"{prefix}_alanine_content"].mean() - 0.1) * 8 * disfavor_alanines 
    print(f"alanine_content_weight: {alanine_content_weight}")

    # calc composite score between MPNN score and alanine content:
    mpnn_compscore = poses.calc_composite_score(f"{prefix}_mpnn_compscore", [f"{prefix}_mpnn_score", f"{prefix}_alanine_content"], [1, alanine_content_weight])
    mpnn_seqfilter = poses.filter_poses_by_score(num_esm_inputs, f"{prefix}_mpnn_compscore", prefix=f"{prefix}_mpnn_seqfilter", remove_layers=1, plot=[f"{prefix}_mpnn_score", f"{prefix}_alanine_content"])
    print(f"mpnn_design_and_esmfold, after mpnn filtering: {len(poses.poses_df['poses'])}")

    # Run ESMFold and calc bb_ca_rmsd, motif_ca_rmsd and motif_heavy RMSD
    if calc_perplexity: esm_singlepass_logprobs = poses.calc_esm2_pseudo_perplexity(options="--singlepass True", prefix=f"round1_singlepass", max_cores=1000)
    esm_preds = poses.predict_sequences(run_ESMFold, prefix=f"{prefix}_esm")
    print(f"mpnn_design_and_esmfold, After prediction: {len(poses.poses_df['poses'])}")
    esm_bb_ca_rmsds = poses.calc_bb_rmsd_df(ref_pdb=bb_rmsd_col, metric_prefix=f"{prefix}_esm")
    esm_motif_rmsds = poses.calc_motif_bb_rmsd_df(ref_pdb=motif_ref_pdb_col, ref_motif=ref_motif_col, target_motif=motif_col, metric_prefix=f"{prefix}_esm_bb_ca")
    esm_motif_heavy_rmsds = poses.calc_motif_heavy_rmsd_df(ref_pdb=motif_ref_pdb_col, ref_motif=ref_catres_motif_col, target_motif=catres_motif_col, metric_prefix=f"{prefix}_esm_catres")

    # Filter Redesigns based on confidence and RMSDs
    print(f"mpnn_design_and_esmfold, Before filtering: {len(poses.poses_df['poses'])}")
    esm_comp_score = poses.calc_composite_score(f"{prefix}_esm_comp_score", [f"{prefix}_esm_plddt", f"{prefix}_esm_bb_ca_motif_rmsd"], [-1, rmsd_weight])
    esm_filter = poses.filter_poses_by_score(num_esm_outputs_per_input_backbone, f"{prefix}_esm_comp_score", remove_layers=1, prefix=f"{prefix}_esm_filter", plot=[f"{prefix}_esm_comp_score", f"{prefix}_esm_plddt", f"{prefix}_esm_bb_ca_rmsd", f"{prefix}_esm_bb_ca_motif_rmsd", f"{prefix}_esm_catres_motif_heavy_rmsd"])
    print(f"mpnn_design_and_esmfold, After filtering: {len(poses.poses_df['poses'])}")
    
    # Plot Results
    if not os.path.isdir((plotdir := f"{poses.dir}/plots")): os.makedirs(plotdir, exist_ok=True)

    # ESM stats:
    cols = [f"{prefix}_esm_plddt", f"{prefix}_esm_bb_ca_rmsd", f"{prefix}_esm_bb_ca_motif_rmsd", f"{prefix}_esm_catres_motif_heavy_rmsd"]
    titles = ["ESM pLDDT", "ESM BB-Ca RMSD", "ESM Motif-Ca RMSD", "ESM Catres\nSidechain RMSD"]
    y_labels = ["pLDDT", "RMSD [\u00C5]", "RMSD [\u00C5]", "RMSD [\u00C5]"]
    dims = [(0,100), (0,15), (0,8), (0,8)]
    _ = plots.violinplot_multiple_cols(poses.poses_df, cols=cols, titles=titles, y_labels=y_labels, dims=dims, out_path=f"{plotdir}/{prefix}_esm_stats.png")
    return poses

def convert_sampled_mask(old_contig):
    '''converts sampled mask output of RFDiffusion into new motif'''
    def parse_filler(elem):
        return [x for x in elem.split("-") if x][-1]
    def parse_contig_elem(elem, counter):
        start, end = [int(x) for x in elem[1:].split("-")]
        return end+counter, f"A{start+counter}-{end+counter}"
    
    contig_elems = old_contig[0].split("/") if type(old_contig) == list else old_contig.split("/")
    new_contig_list = list()
    counter = 0
    
    for elem in contig_elems:
        if elem[0].isalpha():
            counter, sub_contig = parse_contig_elem(elem, counter)
            new_contig_list.append(sub_contig)
        else:
            new_contig_list.append((num := parse_filler(elem)))
            counter += int(num)
            
    return "/".join(new_contig_list)

def create_inpaint_seq(contig, motif):
    '''AAA'''
    def parse_inpaint_substr(elem, fixedres) -> str:
        inpaint_str_l = list()
        start, end = [int(x) for x in elem[1:].split("-")]
        for i in range(start, end+1):
            inpaint_str_l.append(elem[0]+str(i))
        return [x for x in inpaint_str_l if x not in fixedres]
    
    # split contigs, start list
    contig_elems = contig.split("/")
    inpaint_str_list = list()
    
    # assemble fixed_residues:
    fixed_res = [str(k)+str(idx) for k, v in motif.items() for idx in v]
    
    # assemble inpaint_seq piece by piece
    for elem in contig_elems:
        if elem[0].isalpha():
            [inpaint_str_list.append(x) for x in parse_inpaint_substr(elem, fixed_res)]
    return "/".join(inpaint_str_list)

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

def adjust_flanking(rfdiffusion_pose_opts: str, flanking_type: str, total_flanker_length:int=None) -> str:
    '''AAA'''
    def get_contigs_str(rfdiff_opts: str) -> str:
        elem = [x for x in rfdiff_opts.split(" ") if x.startswith("'contigmap.contigs=")][0]
        contig_start = elem.find("[") +1
        contig_end = elem.find("]")
        return elem[contig_start:contig_end]
    
    # extract contig from contigs_str
    contig = get_contigs_str(rfdiffusion_pose_opts)
    
    # extract flankings and middle part
    csplit = contig.split("/")
    og_nterm, middle, og_cterm = int(csplit[0]), "/".join(csplit[1:-1]), int(csplit[-1])
    
    # readjust flankings according to flanking_type and max_pdb_length
    pdb_length = total_flanker_length or og_nterm+og_cterm
    nterm, cterm = divide_flanking_residues(pdb_length, flanking=flanking_type)
    
    # reassemble contig string and replace with hallucinate pose opts.
    reassembled = f"{nterm}/{middle}/{cterm}"
    return rfdiffusion_pose_opts.replace(contig, reassembled)

def update_and_copy_reference_frags(input_df: pd.DataFrame, ref_col:str, desc_col:str, motif_prefix: str, out_pdb_path=None, keep_ligand_chain:str="") -> list[str]:
    ''''''
    # create residue mappings {old: new} for renaming
    list_of_mappings = [utils.biopython_tools.residue_mapping_from_motif(ref_motif, inp_motif) for ref_motif, inp_motif in zip(input_df[f"{motif_prefix}_con_ref_pdb_idx"].to_list(), input_df[f"{motif_prefix}_con_hal_pdb_idx"].to_list())]

    # compile list of output filenames
    output_pdb_names_list = [f"{out_pdb_path}/{desc}.pdb" for desc in input_df[desc_col].to_list()]

    # renumber
    return [utils.biopython_tools.renumber_pdb_by_residue_mapping(ref_frag, res_mapping, out_pdb_path=pdb_output, keep_chain=keep_ligand_chain) for ref_frag, res_mapping, pdb_output in zip(input_df[ref_col].to_list(), list_of_mappings, output_pdb_names_list)]

def parse_outfilter_args(scoreterm_str: str, weights_str: str, df: pd.DataFrame, prefix:str=None) -> tuple[list]:
    ''''''
    def check_for_col_in_df(col: str, datf: pd.DataFrame) -> None:
        if col not in datf.columns: raise KeyError(f"Scoreterm {col} not found in poses_df. Available scoreterms: {','.join(datf.columns)}")
    scoreterms = scoreterm_str.split(",")
    if prefix: scoreterms = [prefix + "_" + x for x in scoreterms]
    weights = [float(x) for x in weights_str.split(",")]
    check = [check_for_col_in_df(scoreterm, df) for scoreterm in scoreterms]

    if not len(scoreterms) == len(weights): raise ValueError(f"Length of --output_scoreterms ({scoreterm_str}: {len(scoreterm_str)}) and --output_scoreterm_weights ({weights_str}: {len(weights_str)}) is not the same. Both arguments must be of the same length!")

    return scoreterms, weights

def extract_rosetta_pose_opts(input_data: pd.Series) -> str:
    '''AAA'''
    def collapse_dict_values(in_dict: dict) -> str:
        return ",".join([str(y) for x in in_dict.values() for y in list(x)])
    native_file = f"-in:file:native ref_fragments/{input_data['poses_description']}.pdb"
    script_vars = f"-parser:script_vars motif_res='{collapse_dict_values(input_data['motif_residues'])}' cat_res='{collapse_dict_values(input_data['fixed_residues'])}'"
    return [" ".join([native_file, script_vars])]

def write_rosetta_pose_opts_to_json(input_df: pd.DataFrame, path_to_json_file: str) -> str:
    '''AAA'''
    pose_opts_dict = {input_df.loc[index, "poses_description"]: extract_rosetta_pose_opts(input_df.loc[index]) for index in input_df.index}
    with open(path_to_json_file, 'w') as f:
        json.dump(pose_opts_dict, f)
    return path_to_json_file

def parse_diffusion_options(default_opts: str, additional_opts: str) -> str:
    '''AAA'''
    def_opts = [x for x in default_opts.split(" ") + additional_opts.split(" ") if x]
    def_opts_dict = {x.split("=")[0]: "=".join(x.split("=")[1:]) for x in def_opts}
    return " ".join([f"{k}={v}" for k, v in def_opts_dict.items()])

def check_for_params(path: str, ligand_chain: str):
    '''Checks if params-file is present in output of run_ensemble_evaluator.py //
        If no params file could be generated, then we also do not want Rosetta to use the ligand information.
    
        TODO: Implement flag -extra_res_fa if params file is found to make sure that the ligand is properly loaded.
    '''
    if os.path.isfile(f"{path}/ligand/LG1.params"): return ligand_chain
    else: return None

def calc_ligand_stats(input_df: pd.DataFrame, ref_frags_col:str, ref_motif_col:str, poses_motif_col:str, prefix:str, ligand_chain:str="Z") -> None:
    '''
    Superimposes the poses onto reference fragments in input_df[ref_frags_col] by specified motifs in input_df. Then calculates statistics over ligands. (if it is clashing and the number of contacts).
    '''
    # superimpose reference frags onto poses to make sure ligand calculation works in the same coordinate frame:
    poses = [superimposition_tools.superimpose_pdb_by_motif(ref_frag, pose, fixed_motif=ref_motif, mobile_motif=pose_motif, atoms=["CA"]) for pose, ref_frag, pose_motif, ref_motif in zip(input_df["poses"].to_list(), input_df[ref_frags_col].to_list(), input_df[poses_motif_col].to_list(), input_df[ref_motif_col].to_list())]

    # calculate statistics of ligands:
    loaded_poses = [utils.biopython_tools.load_structure_from_pdbfile(pose) for pose in poses]
    input_df[f"{prefix}_ligand_clash"] = [utils.metrics.check_for_ligand_clash_of_pdb(pose, ligand_chain=ligand_chain, ligand_pdb_path=ref_pose, dist=1.8, ignore_atoms=["H"]) for pose, ref_pose in zip(input_df["poses"].to_list(), input_df[ref_frags_col].to_list())]
    input_df[f"{prefix}_peratom_ligand_contacts"] = [utils.metrics.calc_ligand_contacts_of_pdb(pose, ligand_chain=ligand_chain, ligand_pdb_path=ref_pose, d_0=3.4, r_0=3.5, ignore_atoms=["H"]) for pose, ref_pose in zip(input_df["poses"].to_list(), input_df[ref_frags_col].to_list())]

    # calculate pocket scores
    input_df[f"{prefix}_pocket_score"] = [utils.metrics.calc_pocket_score(pose, ligand_chain=ligand_chain, rep_weight=4, coordination_strength=5, coordination_radius=8) for pose in loaded_poses]
    input_df[f"{prefix}_pocket_score_v2"] = [utils.metrics.calc_pocket_score_v2(pose, ligand_chain=ligand_chain) for pose in loaded_poses]
    input_df[f"{prefix}_pocket_score_v3"] = [utils.metrics.calc_pocket_score_v3(pose, ligand_chain=ligand_chain) for pose in loaded_poses]

    return input_df

def overwrite_linker_length(pose_opts: str, total_length:int, max_linker_length:int=100) -> str:
    '''overwrites linker length and allows linkers to be of any length (with at least the provided linker length)'''
    # extract contig string from pose_opts
    full_contig_str = [x for x in pose_opts.split(" ") if x.startswith("'contigmap.contigs")][0]
    contig_str = full_contig_str[full_contig_str.find("[")+1:full_contig_str.find("]")]
    contigs = [x for x in contig_str.split("/") if x][1:-1]
    
    # replace fixed linkers in contigs string with linker ranges
    new_contigs = "/".join([x if x[0].isalpha() else f"{x}-{str(max_linker_length)}" for x in contigs])
    new_contig_str = full_contig_str.replace("/".join(contigs), new_contigs)

    # return replaced contig pose-opts:
    return pose_opts.replace(full_contig_str, f"{new_contig_str} contigmap.length={str(total_length)}-{str(total_length)} ")

def get_design_residues(row: pd.Series, motif_res_col:str, cat_res_col: str, lig_chain:str) -> dict:
    ''' calculates design residues from poses '''
    pose = utils.biopython_tools.load_structure_from_pdbfile(row["poses"])
    motif_centroid_res = utils.biopython_tools.select_motif_centroid_contacts(pose, motif=row[motif_res_col], dist=8, pose_sidechains_only=True)
    lig_contacts = utils.biopython_tools.select_ligand_contacts(pose, ligand_chain=lig_chain, dist=7, pose_sidechains_only=True)
    return {"A": utils.biopython_tools.concat_motifs([motif_centroid_res, lig_contacts, row[cat_res_col]])["A"]}

def update_covalent_bonds_str(covalent_bonds_str: str, template_motif: dict, motif: dict) -> str:
    ''''''
    def collapse_dict_values(in_dict: dict) -> str:
        return [f"{str(res)}{chain}" for chain, reslist in in_dict.items() for res in list(reslist)]
    
    # split covalent_bonds string into its individual entries:
    covalent_bonds = [x for x in covalent_bonds_str.split(",") if x]

    # create residue mapping from motifs to change res_id
    res_mapping = {old_id: new_id for old_id, new_id in zip(collapse_dict_values(template_motif), collapse_dict_values(motif))}

    # replace res_id (resnum + chain) with res_id in new motif:
    return ",".join([bond_str.replace((res_id := bond_str.split("_")[0]), res_mapping[res_id]) for bond_str in covalent_bonds])

def mpnn_probs_fd(poses, motif_col:str):
    '''AAA'''
    # collect mpnn_probabilities from poses and write resfiles

    # write fastdesign pose_options and run fastdesign

    # predict w/ ESMFold and calculate stats

    # plot 
    return poses, index_layers

def get_params_file(string: str) -> str:
    '''Checks if args.params_file contains a params file path. If not, it looks for an automatically generated params file. If there is none either, it will not use any params file (return None)'''
    if string: return string
    else:
        if os.path.isfile((params := f"{args.input_dir}/ligand/LG1.params")):
            return params
        else:
            return None

def main(args):
    # print Status
    print(f"\n{'#'*50}\nRunning rfdiffusion_ensembles_sampling.py on {args.input_dir}\n{'#'*50}\n")

    # Parse Poses
    pdb_dir = f"{args.input_dir}/pdb_in/"
    ensembles = Poses(args.output_dir, glob(f"{pdb_dir}/*.pdb"))
    ensembles.max_rfdiffusion_gpus = args.max_rfdiffusion_gpus
    plot_dir = ensembles.plot_dir
    keep_ligand_chain = args.ligand_chain
    params_file = get_params_file(args.params_file)
    if params_file: ensembles.poses_df["params_file_path"] = [f"{args.input_dir}/ligand/LG1.params" for x in ensembles.poses_df["poses"].to_list()]

    # Read scores of selected paths from ensemble_evaluator and store them in poses_df:
    path_df = pd.read_json(f"{args.input_dir}/selected_paths.json").reset_index().rename(columns={"index": "rdescription"})
    ensembles.poses_df = ensembles.poses_df.merge(path_df, left_on="poses_description", right_on="rdescription")
    ensembles.max_rosetta_cpus = 1000
    ensembles.max_mpnn_gpus = 10

    # change cterm and nterm flankers according to input args.
    if args.flanking: ensembles.poses_df["rfdiffusion_pose_opts"] = [adjust_flanking(rfdiffusion_pose_opts_str, args.flanking, args.total_flanker_length) for rfdiffusion_pose_opts_str in ensembles.poses_df["rfdiffusion_pose_opts"].to_list()]
    elif args.total_flanker_length:
        raise ValueError(f"Argument 'total_flanker_length' was given, but not 'flanking'! Both args have to be provided.")
    
    # adjust linkers if overwrite_linker_lengths option was set:
    if args.overwrite_linker_lengths:
        linker_length, total_length = [int(x) for x in args.overwrite_linker_lengths.split(",")]
        ensembles.poses_df["rfdiffusion_pose_opts"] = [overwrite_linker_length(pose_opts, total_length, linker_length) for pose_opts in ensembles.poses_df["rfdiffusion_pose_opts"].to_list()]

    # Check if merger was successful:
    if len(ensembles.poses_df) == len(ensembles.poses): print(f"Loading of Pose contigs into poses_df successful. Continuing to hallucination.")
    else: raise ValueError(f"Merging of diffusion_opts into poses_df failed! Check if keys in hallucination_opts match with pose_names!!!")

    # store original motifs for calculation of Motif RMSDs later
    motif_cols = ["fixed_residues", "motif_residues"]
    ensembles.poses_df["template_motif"] = ensembles.poses_df["motif_residues"]
    ensembles.poses_df["template_fixedres"] = ensembles.poses_df["fixed_residues"]

    # RFdiffusion:
    diffusion_options = f"diffuser.T={str(args.rfdiffusion_timesteps)} potentials.guide_scale={args.rfdiff_guide_scale} inference.num_designs={args.num_rfdiffusions} potentials.guiding_potentials=[\\'type:substrate_contacts,weight:0\\',\\'type:substrate_contacts_positive,weight:{args.pot_weight},pos_weight:{args.pos_weight},attr_dist:{args.attr_dist},decentralize:{args.decentralize}\\'] potentials.guide_decay={args.guide_decay}"
    diffusion_options = parse_diffusion_options(diffusion_options, args.rfdiffusion_additional_options)

    # if custom center should be added:
    if args.custom_diffusion_center.lower() == "true":
        # from dataframe (added during run_ensemble_evaluator.py)
        c_x, c_y, c_z = ensembles.poses_df.loc[0, "diffusion_custom_center"].split(",")
        diffusion_options = diffusion_options.replace(",decentralize", f",rc_x:{c_x},rc_y:{c_y},rc_z:{c_z},decentralize")
    elif len(args.custom_diffusion_center.split(",")) == 3:
        # from commandline by specifying coordinates:
        c_x, c_y, c_z = args.custom_diffusion_center.split(",")
        diffusion_options = diffusion_options.replace(",decentralize", f",rc_x:{c_x},rc_y:{c_y},rc_z:{c_z},decentralize")

    ensembles.poses_df["rfdiffusion_pose_opts"] = [x.replace("contigmap.contigs=[", f"contigmap.contigs=[{args.channel_contig}/0 ") for x in ensembles.poses_df["rfdiffusion_pose_opts"].to_list()]
    diffusions = ensembles.rfdiffusion(options=diffusion_options, pose_options=list(ensembles.poses_df["rfdiffusion_pose_opts"]), prefix="rfdiffusion", max_gpus=args.max_rfdiffusion_gpus)

    ######################### RFDiffusion POSTPROCESSING #####################################################
    # remove channel peptide from diffused poses:
    chains_removed = ensembles.remove_chain_from_poses(remove_chain="B", prefix="postdiffusion_chainremoval") 

    # Update motif_res and fixedres to residue mapping after rfdiffusion 
    _ = [ensembles.update_motif_res_mapping(motif_col=col, inpaint_prefix="rfdiffusion") for col in motif_cols]
    _ = ensembles.update_res_identities(identity_col="catres_identities", inpaint_prefix="rfdiffusion")
    ensembles.poses_df["covalent_bonds"] = [update_covalent_bonds_str(cov_bond_str, template_motif, motif) for cov_bond_str, template_motif, motif in zip(ensembles.poses_df["covalent_bonds"].to_list(), ensembles.poses_df["template_motif"].to_list(), ensembles.poses_df["motif_residues"].to_list())]

    # calculate ROG and contacts:
    ensembles.poses_df["rfdiffusion_rog"] = [metrics.calc_rog_of_pdb(pose) for pose in ensembles.poses_df["poses"].to_list()]
    #ensembles.poses_df["rfdiffusion_contacts_short"] = [metrics.calc_intra_contacts_of_pdb(pose) for pose in ensembles.poses_df["poses"].to_list()]
    #ensembles.poses_df["rfdiffusion_contacts_long"] = [metrics.calc_intra_contacts_of_pdb(pose, d_0=3.9, r_0=3.9) for pose in ensembles.poses_df["poses"].to_list()]

    # Calculate RMSD and composite score:
    diffusion_template_rmsd = ensembles.calc_motif_bb_rmsd_dir(ref_pdb_dir=pdb_dir, ref_motif=list(ensembles.poses_df["template_motif"]), target_motif=list(ensembles.poses_df["motif_residues"]), metric_prefix="rfdiffusion_template_bb_ca", remove_layers=1)

    # Copy and rewrite Fragments into output_dir/reference_fragments
    if not os.path.isdir((updated_ref_frags_dir := f"{ensembles.dir}/updated_reference_frags/")): os.makedirs(updated_ref_frags_dir)
    
    ensembles.poses_df["updated_reference_frags_location"] = update_and_copy_reference_frags(ensembles.poses_df, ref_col="input_poses", desc_col="poses_description", motif_prefix="rfdiffusion", out_pdb_path=updated_ref_frags_dir, keep_ligand_chain=args.ligand_chain)
    
    # superimpose poses on reference frags and calculate ligand scores:
    #if keep_ligand_chain: 
    ligposes = ensembles.add_ligand_from_ref(ref_col="updated_reference_frags_location", ref_motif="motif_residues", target_motif="motif_residues", lig_chain=args.ligand_chain, prefix="postdiffusion_lig_poses")
    calc_ligand_stats(input_df=ensembles.poses_df, ref_frags_col="updated_reference_frags_location", ref_motif_col="motif_residues", poses_motif_col="motif_residues", prefix="rfdiffusion", ligand_chain=args.ligand_chain)

    # filter based on rfdiffusion pLDDT (implement args.rfdiffusion_plddt_fraction):
    rfdiff_plddt_filter = ensembles.filter_poses_by_score(0.95, "rfdiffusion_plddt", prefix="rfdiffusion_plddt_filter", ascending=False, plot=["rfdiffusion_plddt", "rfdiffusion_template_bb_ca_motif_rmsd", "rfdiffusion_peratom_ligand_contacts", "rfdiffusion_pocket_score", "rfdiffusion_pocket_score_v2", "rfdiffusion_pocket_score_v3"])

    # filter based on ligand_contacts:
    rfdiff_pocket_comp_score = ensembles.calc_composite_score(f"pocket_comp_score", ["rfdiffusion_pocket_score_v2", "rfdiffusion_peratom_ligand_contacts"], [-1, 1])
    rfdiff_contacts_filter = ensembles.filter_poses_by_score(0.95, "pocket_comp_score", prefix="rfdiffusion_pocket_filter", ascending=True, plot=["rfdiffusion_plddt", "rfdiffusion_template_bb_ca_motif_rmsd", "rfdiffusion_peratom_ligand_contacts", "rfdiffusion_pocket_score", "rfdiffusion_pocket_score_v2", "rfdiffusion_pocket_score_v3"])

    # remove structures with template RMSDs > 1
    ensembles.poses_df = ensembles.poses_df[ensembles.poses_df["rfdiffusion_template_bb_ca_motif_rmsd"] <= 1]

    ######################## MPNN-FASTDesign-MPNN #################################################
    # cycle MPNN and FastRelax:
    index_layers=1
    pdb_loc_col = "postdiffusion_chainremoval_location" 
    fr_mpnn_rmsd_traj = PlottingTrajectory(y_label="RMSD [\u00C5]", location=f"{plot_dir}/fr_mpnn_rmsd_trajectory.png", title="Motif BB-Ca\nTrajectory", dims=(0,3))
    for i in range(1):
        cycle_prefix = f"cycle_{str(i)}"

        # run mpnn and fr
        ensembles, fr_pdb_dir = mpnn_fr(ensembles, prefix=f"cycle_{str(i)}", fastrelax_pose_opts="fr_pose_opts", pdb_location_col=pdb_loc_col, reference_location_col="updated_reference_frags_location", params_file=params_file)

        # plot
        fr_mpnn_rmsd_traj.add_and_plot(ensembles.poses_df[f"cycle_{str(i)}_fr_bb_ca_motif_rmsd"], f"cycle_{str(i)}")

        # setup next cycle
        pdb_loc_col = f"{cycle_prefix}_fr_location"

    # superimpose poses on reference frags and calculate ligand scores:
    if keep_ligand_chain: calc_ligand_stats(input_df=ensembles.poses_df, ref_frags_col="updated_reference_frags_location", ref_motif_col="motif_residues", poses_motif_col="motif_residues", prefix="pre_esm", ligand_chain=args.ligand_chain)

    # filter down by total_score to max_esm_inputs:
    pre_mpnn_comp_score = ensembles.calc_composite_score("pre_esm_comp_score", [f"{cycle_prefix}_fr_total_score", f"pre_esm_pocket_score_v2", f"pre_esm_peratom_ligand_contacts", f"rfdiffusion_rog"], [1, -1, 1, 1])
    total_score_filter = ensembles.filter_poses_by_score(args.num_mpnn_inputs, f"pre_esm_comp_score", prefix=f"pre_mpnn_filter", plot=["pre_esm_peratom_ligand_contacts", f"{cycle_prefix}_fr_total_score", f"{cycle_prefix}_fr_bb_ca_motif_rmsd", f"pre_esm_pocket_score", f"pre_esm_pocket_score_v2", f"pre_esm_pocket_score_v3", f"rfdiffusion_rog"])

    ######################### ROUND1 ESMFold ###############################################
    # run mpnn and predict with ESMFold:
    ensembles = mpnn_design_and_esmfold(ensembles, prefix="round1", num_mpnn_seqs=args.num_mpnn_seqs, num_esm_inputs=args.num_esm_inputs, num_esm_outputs_per_input_backbone=args.num_esm_outputs_per_input_backbone, bb_rmsd_col=f"{cycle_prefix}_fr_location", motif_ref_pdb_col="updated_reference_frags_location", disfavor_alanines=1, calc_perplexity=True)

    # superimpose poses on reference frags and calculate ligand scores:
    if keep_ligand_chain: 
        post_esm_lig_poses = ensembles.add_ligand_from_ref(ref_col="updated_reference_frags_location", ref_motif="motif_residues", target_motif="motif_residues", lig_chain=args.ligand_chain, prefix=f"post_esm_lig_poses")
        calc_ligand_stats(input_df=ensembles.poses_df, ref_frags_col="updated_reference_frags_location", ref_motif_col="motif_residues", poses_motif_col="motif_residues", prefix="post_esm", ligand_chain=args.ligand_chain)

    # remove poses that have ligand clashes:
    print(f"Removing {len(ensembles.poses_df[ensembles.poses_df['post_esm_ligand_clash'] == True])} poses from poses_df because of Ligand Clashes." )
    ensembles.poses_df = ensembles.poses_df[ensembles.poses_df["post_esm_ligand_clash"] == False]
    ensembles.poses_df["post_esm_rog"] = [utils.metrics.calc_rog_of_pdb(pose) for pose in ensembles.poses_df["poses"]]

    # Filter down to final set of .pdbs that will be input for Rosetta Refinement: Filter based on pLDDT, ligand contacts (pocket), protein ROG and motif RMSD.
    out_filterscore = ensembles.calc_composite_score("out_filter_comp_score", (st := [f"round1_esm_plddt", f"round1_esm_bb_ca_motif_rmsd", f"post_esm_rog", f"post_esm_pocket_score_v2", f"post_esm_peratom_ligand_contacts"]), [-0.75,1.5,0.2,-0.3,0.4])
    out_filter = ensembles.filter_poses_by_score(args.num_refinement_inputs, f"out_filter_comp_score", prefix="out_filter", plot=st)
    results_dir = f"{args.output_dir}/intermediate_results/"
    ref_frag_dir = f"{results_dir}/ref_fragments/"
    if not os.path.isdir(ref_frag_dir): os.makedirs(ref_frag_dir, exist_ok=True)
    ensembles.dump_poses(results_dir)

    # Copy and rewrite Fragments into output_dir/reference_fragments
    updated_ref_pdbs = update_and_copy_reference_frags(ensembles.poses_df, ref_col="input_poses", desc_col="poses_description", motif_prefix="rfdiffusion", out_pdb_path=ref_frag_dir, keep_ligand_chain=args.ligand_chain)

    # Write PyMol Alignment Script
    ref_originals = [shutil.copy(ref_pose, f"{results_dir}/") for ref_pose in ensembles.poses_df["input_poses"].to_list()]
    pymol_script = utils.pymol_tools.write_pymol_alignment_script(ensembles.poses_df, scoreterm="out_filter_comp_score", top_n=args.num_refinement_inputs, path_to_script=f"{results_dir}/align.pml")

    # write Rosetta Pose Options to a .json file:
    ros_pose_opts = write_rosetta_pose_opts_to_json(ensembles.poses_df, path_to_json_file=f"{results_dir}/rosetta_pose_opts.json")
    dumped = ensembles.dump_poses(f"{results_dir}/pdb_in/")
    out_df = ensembles.poses_df[["poses_description", "fixed_residues", "motif_residues"]].set_index("poses_description")
    out_df.to_json(f"{results_dir}/motif_res.json")

    if args.diffuse_only.lower() == "true": sys.exit(1)

    #################### REFINEMENT ######################################
    # initial number of refinement runs is higher:
    fr_n = 25

    # instantiate plotting trajectories:
    esm_plddt_traj = PlottingTrajectory(y_label="ESMFold pLDDT", location=f"{plot_dir}/esm_plddt_trajectory.png", title="ESMFold Trajectory", dims=(0,100))
    esm_bb_ca_rmsd_traj = PlottingTrajectory(y_label="RMSD [\u00C5]", location=f"{plot_dir}/esm_bb_ca_trajectory.png", title="ESMFold BB-Ca\nRMSD Trajectory", dims=(0,10))
    esm_motif_ca_rmsd_traj = PlottingTrajectory(y_label="RMSD [\u00C5]", location=f"{plot_dir}/esm_motif_ca_trajectory.png", title="ESMFold Motif-Ca\nRMSD Trajectory", dims=(0,8))
    esm_catres_rmsd_traj = PlottingTrajectory(y_label="RMSD [\u00C5]", location=f"{plot_dir}/esm_catres_rmsd_trajectory.png", title="ESMFold Motif\nSidechain RMSD Trajectory", dims=(0,8))
    refinement_total_score_traj = PlottingTrajectory(y_label="Rosetta total score [REU]", location=f"{plot_dir}/rosetta_total_score_trajectory.png", title="FastDesign Total Score Trajectory")
    refinement_motif_ca_rmsd_traj = PlottingTrajectory(y_label="RMSD [\u00C5]", location=f"{plot_dir}/refinement_motif_rmsd_trajectory.png", title="Refinement Motif\nBB-Ca RMSD Trajectory", dims=(0,8))

    # cycle fastrelax, proteinmpnn and ESMFold
    filter_layers = 2
    index_layers_to_remove = 2
    idx = index_layers
    
    # add back the ligand
    lig_poses = ensembles.add_ligand_from_ref(ref_col="updated_reference_frags_location", ref_motif="motif_residues", target_motif="motif_residues", lig_chain=args.ligand_chain, prefix=f"pre_refinement_lig_poses")
    
    for i in range(args.refinement_cycles):
        c_pref = f"refinement_cycle_{str(i).zfill(2)}"

        # calculate MPNN probabilities and write resfiles
        ensembles.poses_df[f"{c_pref}_mpnn_fixed_residues"] = [get_design_residues(row, motif_res_col="motif_residues", cat_res_col="fixed_residues", lig_chain=args.ligand_chain) for index, row in ensembles.poses_df.iterrows()]
        probs = ensembles.get_mpnn_probs(f"{c_pref}_probs")
        resfiles = ensembles.write_resfiles_from_mpnn_probs(colname=f"{c_pref}_resfiles", probabilities_location_col=f"{c_pref}_probs", motif_col=f"{c_pref}_mpnn_fixed_residues", motif_chain="A", motif_threshold=0.05)

        # refine
        ensembles.poses_df["fastdesign_opts"] = [write_fastdesign_opts(row, cycle=i, total_cycles=args.refinement_cycles, reference_location_col="updated_reference_frags_location", motif_res_col="motif_residues", cat_res_col="fixed_residues", designres_col=f"{c_pref}_mpnn_fixed_residues", resfile_col=f"{c_pref}_resfiles") for index, row in ensembles.poses_df.iterrows()]
        #ensembles.poses_df["refinement_opts"] = ensembles.poses_df["fr_pose_opts"].str.replace(" -parser:script_vars ", f" -parser:script_vars sd={str(0.5 + i)} ")
        ensembles = fr_mpnn_esmfold(ensembles, prefix=c_pref, n=fr_n, fastrelax_pose_opts="fastdesign_opts", ref_pdb_col="updated_reference_frags_location", mpnn_fixedres_col=f"{c_pref}_mpnn_fixed_residues", use_soluble_model=True, params_file=params_file)

        # add back the ligand
        lig_poses = ensembles.add_ligand_from_ref(ref_col="updated_reference_frags_location", ref_motif="motif_residues", target_motif="motif_residues", lig_chain=args.ligand_chain, prefix=f"{c_pref}_lig_poses")

        # remove outputs that have ligand clashes:
        if args.high_resolution_clash_detection.lower() == "true":
            print(f"Running high resolution ligand clash detection.")
            fl = len(ensembles.poses_df)
            ensembles = clash_detection(ensembles, ref_frags_col="updated_reference_frags_location", ref_motif_col="motif_residues", poses_motif_col="motif_residues", prefix=f"{c_pref}_highres", ligand_chain=args.ligand_chain, database_dir="database", bb_clash_vdw_multiplier=args.bb_clash_vdw_multiplier, save_path_list=None)
            ensembles.poses_df = ensembles.poses_df[ensembles.poses_df[f"{c_pref}_highres_ligand_clash"] == False]
            print(f"Removed {fl - len(ensembles.poses_df)} of {fl} poses from poses because of ligand clashes")
        else:
            print(f"Running low resolution ligand clash detection with detection radius {args.refinement_ligand_clash_dist} Angstrom.")
            ensembles.poses_df[f"{c_pref}_ligand_clash"] = [utils.metrics.check_for_ligand_clash_of_pdb(pdb_path=pose, ligand_chain=args.ligand_chain, dist=args.refinement_ligand_clash_dist) for pose in ensembles.poses_df["poses"].to_list()]
            fl = len(ensembles.poses_df)
            ensembles.poses_df = ensembles.poses_df[ensembles.poses_df[f"{c_pref}_ligand_clash"] == False]
            print(f"Removed {fl - len(ensembles.poses_df)} of {fl} poses from poses because of ligand clashes")

        # plot
        esm_plddt_traj.add_and_plot(ensembles.poses_df[f"{c_pref}_esm_plddt"], c_pref)
        esm_bb_ca_rmsd_traj.add_and_plot(ensembles.poses_df[f"{c_pref}_esm_bb_ca_rmsd"], c_pref)
        esm_motif_ca_rmsd_traj.add_and_plot(ensembles.poses_df[f"{c_pref}_esm_bb_ca_motif_rmsd"], c_pref)
        esm_catres_rmsd_traj.add_and_plot(ensembles.poses_df[f"{c_pref}_esm_catres_motif_heavy_rmsd"], c_pref)
        refinement_total_score_traj.add_and_plot(ensembles.poses_df[f"{c_pref}_refinement_total_score"], c_pref)
        refinement_motif_ca_rmsd_traj.add_and_plot(ensembles.poses_df[f"{c_pref}_refinement_bb_ca_motif_rmsd"], c_pref)

        #filter down for next cycle, or if in last cycle, filter for AF2 prediction input:
        if i+1 < args.refinement_cycles:
            cycle_filter = ensembles.filter_poses_by_score(5, f"{c_pref}_esm_comp_score", prefix=f"{c_pref}_final_filter", remove_layers=filter_layers, plot=(fst := [f"{c_pref}_esm_comp_score", f"{c_pref}_esm_plddt", f"{c_pref}_esm_bb_ca_rmsd", f"{c_pref}_esm_bb_ca_motif_rmsd", f"{c_pref}_esm_catres_motif_heavy_rmsd"]))
        else:
            cycle_filter = ensembles.filter_poses_by_score(25, f"{c_pref}_esm_comp_score", prefix=f"{c_pref}_final_filter", remove_layers=filter_layers, plot=(fst := [f"{c_pref}_esm_comp_score", f"{c_pref}_esm_plddt", f"{c_pref}_esm_bb_ca_rmsd", f"{c_pref}_esm_bb_ca_motif_rmsd", f"{c_pref}_esm_catres_motif_heavy_rmsd"]))
        
        # reindex poses
        index_removed = ensembles.reindex_poses(out_dir=f"{c_pref}_reindexed_poses", remove_layers=index_layers_to_remove)
        fr_n = 5
        filter_layers = 3
        index_layers_to_remove = 3
        index_layers = idx+1
    
    ############################################ POST REFINEMENT WORKUP ############################################################
    # repredict with AlphaFold2 and calculate RMSDs
    af2_preds = ensembles.predict_sequences(run_AlphaFold2, options="--msa-mode single_sequence ", prefix="af2")
    af2_bb_ca_rmsd = ensembles.calc_bb_rmsd_df(ref_pdb=f"{c_pref}_refinement_location", metric_prefix="af2")
    af2_motif_ca_rmsd = ensembles.calc_motif_bb_rmsd_df(ref_pdb="updated_reference_frags_location", ref_motif="motif_residues", target_motif="motif_residues", metric_prefix="af2_bb_ca")
    af2_catres_rmsd = ensembles.calc_motif_heavy_rmsd_df(ref_pdb="updated_reference_frags_location", ref_motif="fixed_residues", target_motif="fixed_residues", metric_prefix="af2_catres")

    # calculate average sidechain RMSD:
    sc_rmsd_opts = "-parser:protocol /home/mabr3112/riff_diff/rosetta/fastrelax_rmsdcheck.xml -beta"
    ensembles = calculate_fastrelax_sidechain_rmsd(ensembles, prefix="post_refinement_rmsdcheck", options=sc_rmsd_opts, sidechain_residues="fixed_residues", sidechain_ref_pdb_col="updated_reference_frags_location", n=15, pose_options=None)

    # check calculate catres RMSD:
    check_rmsd = ensembles.calc_motif_heavy_rmsd_df(ref_pdb="updated_reference_frags_location", ref_motif="fixed_residues", target_motif="fixed_residues", metric_prefix="check_postrelax_catres")

    # plot af2_stats:
    cols = [f"af2_top_plddt", "af2_mean_plddt", "af2_bb_ca_rmsd", "af2_bb_ca_motif_rmsd", "af2_catres_motif_heavy_rmsd", "post_refinement_rmsdcheck_mean_sidechain_motif_heavy_rmsd", "post_refinement_rmsdcheck_sidechain_motif_heavy_rmsd"]
    titles = ["Top AF2-pLDDT", "Mean AF2-pLDDT", "AF2 BB-Ca RMSD", "AF2 Motif-Ca RMSD", "AF2 Catres\nSidechain RMSD", "Relax Mean\nSidechain RMSD", "Relax Min.\n Sidechain RMSD"]
    y_labels = ["pLDDT", "pLDDT", "RMSD [\u00C5]", "RMSD [\u00C5]", "RMSD [\u00C5]", "RMSD [\u00C5]", "RMSD [\u00C5]"]
    dims = [(0,100), (0,100), (0,5), (0,5), (0,5), (0,5), (0,5)]
    _ = plots.violinplot_multiple_cols(ensembles.poses_df, cols=cols, titles=titles, y_labels=y_labels, dims=dims, out_path=f"{plot_dir}/af2_stats.png")
    
    # superimpose poses on reference frags and calculate ligand scores:
    if keep_ligand_chain: 
        # add back the ligand and calculate lig scores:
        lig_poses = ensembles.add_ligand_from_ref(ref_col="updated_reference_frags_location", ref_motif="motif_residues", target_motif="motif_residues", lig_chain=args.ligand_chain, prefix=f"final_redesign_lig_poses")
        calc_ligand_stats(input_df=ensembles.poses_df, ref_frags_col="updated_reference_frags_location", ref_motif_col="motif_residues", poses_motif_col="motif_residues", prefix="post_refinement", ligand_chain=args.ligand_chain)

    # Ligand added back in, now run GALigandDock:
    #docking_options = f"-parser:protocol {args.docking_protocol} -parser:script_vars ligchain={args.ligand_chain}"
    #docked_poses = rosetta_scripts_and_mean(ensembles, prefix="final_dock", n=15, options=docking_options, pose_options=None, filter_scoreterm="final_dock_dG", scoreterms=None)

    # remove any structures that have an AF2 pLDDT below 85, Ca RMSD > 1
    #ensembles.poses_df = ensembles.poses_df[(ensembles.poses_df["af2_top_plddt"] <= 85) & (ensembles.poses_df["af2_bb_ca_rmsd"] <= 1) & (ensembles.poses_df["af2_bb_ca_motif_rmsd"] <= 1.5)]
    
    # store poses before final downsampling
    #os.makedirs((docked_poses_dir := f"{ensembles.dir}/docked_poses/"), exist_ok=True)
    #ensembles.dump_poses(docked_poses_dir)
    #ensembles.poses_df.to_json(f"{docked_poses_dir}/docked_scores.json")


    # calc perres Rosetta stats
    ensembles.poses_df["post_refinement_rmsdcheck_total_score_perres"] = ensembles.poses_df["post_refinement_rmsdcheck_fr_total_score"] / ensembles.poses_df["post_refinement_rmsdcheck_fr_all_selection_count"]
    ensembles.poses_df["perres_core_fa_atr"] = ensembles.poses_df["post_refinement_rmsdcheck_fr_total_energy"] / ensembles.poses_df["post_refinement_rmsdcheck_fr_core_selection_count"]
    ensembles.poses_df["perres_contacts"] = ensembles.poses_df["post_refinement_rmsdcheck_fr_contacts"] / ensembles.poses_df["post_refinement_rmsdcheck_fr_all_selection_count"]
    ensembles.poses_df["perres_sap"] = ensembles.poses_df["post_refinement_rmsdcheck_fr_sap_score"] / ensembles.poses_df["post_refinement_rmsdcheck_fr_all_selection_count"]

    # plot Rosetta stats:
    cols_r = ["post_refinement_rmsdcheck_total_score_perres", "perres_core_fa_atr", "perres_contacts", "perres_sap"]
    titles_r = ["Total Score", "Core Stability", "Atomic Density", "SAP Score"]
    y_labels_r = ["[REU] / residue", "fa_atr [REU] / residue", "count", "SAP / residue"]
    dims_r = [(-5, 0), (-12, 0), (0, 5), (0, 1.5)]
    plots.violinplot_multiple_cols(ensembles.poses_df, cols=cols_r, titles=titles_r, y_labels=y_labels_r, dims=dims_r, out_path=f"{plot_dir}/rosetta_final_stats.png")
  
    # final backbone downsampling
    final_downsampling_score = ensembles.calc_composite_score(f"final_downsampling_comp_score", [f"post_refinement_rmsdcheck_mean_sidechain_motif_heavy_rmsd", f"af2_bb_ca_motif_rmsd", f"af2_mean_plddt", f"post_refinement_rmsdcheck_fr_sap_score"], [1, 0.25, -0.25, 0.25])
    final_downsampling = ensembles.filter_poses_by_score(1, f"final_downsampling_comp_score", prefix=f"output_filter", remove_layers=3, plot=[f"final_downsampling_comp_score", f"post_refinement_rmsdcheck_mean_sidechain_motif_heavy_rmsd", "af2_bb_ca_rmsd", "af2_mean_plddt", "post_refinement_rmsdcheck_fr_sap_score"])

    # make new results, copy fragments and write alignment_script
    results_dir = f"{args.output_dir}/results/"
    ref_frag_dir = f"{results_dir}/ref_fragments/"
    if not os.path.isdir(ref_frag_dir): os.makedirs(ref_frag_dir, exist_ok=True)
    ensembles.dump_poses(results_dir)
    _ = plots.violinplot_multiple_cols(ensembles.poses_df, cols=cols, titles=titles, y_labels=y_labels, dims=dims, out_path=f"{plot_dir}/af2_final_stats.png")
    _ = plots.violinplot_multiple_cols(ensembles.poses_df, cols=cols, titles=titles, y_labels=y_labels, dims=dims, out_path=f"{results_dir}/af2_final_stats.png")
    _ = plots.violinplot_multiple_cols(ensembles.poses_df, cols=cols_r, titles=titles_r, y_labels=y_labels_r, dims=dims_r, out_path=f"{results_dir}/rosetta_final_stats.png")

    # Copy and rewrite Fragments into output_dir/reference_fragments
    updated_ref_pdbs = update_and_copy_reference_frags(ensembles.poses_df, ref_col="input_poses", desc_col="poses_description", motif_prefix="rfdiffusion", out_pdb_path=ref_frag_dir, keep_ligand_chain=args.ligand_chain)

    # Write PyMol Alignment Script
    ref_originals = [shutil.copy(ref_pose, f"{results_dir}/") for ref_pose in ensembles.poses_df["input_poses"].to_list()]
    pymol_script = utils.pymol_tools.write_pymol_alignment_script(ensembles.poses_df, scoreterm=f"final_downsampling_comp_score", top_n=len(ensembles.poses_df), path_to_script=f"{results_dir}/align.pml")

    # write csv file for coupled-moves
    csv_df = pd.DataFrame(ensembles.poses_df.sort_values(by="final_downsampling_comp_score")["poses_description"])
    csv_df["mutations"] = ["" for i in list(csv_df["poses_description"])]
    csv_df.to_csv(f"{results_dir}/coupled_moves_input_selection.csv")

    print("done")


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--input_dir", type=str, required=True, help="input_directory that contains all ensemble *.pdb files to be hallucinated (max 1000 files).")
    argparser.add_argument("--output_dir", type=str, required=True, help="output_directory")
    argparser.add_argument("--fastrelax_protocol", type=str, default="/home/mabr3112/riff_diff/rosetta/mpnn_fastdesign_substrate.xml", help="Protocol of fastrelax-MPNN cycles")
    argparser.add_argument("--refinement_protocol", type=str, default="/home/mabr3112/riff_diff/rosetta/fd_prob_refine_dev.xml")
    argparser.add_argument("--refinement_cycles", type=int, default=5, help="Number of Fastrelax-mpnn-esmfold refinement cycles to run.")
    argparser.add_argument("--diffuse_only", type=str, default="False", help="Set to 'True' if you want only to run RFdiffusion and no refinement!")
    argparser.add_argument("--channel_contig", type=str, default="Q5-16", help="Contig string of the channel protein")

    # rfdiffusion options
    argparser.add_argument("--num_rfdiffusions", type=int, default=10, help="Number of rfdiffusion trajectories.")
    argparser.add_argument("--rfdiffusion_timesteps", type=int, default=50, help="Number of RFdiffusion timesteps to diffuse.")
    argparser.add_argument("--max_rfdiffusion_gpus", type=int, default=10, help="On how many GPUs at a time to you want to run Hallucination?")
    argparser.add_argument("--rfdiffusion_additional_options", type=str, default="", help="Any additional options that you want to parse to RFdiffusion.")
    argparser.add_argument("--rfdiff_guide_scale", type=int, default=5, help="Guide_scale value for RFDiffusion")
    argparser.add_argument("--pos_weight", type=float, default=16, help="Attractive substrate weight")
    argparser.add_argument("--pot_weight", type=float, default=4, help="weight of the potential")
    argparser.add_argument("--guide_decay", type=str, default="quadratic", help="potential decay for RFdiffusion")
    argparser.add_argument("--attr_dist", type=float, default=0, help="weight of the potential")
    argparser.add_argument("--decentralize", type=float, default=2, help="Set this value higher if you want your substrate more buried.")
    argparser.add_argument("--custom_diffusion_center", type=str, default="False", help="Do you want to use a custom center for diffusion?")

    # rosetta
    argparser.add_argument("--params_file", type=str, default=None, help="Path to a custom params file, if you want Rosetta to use your params file.")

    # linkers
    argparser.add_argument("--flanking", type=str, default="split", help="Overwrites contig output of 'run_ensemble_evaluator.py'. Can be either 'split', 'nterm', 'cterm'")
    argparser.add_argument("--total_flanker_length", type=int, default=40, help="Overwrites contig output of 'run_ensemble_evaluator.py'. Set the max length of the pdb-file that is being hallucinated. Will only be used in combination with 'flanking'")
    argparser.add_argument("--overwrite_linker_lengths", type=str, default='50,200', help="specify first total length of the protein, then maximum length that should be assigned to the linkers.\nExample: --overwrite_linker_lengths='50,200' -> would set the maximum length of the protein to 200 and would overwrite the linkers to ranges between ~5-75.")

    # mpnn options
    argparser.add_argument("--num_mpnn_inputs", type=int, default=300, help="Number of input backbones to ProteinMPNN before predicting them with ESMFold")
    argparser.add_argument("--num_mpnn_seqs", type=int, default=60, help="Number of MPNN Sequences to generate for each input backbone.")
    argparser.add_argument("--num_esm_inputs", type=int, default=12, help="Number of MPNN Sequences for each input backbone that should be predicted. Typically quarter to half of the sequences generated by MPNN is a good value.")
    argparser.add_argument("--num_esm_outputs_per_input_backbone", type=int, default=1, help="Number of ESM Outputs for each backbone that is inputted to ESMFold.")

    # output options
    argparser.add_argument("--num_refinement_inputs", type=int, default=25, help="Number of .pdb files that will be stored into the final output directory.")
    argparser.add_argument("--output_scoreterms", type=str, default="esm_plddt,esm_bb_ca_motif_rmsd", help="Scoreterms to use to filter ESMFolded PDBs to the final output pdbs. IMPORTANT: if you supply scoreterms, also supply weights and always check the filter output plots in the plots/ directory!")
    argparser.add_argument("--output_scoreterm_weights", type=str, default="-1,1.5", help="Weights for how to combine the scoreterms listed in '--output_scoreterms'")
    argparser.add_argument("--ligand_chain", type=str, default="Z", help="Chain name of your ligand chain.")
    argparser.add_argument("--num_outputs", type=int, default=20, help="Number of .pdb-files you would like to have as output.")
    argparser.add_argument("--filter_results_by_backbone", type=bool, default=True, help="Output only one structure per refinement input backbone.")

    # refinement opts:
    argparser.add_argument("--high_resolution_clash_detection", type=str, default="False", help="Run Adrian's high resolution ligand clash detection (calculates VdW radii)")
    argparser.add_argument("--bb_clash_vdw_multiplier", default=0.7, type=float, help="vdw multiplier for bb clash detection during refinement.")
    argparser.add_argument("--refinement_ligand_clash_dist", type=float, default=1.5, help="Default distance to calculate ligand clashes during refinement for vanilla clash detection")

    # docking
    argparser.add_argument("--docking_protocol", type=str, default="/home/mabr3112/riff_diff/rosetta/GA_dock.xml", help="RosettaScript that executes ligand docking.")
    args = argparser.parse_args()

    main(args)
