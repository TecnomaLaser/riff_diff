#!/home/tripp/mambaforge/envs/protflow_new/bin/python
'''
Script to run RFdiffusion active-site model on artificial motif libraries.
'''
#imports
import json
import logging
import os
import re
import sys
import glob
import copy
import itertools
import shutil

# dependency
import numpy as np
import pandas as pd
import matplotlib
import protflow
import protflow.config
from protflow.jobstarters import SbatchArrayJobstarter
import protflow.poses
import protflow.residues
import protflow.tools
import protflow.tools.colabfold
import protflow.tools.esmfold
import protflow.tools.ligandmpnn
import protflow.tools.attnpacker
import protflow.metrics.rmsd
import protflow.metrics.tmscore
import protflow.metrics.fpocket
import protflow.tools.protein_edits
import protflow.tools.rfdiffusion
from protflow.metrics.generic_metric_runner import GenericMetric
from protflow.metrics.ligand import LigandClashes, LigandContacts
from protflow.metrics.rmsd import BackboneRMSD, MotifRMSD, MotifSeparateSuperpositionRMSD
import protflow.tools.rosetta
from protflow.utils.biopython_tools import renumber_pdb_by_residue_mapping, load_structure_from_pdbfile, save_structure_to_pdbfile
import protflow.utils.plotting as plots
#from protflow.utils.metrics import calc_rog_of_pdb

# custom

# local
sys.path.append("/home/mabr3112/riff_diff/")
from utils.pymol_tools_protflow import write_pymol_alignment_script


if __file__.startswith("/home/mabr3112"):
    matplotlib.use('Agg')
else:
    print("Using Matplotlib without 'Agg' backend.")

def divide_flanking_residues(residual: int, flanking: str) -> tuple:
    '''Splits up flanking residues. This function is a relic of past times.'''
    def split_flankers(residual) -> tuple:
        ''''''
        cterm = residual // 2
        nterm = residual - cterm
        return nterm, cterm

    residual = int(residual)
    if residual < 6 or flanking == "split":
        return split_flankers(residual)
    elif flanking == "nterm":
        return residual-3, 3
    elif flanking == "cterm":
        return 3, residual-3
    else:
        raise ValueError(f"Paramter <flanking> can only be 'split', 'nterm', or 'cterm'. flanking: {flanking}")

def adjust_flanking(rfdiffusion_pose_opts: str, flanking_type: str, total_flanker_length:int=None) -> str:
    '''adjusts length of flanking residues in contig. Another relic of long gone times.'''
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

def update_and_copy_reference_frags(input_df: pd.DataFrame, ref_col:str, desc_col:str, prefix: str, out_pdb_path=None, keep_ligand_chain:str="") -> "list[str]":
    '''Updates reference fragments (input_pdbs) to the motifs that were set during diffusion.'''
    # create residue mappings {old: new} for renaming
    list_of_mappings = [protflow.tools.rfdiffusion.get_residue_mapping(ref_motif, inp_motif) for ref_motif, inp_motif in zip(input_df[f"{prefix}_con_ref_pdb_idx"].to_list(), input_df[f"{prefix}_con_hal_pdb_idx"].to_list())]

    # compile list of output filenames
    output_pdb_names_list = [f"{out_pdb_path}/{desc}.pdb" for desc in input_df[desc_col].to_list()]

    # renumber
    return [renumber_pdb_by_residue_mapping(ref_frag, res_mapping, out_pdb_path=pdb_output, keep_chain=keep_ligand_chain) for ref_frag, res_mapping, pdb_output in zip(input_df[ref_col].to_list(), list_of_mappings, output_pdb_names_list)]

def active_site_pose_opts(input_opt: str, motif: protflow.residues.ResidueSelection, as_model_path: str) -> str:
    '''Converts rfdiffusion_pose_opts string from default model to pose_opts string for active_site model (removes inpaint_seq and stuff.)'''
    def re_split_rfdiffusion_opts(command: str) -> list:
        if command is None:
            return []
        return re.split(r"\s+(?=(?:[^']*'[^']*')*[^']*$)", command)
    # split args in rfdiffusion string and remove inpaint_seq
    opts_l = [x for x in re_split_rfdiffusion_opts(input_opt) if "inpaint_seq" not in x]
    contig = opts_l[0]

    # change linker minimum lengths:
    contig = replace_number_with_10(contig)

    # exchange fixed residues in contig:
    for (chain, res) in motif.residues:
        contig = contig.replace(f"{chain}1-7", f"{chain}{res}-{res}")

    # remerge contig into opts_l and return concatenated opts:
    opts_l[0] = contig
    opts_l.append(f"inference.ckpt_override_path={as_model_path}")
    return " ".join(opts_l)

def replace_number_with_10(input_string):
    '''Replaces minimum linker length in rfdiffusion contig (from x-50 to 10-50)'''
    # This regex matches any sequence of digits followed by '-50'
    pattern = r'\d+-50'
    # Replace found patterns with '10-50'
    return re.sub(pattern, '10-50', input_string)

def adjust_linker_length(motif_residues: dict, total_length: int, flanker_length: int, pose_opts: str):
    num_linkers = len(motif_residues) - 1
    total_motif_len = sum(len(motif_res) for motif_res in motif_residues)
    linker_length = int(3 + (total_length - flanker_length - total_motif_len) / num_linkers) # 3 is added to conserve original 50/200 linker_length/total_length ratio in case of 4 AS residues with 7 res fragments
    adjusted_pose_opts = overwrite_linker_length(pose_opts, total_length, linker_length)
    return adjusted_pose_opts

def instantiate_trajectory_plotting(plot_dir, df):
    # instantiate plotting trajectories:
    esm_plddt_traj = plots.PlottingTrajectory(y_label="pLDDT", location=os.path.join(plot_dir, "trajectory_plddt.png"), title="pLDDT Trajectory", dims=(0,100))
    esm_plddt_traj.add_and_plot(df["esm_plddt"], "screening")
    esm_bb_ca_rmsd_traj = plots.PlottingTrajectory(y_label="RMSD [\u00C5]", location=os.path.join(plot_dir, "trajectory_esm_bb_ca.png"), title="ESMFold BB-Ca\nRMSD Trajectory", dims=(0,5))
    esm_bb_ca_rmsd_traj.add_and_plot(df["esm_backbone_rmsd"], "screening")
    esm_motif_ca_rmsd_traj = plots.PlottingTrajectory(y_label="RMSD [\u00C5]", location=os.path.join(plot_dir, "trajectory_esm_fixedres_ca.png"), title="ESMFold Fixed Residues\nCa RMSD Trajectory", dims=(0,5))
    esm_motif_ca_rmsd_traj.add_and_plot(df["esm_catres_bb_rmsd"], "screening")
    esm_catres_rmsd_traj = plots.PlottingTrajectory(y_label="RMSD [\u00C5]", location=os.path.join(plot_dir, "trajectory_esm_catres_rmsd.png"), title="ESMFold Fixed Residues\nSidechain RMSD Trajectory", dims=(0,5))
    esm_catres_rmsd_traj.add_and_plot(df["esm_catres_heavy_rmsd"], "screening")
    fastrelax_total_score_traj = plots.PlottingTrajectory(y_label="Rosetta total score [REU]", location=os.path.join(plot_dir, "trajectory_rosetta_total_score.png"), title="FastRelax Total Score Trajectory")
    postrelax_motif_ca_rmsd_traj = plots.PlottingTrajectory(y_label="RMSD [\u00C5]", location=os.path.join(plot_dir, "trajectory_postrelax_fixedres_rmsd.png"), title="Postrelax Fixed Residues\nCa RMSD Trajectory", dims=(0,5))
    postrelax_motif_catres_rmsd_traj = plots.PlottingTrajectory(y_label="RMSD [\u00C5]", location=os.path.join(plot_dir, "trajectory_postrelax_fixedres_catres.png"), title="Postrelax Fixed Residues\nSidechain RMSD Trajectory", dims=(0,5))
    delta_apo_holo_traj = plots.PlottingTrajectory(y_label="Rosetta delta total score [REU]", location=os.path.join(plot_dir, "trajectory_delta_apo_holo.png"), title="Delta Apo Holo Total Score Trajectory")
    postrelax_ligand_rmsd_traj = plots.PlottingTrajectory(y_label="RMSD [\u00C5]", location=os.path.join(plot_dir, "trajectory_postrelax_ligand_rmsd.png"), title="Postrelax Ligand\nRMSD Trajectory", dims=(0,5))
    return {'esm_plddt': esm_plddt_traj, 'esm_backbone_rmsd': esm_bb_ca_rmsd_traj, 'esm_catres_bb_rmsd': esm_motif_ca_rmsd_traj, 'esm_catres_heavy_rmsd': esm_catres_rmsd_traj, 'fastrelax_total_score': fastrelax_total_score_traj, 'postrelax_catres_heavy_rmsd': postrelax_motif_catres_rmsd_traj, 'postrelax_catres_bb_rmsd': postrelax_motif_ca_rmsd_traj, 'delta_apo_holo': delta_apo_holo_traj, 'postrelax_ligand_rmsd': postrelax_ligand_rmsd_traj}

def update_trajectory_plotting(trajectory_plots:dict, df:pd.DataFrame, cycle:int):
    for traj in trajectory_plots:
        trajectory_plots[traj].add_and_plot(df[f"cycle_{cycle}_{traj}"], f"cycle_{cycle}")
    return trajectory_plots

def add_final_data_to_trajectory_plots(df: pd.DataFrame, trajectory_plots):
    trajectory_plots['esm_plddt'].add_and_plot(df['final_AF2_plddt'], "eval (AF2)")
    trajectory_plots['esm_backbone_rmsd'].add_and_plot(df['final_AF2_backbone_rmsd'], "eval (AF2)")
    trajectory_plots['esm_catres_bb_rmsd'].add_and_plot(df['final_AF2_catres_bb_rmsd'], "eval (AF2)")
    trajectory_plots['esm_catres_heavy_rmsd'].add_and_plot(df['final_AF2_catres_heavy_rmsd'], "eval (AF2)")
    trajectory_plots['fastrelax_total_score'].add_and_plot(df['final_fastrelax_total_score'], "eval (AF2)")
    trajectory_plots['postrelax_catres_heavy_rmsd'].add_and_plot(df['final_postrelax_catres_heavy_rmsd'], "eval (AF2)")
    trajectory_plots['postrelax_catres_bb_rmsd'].add_and_plot(df['final_postrelax_catres_bb_rmsd'], "eval (AF2)")
    trajectory_plots['delta_apo_holo'].add_and_plot(df['final_delta_apo_holo'], "eval (AF2)")
    trajectory_plots['postrelax_ligand_rmsd'].add_and_plot(df['final_postrelax_ligand_rmsd'], "eval (AF2)")
    return trajectory_plots

def create_ref_results_dir(poses, dir:str, cycle:int):
    # plot outputs and write alignment script
    logging.info(f"Creating refinement output directory for refinement cycle {cycle} at {dir}...")
    os.makedirs(dir, exist_ok=True)

    logging.info(f"Plotting outputs of cycle {cycle}.")
    cols = [f"cycle_{cycle}_esm_plddt", f"cycle_{cycle}_esm_backbone_rmsd", f"cycle_{cycle}_esm_catres_heavy_rmsd", f"cycle_{cycle}_fastrelax_total_score", f"cycle_{cycle}_postrelax_ligand_rmsd"]
    titles = ["ESMFold pLDDT", "ESMFold BB-Ca RMSD", "ESMFold Sidechain\nRMSD", "Rosetta total_score", "Postrelax ligand RMSD"]
    y_labels = ["pLDDT", "Angstrom", "Angstrom", "[REU]", "Angstrom"]
    dims = [(0,100), (0,5), (0,5), None, (0,5)]

    # plot results
    plots.violinplot_multiple_cols(
        dataframe = poses.df,
        cols = cols,
        titles = titles,
        y_labels = y_labels,
        dims = dims,
        out_path = os.path.join(dir, f"cycle_{cycle}_results.png"),
        show_fig = False
    )

    poses.save_poses(out_path=dir)
    poses.save_poses(out_path=dir, poses_col="input_poses")
    poses.save_scores(out_path=dir)

    # write pymol alignment script?
    logging.info(f"Writing pymol alignment script for backbones after refinement cycle {cycle} at {dir}.")
    write_pymol_alignment_script(
        df = poses.df,
        scoreterm = f"cycle_{cycle}_refinement_composite_score",
        top_n = np.min([len(poses.df.index), 25]),
        path_to_script = os.path.join(dir, "align_results.pml"),
        ref_motif_col = "template_fixedres",
        ref_catres_col = "template_fixedres",
        target_catres_col = "fixed_residues",
        target_motif_col = "fixed_residues"
    )

def create_final_results_dir(poses, dir:str):
    # plot outputs and write alignment script

    os.makedirs(dir, exist_ok=True)

    logging.info(f"Plotting final outputs.")
    cols = ["final_AF2_plddt", "final_AF2_mean_plddt", "final_AF2_backbone_rmsd", "final_AF2_catres_heavy_rmsd", "final_fastrelax_total_score", "final_postrelax_catres_heavy_rmsd", "final_postrelax_catres_bb_rmsd", "final_delta_apo_holo", "final_AF2_catres_heavy_rmsd_mean", "final_postrelax_catres_heavy_rmsd_mean", "final_postrelax_ligand_rmsd", "final_postrelax_ligand_rmsd_mean"]
    titles = ["AF2 pLDDT", "AF2 pLDDT", "AF2 BB-Ca RMSD", "AF2 Sidechain\nRMSD", "Rosetta total_score", "Relaxed Sidechain\nRMSD", "Relaxed BB-Ca RMSD", "Delta Apo Holo", "Mean AF2 Sidechain\nRMSD", "Mean Relaxed Sidechain\nRMSD", "Postrelax Ligand\nRMSD", "Mean Postrelax Ligand\nRMSD"]
    y_labels = ["pLDDT", "pLDDT", "Angstrom", "Angstrom", "[REU]", "Angstrom", "Angstrom", "[REU]", "Angstrom", "Angstrom", "Angstrom", "Angstrom"]
    dims = [(0,100), (0,100), (0,5), (0,5), None, (0,5), (0,5), None, (0,5), (0,5), (0,5), (0,5)]

    # plot results
    plots.violinplot_multiple_cols(
        dataframe = poses.df,
        cols = cols,
        titles = titles,
        y_labels = y_labels,
        dims = dims,
        out_path = os.path.join(dir, f"evaluation_results.png"),
        show_fig = False
    )

    plots.violinplot_multiple_cols(
        dataframe=poses.df,
        cols=["final_AF2_catres_heavy_rmsd_mean", "final_AF2_catres_bb_rmsd_mean", "final_postrelax_catres_heavy_rmsd_mean", "final_postrelax_catres_bb_rmsd_mean", "final_postrelax_ligand_rmsd_mean"],
        y_labels=["Angstrom", "Angstrom", "Angstrom", "Angstrom", "Angstrom"],
        titles=["Mean AF2\nSidechain RMSD", "Mean AF2 catres\nBB RMSD", "Mean Relaxed\nSidechain RMSD", "Mean Relaxed catres\nBB RMSD", "Mean Postrelax\nLigand RMSD"],
        out_path=os.path.join(dir, "evaluation_mean_rmsds.png"),
        show_fig=False
    )

    poses.save_poses(out_path=dir)
    poses.save_poses(out_path=dir, poses_col="input_poses")
    poses.save_scores(out_path=dir)

    # write pymol alignment script?
    logging.info(f"Writing pymol alignment script for backbones after evaluation at {dir}.")
    write_pymol_alignment_script(
        df = poses.df,
        scoreterm = "final_composite_score",
        top_n = np.min([25, len(poses.df.index)]),
        path_to_script = os.path.join(dir, "align_results.pml"),
        ref_motif_col = "template_fixedres",
        ref_catres_col = "template_fixedres",
        target_catres_col = "fixed_residues",
        target_motif_col = "fixed_residues"
    )

def write_bbopt_opts(row: pd.Series, cycle: int, total_cycles: int, reference_location_col:str, motif_res_col: str, cat_res_col: str, ligand_chain: str) -> str:
    return f"-in:file:native {row[reference_location_col]} -parser:script_vars motif_res={row[motif_res_col].to_string(ordering='rosetta')} cat_res={row[cat_res_col].to_string(ordering='rosetta')} substrate_chain={ligand_chain} sd={0.8 - (0.4 * cycle/total_cycles)}"

def get_params_file(dir:str, params) -> str:
    '''Checks if args.params_file contains a params file path. If not, it looks for an automatically generated params file. If there is none either, it will not use any params file (return None)'''
    if isinstance(params, str):
        params = params.split(';')
        for param in params:
            if not os.path.isfile(param):
                raise RuntimeError(f"Could not find params file at {params}")
        logging.info(f'Using params files: {params}')
        return params
    else:
        params = glob.glob(os.path.join(dir, "ligand/LG*.params"))
        if params:
            logging.info(f'Using params files: {params}')
            return params
        else:
            logging.warning(f'Could not find any params file at {dir}.')
            return None
        
def import_multiple_ligands(dir):
    ligand_paths = glob(os.path.join(dir, "ligand/LG?.pdb"))
    ligands = []
    print(f'Importing ligands {ligand_paths}.')
    for lig in ligand_paths:
        lig = load_structure_from_pdbfile(lig)
        lig = [i for i in lig.get_residues()][0]
        ligands.append(lig)
    
    return ligands

def calculate_mean_scores(poses: protflow.poses.Poses, scores: list, remove_layers: int=None):
    for score in scores:
        poses.calculate_mean_score(name=f"{score}_mean", score_col=score, remove_layers=remove_layers)
    return poses

def combine_screening_results(dir: str, prefixes: list, scores: list, weights: list, residue_cols: list, model: str):
    if len(prefixes) == 0:
        logging.error("No poses passed in any of the screening runs. Aborting!"); sys.exit(1)
    
    # set up output dir
    out_dir = os.path.join(dir, 'screening_results')
    os.makedirs(out_dir, exist_ok=True)

    # combine all screening outputs into new poses
    pose_df = []
    for prefix in prefixes:
        df = pd.read_json(os.path.join(dir, prefix, f"{prefix}_scores.json"))
        df['screen_passed_poses'] = len(df.index)
        pose_df.append(df)
    pose_df = pd.concat(pose_df).reset_index(drop=True)
    poses = protflow.poses.Poses(poses=pose_df, work_dir=dir)

    # recalculate composite score over all screening runs
    poses.calculate_composite_score(
        name="design_composite_score",
        scoreterms=["esm_plddt", "esm_tm_TM_score_ref", "esm_catres_bb_rmsd", "esm_catres_heavy_rmsd", "esm_lig_contacts", "esm_ligand_clashes"],
        weights=[-1, -1, 4, 3, -0.5, 0.5],
        plot=True
    )

    # convert columns to residues (else, pymol script writer and refinement crash)
    for residue_col in residue_cols:
        poses.df[residue_col] = [protflow.residues.ResidueSelection(motif, from_scorefile=True) for motif in poses.df[residue_col].to_list()]
    # calculate screening composite score
    poses.calculate_composite_score(name='screening_composite_score', scoreterms=scores, weights=weights, plot=True)

    # filter poses to 'baker success' (kind of):
    poses.filter_poses_by_value(score_col="esm_plddt", value=75, operator=">=")
    poses.filter_poses_by_value(score_col=f"esm_tm_TM_score_ref", value=0.8, operator=">=")
    poses.filter_poses_by_value(score_col="esm_catres_bb_rmsd", value=1.5, operator="<=")

    poses.reindex_poses(prefix="reindexed_screening_poses", remove_layers=1, force_reindex=1)

    grouped_df = poses.df.groupby('screen', sort=True)
    # plot all scores
    df_names, dfs = zip(*[(name, df) for name, df in grouped_df])
    if model == "default": plot_scores = scores + ['design_composite_score', 'screen_decentralize_weight', 'screen_decentralize_distance', 'screen', 'screening_composite_score']
    elif model == "active_site": plot_scores = scores + ['design_composite_score', 'screen_substrate_contacts_weight', 'screen_rog_weight', 'screen', 'screening_composite_score']
    for score in plot_scores:
        plots.violinplot_multiple_cols_dfs(dfs=dfs, df_names=df_names, cols=[score], y_labels=[score], out_path=os.path.join(out_dir, f'{score}_violin.png'))

    # save poses dataframe as well
    poses.df.sort_values("design_composite_score", ascending=True, inplace=True)
    poses.df.reset_index(drop=True, inplace=True)
    poses.save_scores(out_path=os.path.join(out_dir, 'screening_results_all.json'))
    poses.save_scores(out_path=os.path.join(out_dir, 'refinement_input_poses.csv'), out_format="csv")

    logging.info(f"Writing pymol alignment script for screening results at {out_dir}")
    write_pymol_alignment_script(
        df=poses.df,
        scoreterm = "design_composite_score",
        top_n = np.min([len(poses.df.index), 25]),
        path_to_script = os.path.join(out_dir, "align_results.pml"),
        ref_motif_col = "template_fixedres",
        ref_catres_col = "template_fixedres",
        target_catres_col = "fixed_residues",
        target_motif_col = "fixed_residues"
    )

    poses.save_poses(out_path=out_dir)
    poses.save_poses(out_path=out_dir, poses_col="input_poses")
    return poses
    
def ramp_cutoff(start_value, end_value, cycle, total_cycles):
    if total_cycles == 1:
        return end_value
    step = (end_value - start_value) / (total_cycles - 1)
    return start_value + (cycle - 1) * step
        

def split_combined_ligand(poses_dir:str, ligand_dir: list, ligand_chain: str = "Z"):
    logging.info('Replacing RFdiffusion combined ligand with separated ligands...')
    ligand_paths = glob.glob(os.path.join(ligand_dir, f"LG?.pdb"))
    new_ligands = [[res for res in load_structure_from_pdbfile(new_lig).get_residues()][0] for new_lig in ligand_paths]
    poses = protflow.poses.Poses(poses=poses_dir, glob_suffix="*.pdb")
    for pose_path in poses.df['poses'].to_list():
        pose = load_structure_from_pdbfile(pose_path)
        original_ligands = [lig for lig in pose[ligand_chain].get_residues()]
        for old_lig in original_ligands:
            pose[ligand_chain].detach_child(old_lig.id)
        for new_lig in new_ligands:
            pose[ligand_chain].add(new_lig)
        save_structure_to_pdbfile(pose, pose_path)

def create_reduced_motif(fixed_res:protflow.residues.ResidueSelection, motif_res:protflow.residues.ResidueSelection):
    reduced_dict = {}
    fixed_res = fixed_res.to_dict()
    motif_res = motif_res.to_dict()
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
    return protflow.residues.from_dict(reduced_dict)

def aa_one_letter_code() -> str:
    return "ARDNCEQGHILKMFPSTWYV"

def omit_AAs(omitted_aas:str, allowed_aas:str) -> str:
    mutations_dict = {}
    if not allowed_aas and not omitted_aas:
        return None
    elif allowed_aas and omitted_aas:
        raise ValueError("omitted_aas and allowed_aas are mutually exclusive!")
    elif omitted_aas:
        omitted_aas = omitted_aas.split(";")
        for mutation in omitted_aas:
            position, omitted_aas = mutation.split(":")
            mutations_dict[position.strip()] = omitted_aas.strip()
            return f"--omit_AA_per_residue {mutations_dict}" 
    elif allowed_aas:
        allowed_aas = allowed_aas.split(";")
        for mutation in allowed_aas:
            position, allowed_aas = mutation.split(":")
            all_aas = aa_one_letter_code()
            for aa in allowed_aas:
                all_aas.remove(aa)
            mutations_dict[position.strip()] = all_aas.strip()   
            return f"--omit_AA_per_residue {mutations_dict}" 

def log_cmd(arguments):
    cmd = ''
    for key, value in vars(arguments).items():
        cmd += f'--{key} {value} '
    cmd = f'{sys.argv[0]} {cmd}'
    logging.info(f"{sys.argv[0]} {cmd}")


def main(args):
    '''executes everyting (duh)'''
    ################################################# SET UP #########################################################
    # logging and checking of inputs
    if not os.path.isdir(args.input_dir):
        raise ValueError(f"Not a directory: {args.input_dir}.")
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(args.output_dir, "riffdiff.log"),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    log_cmd(args)

    if args.ref_input_json and args.eval_input_json:
        raise ValueError(":ref_input_json: and :eval_input_json: are mutually exclusive!")

    # setup jobstarters
    cpu_jobstarter = SbatchArrayJobstarter(max_cores=args.max_cpus)
    small_cpu_jobstarter = SbatchArrayJobstarter(max_cores=10)
    gpu_jobstarter = cpu_jobstarter if args.cpu_only else SbatchArrayJobstarter(max_cores=args.max_gpus, gpus=1)
    real_gpu_jobstarter = SbatchArrayJobstarter(max_cores=args.max_gpus, gpus=1)

    # set up runners
    logging.info(f"Settung up runners.")
    rfdiffusion = protflow.tools.rfdiffusion.RFdiffusion(jobstarter = gpu_jobstarter)
    #fpocket_runner = protflow.metrics.fpocket.FPocket(jobstarter=cpu_jobstarter)
    chain_adder = protflow.tools.protein_edits.ChainAdder(jobstarter = small_cpu_jobstarter)
    chain_remover = protflow.tools.protein_edits.ChainRemover(jobstarter = small_cpu_jobstarter)
    bb_rmsd = BackboneRMSD(chains="A", jobstarter = small_cpu_jobstarter)
    catres_motif_bb_rmsd = MotifRMSD(ref_col = "updated_reference_frags_location", target_motif = "fixed_residues", ref_motif = "fixed_residues", atoms=["N", "CA", "C"], jobstarter=small_cpu_jobstarter)
    catres_motif_heavy_rmsd = MotifRMSD(ref_col = "updated_reference_frags_location", target_motif = "fixed_residues", ref_motif = "fixed_residues", jobstarter=small_cpu_jobstarter)
    ligand_clash = LigandClashes(ligand_chain=args.ligand_chain, factor=args.ligand_clash_factor, atoms=['N', 'CA', 'C', 'O'], jobstarter=small_cpu_jobstarter)
    ligand_contacts = LigandContacts(ligand_chain=args.ligand_chain, min_dist=0, max_dist=8, atoms=['CA'], jobstarter=small_cpu_jobstarter)
    rog_calculator = GenericMetric(module="protflow.utils.metrics", function="calc_rog_of_pdb", jobstarter=small_cpu_jobstarter)
    tm_score_calculator = protflow.metrics.tmscore.TMalign(jobstarter = small_cpu_jobstarter)
    ligand_mpnn = protflow.tools.ligandmpnn.LigandMPNN(jobstarter = gpu_jobstarter)
    rosetta = protflow.tools.rosetta.Rosetta(jobstarter = cpu_jobstarter, fail_on_missing_output_poses=True)
    esmfold = protflow.tools.esmfold.ESMFold(jobstarter = real_gpu_jobstarter)
    ligand_rmsd = MotifSeparateSuperpositionRMSD(
        ref_col="updated_reference_frags_location",
        super_target_motif="fixed_residues",
        super_ref_motif="fixed_residues",
        super_atoms=['N', 'CA', 'C'],
        rmsd_target_motif="ligand_motif",
        rmsd_ref_motif="ligand_motif",
        rmsd_atoms=None,
        rmsd_include_het_atoms=True,
        jobstarter = small_cpu_jobstarter)
    colabfold = protflow.tools.colabfold.Colabfold(jobstarter=real_gpu_jobstarter)
    if args.attnpacker_repack:
        attnpacker = protflow.tools.attnpacker.AttnPacker(jobstarter=gpu_jobstarter)
    
    params_files = get_params_file(dir=args.input_dir, params=args.params_file)
    ligand_res_dict = {args.ligand_chain: [i for i in range(1, len(params_files)+1)]}
    ligandmpnn_options = f"--ligand_mpnn_use_side_chain_context 1 {args.ligandmpnn_options}"

    residue_cols = ["fixed_residues", "motif_residues", "template_motif", "template_fixedres", "ligand_motif"]

    # set up rosetta options
    bb_opt_options = f"-parser:protocol {args.bbopt_script} -beta"
    fr_options = f"-parser:protocol {args.fastrelax_script} -beta"
    if params_files:
        fr_options = fr_options + f" -extra_res_fa {' '.join(params_files)}"
        bb_opt_options = bb_opt_options + f" -extra_res_fa {' '.join(params_files)}"

    if not args.ref_input_json and not args.eval_input_json:
        # load poses
        input_poses_path = os.path.join(args.output_dir, 'screening_input_poses', 'screening_input_poses.json')
        if os.path.isfile(input_poses_path):
            backbones = protflow.poses.Poses(poses=input_poses_path)
            backbones.set_work_dir(args.output_dir)
        else:
            # very messy for historic reasons
            # format path_df to be a DF readable by Poses class
            logging.info(f"Parsing inputs specified at {args.input_dir}")
            input_df = pd.read_json(f"{args.input_dir}/selected_paths.json", typ="frame")
            input_df = input_df.reset_index().rename(columns={"index": "poses_description"}) # pylint: disable=E1101
            input_df["poses"] = f"{args.input_dir}/pdb_in/" + input_df["poses_description"] + ".pdb"
            input_df["input_poses"] = input_df["poses"]
            input_df.to_json((path_df := f"{args.output_dir}/paths.poses.json"))
            backbones = protflow.poses.load_poses(path_df)
            backbones.set_work_dir(args.output_dir)
            if args.screen_from_top:
                backbones.filter_poses_by_rank(n=args.screen_input_poses, score_col='path_score', ascending=False, prefix='screening_input', plot=True)
            else:
                backbones.df = backbones.df.sample(n=args.screen_input_poses)
            backbones.save_poses(os.path.join(args.output_dir, 'screening_input_poses'))
            backbones.save_scores(input_poses_path)
        
        # change flanker lengths of rfdiffusion motif contigs
        if args.flanking:
            backbones.df["rfdiffusion_pose_opts"] = [adjust_flanking(rfdiffusion_pose_opts_str, "split", args.flanker_length) for rfdiffusion_pose_opts_str in backbones.df["rfdiffusion_pose_opts"].to_list()]
        elif args.flanker_length:
            raise ValueError(f"Argument 'total_flanker_length' was given, but not 'flanking'! Both args have to be provided.")

        # adjust linkers
        if args.linker_length == 'auto':
            backbones.df["rfdiffusion_pose_opts"] = [adjust_linker_length(motif_residues, args.total_length, args.flanker_length, rfdiffusion_pose_opts) for motif_residues, rfdiffusion_pose_opts in zip(backbones.df["motif_residues"].to_list(), backbones.df["rfdiffusion_pose_opts"].to_list())]
        else:
            backbones.df["rfdiffusion_pose_opts"] = [overwrite_linker_length(pose_opts, args.total_length, args.linker_length) for pose_opts in backbones.df["rfdiffusion_pose_opts"].to_list()]

        # convert motifs from dict to ResidueSelection
        backbones.df["fixed_residues"] = [protflow.residues.from_dict(motif) for motif in backbones.df["fixed_residues"].to_list()]
        backbones.df["motif_residues"] = [protflow.residues.from_dict(motif) for motif in backbones.df["motif_residues"].to_list()]
        backbones.df["ligand_motif"] = [protflow.residues.from_dict(ligand_res_dict) for _ in range(len(backbones.df.index))]

        # set motif_cols to keep after rfdiffusion:
        motif_cols = ["fixed_residues"]
        if args.model == "default":
            motif_cols.append("motif_residues")

        # store original motifs for calculation of motif RMSDs later
        backbones.df["template_motif"] = backbones.df["motif_residues"]
        backbones.df["template_fixedres"] = backbones.df["fixed_residues"]


        # save run name in df, makes it easier to identify where poses come from when merging results with other runs
        backbones.df["run_name"] = os.path.basename(args.output_dir)
        if params_files: backbones.df["params_file_path"] = ', '.join(params_files)


        ############################################## RFDiffusion ######################################################
    
        # setup rfdiffusion options:
        if args.recenter:
            logging.info(f"Parameter --recenter specified. Setting direction for custom recentering during diffusion towards {args.recenter}")
            if len(args.recenter.split(";")) != 3:
                raise ValueError(f"--recenter needs to be semicolon separated coordinates. E.g. --recenter=31.123;-12.123;-0.342")
            recenter = f",recenter_xyz:{args.recenter}"
        else:
            recenter = ""

        # change pose_opts according to model being used:
        if args.model == "active_site":
            logging.info("Using Active Site Model. Changing contig strings from pose_options.")
            backbones.df["rfdiffusion_pose_opts"] = [active_site_pose_opts(row["rfdiffusion_pose_opts"], row["template_fixedres"], as_model_path=args.as_model_path) for row in backbones]

        # load channel_contig
        if args.channel_contig != "None":
            backbones.df["rfdiffusion_pose_opts"] = backbones.df["rfdiffusion_pose_opts"].str.replace("contigmap.contigs=[", f"contigmap.contigs=[{args.channel_contig}/0 ")

        input_backbones = copy.deepcopy(backbones)
        decentralize_weights = args.screen_decentralize_weights.split(';')
        decentralize_distances = args.screen_decentralize_distances.split(';')
        substrate_contacts_weights = args.screen_substrate_contacts_weight.split(';')
        rog_weights = args.screen_rog_weight.split(';')
        num_rfdiffusions = args.screen_num_rfdiffusions
        if args.model == "default":
            settings = tuple(itertools.product(decentralize_weights, decentralize_distances))
        elif args.model == "active_site":
            settings = tuple(itertools.product(substrate_contacts_weights, rog_weights))
        prefixes = [f"screen_{i+1}" for i, s in enumerate(settings)]

        num_backbones = 0
        for prefix, setting in zip(prefixes, settings):
            logging.info(f"Running {prefix} with settings: {f'decentralize_weight: {setting[0]}, decentralize_distance: {setting[1]}' if args.model=='default' else f'substrate_contacts_weights: {setting[0]}, rog_weights: {setting[1]}'}")
            backbones = copy.deepcopy(input_backbones)
            if args.model == "default":
                backbones.df['screen_decentralize_weight'] = float(setting[0])
                backbones.df['screen_decentralize_distance'] = float(setting[1])
            elif args.model == "active_site":
                backbones.df['screen_substrate_contacts_weight'] = float(setting[0])
                backbones.df['screen_rog_weight'] = float(setting[1])

            backbones.df['screen'] = int(prefix.split('_')[1])

            backbones.set_work_dir(os.path.join(args.output_dir, prefix))
            # setup empty dictionary for all output metrics that should go into a separate DataFrame:
            output_metrics = {"scTM_success": None, "baker_success": None, "fraction_ligand_clashes": None, "average_ligand_contacts": None, "fraction_ligand_contacts": None}

            # run diffusion
            if args.model == "active_site":
                diffusion_options = f"diffuser.T={str(args.rfdiffusion_timesteps)} potentials.guide_scale=5 inference.num_designs={num_rfdiffusions} potentials.guiding_potentials=[\\'type:substrate_contacts,weight:{setting[0]}\\',\\'type:custom_ROG,weight:{setting[1]}\\'] potentials.guide_decay=quadratic"
            else:
                diffusion_options = f"diffuser.T={str(args.rfdiffusion_timesteps)} potentials.guide_scale=5 inference.num_designs={num_rfdiffusions} potentials.guiding_potentials=[\\'type:substrate_contacts,weight:0\\',\\'type:custom_recenter_ROG,weight:{setting[0]},rog_weight:0,distance:{setting[1]}{recenter}\\'] potentials.guide_decay=quadratic"
            backbones = rfdiffusion.run(
                poses=backbones,
                prefix="rfdiffusion",
                num_diffusions=num_rfdiffusions,
                options=diffusion_options,
                pose_options=backbones.df["rfdiffusion_pose_opts"].to_list(),
                update_motifs=motif_cols,
                fail_on_missing_output_poses=False
            )

            num_backbones = len(backbones)

            # remove channel chain (chain B)
            logging.info(f"Diffusion completed, removing channel chain from diffusion outputs.")
            if args.channel_contig != "None":
                chain_remover.run(
                    poses = backbones,
                    prefix = "channel_removed",
                    chains = "B"
                )
            else:
                backbones.df["channel_removed_location"] = backbones.df["rfdiffusion_location"]

            # create updated reference frags:
            if not os.path.isdir((updated_ref_frags_dir := os.path.join(backbones.work_dir, "updated_reference_frags"))):
                os.makedirs(updated_ref_frags_dir)

            logging.info(f"Channel chain removed, now renumbering reference fragments.")
            backbones.df["updated_reference_frags_location"] = update_and_copy_reference_frags(
                input_df = backbones.df,
                ref_col = "input_poses",
                desc_col = "poses_description",
                prefix = "rfdiffusion",
                out_pdb_path = updated_ref_frags_dir,
                keep_ligand_chain = args.ligand_chain
            )

            if len(params_files) > 1:
                split_combined_ligand(updated_ref_frags_dir, ligand_dir = os.path.join(args.input_dir, "ligand"), ligand_chain=args.ligand_chain)

            # calculate ROG after RFDiffusion, when channel chain is already removed:
            logging.info(f"Calculating rfdiffusion_rog and rfdiffusion_catres_rmsd")

            backbones = rog_calculator.run(poses=backbones, prefix="rfdiffusion_rog")
            backbones.df.rename({"rfdiffusion_rog_data": "rfdiffusion_rog"}, inplace=True, axis=1)
            #backbones.df["rfdiffusion_rog"] = [calc_rog_of_pdb(pose) for pose in backbones.poses_list()]

            # calculate motif_rmsd of RFdiffusion (for plotting later)
            catres_motif_bb_rmsd.run(
                poses = backbones,
                prefix = "rfdiffusion_catres",
            )

            # add back the ligand:
            logging.info(f"Metrics calculated, now adding Ligand chain back into backbones.")
            chain_adder.superimpose_add_chain(
                poses = backbones,
                prefix = "post_rfdiffusion_ligand",
                ref_col = "updated_reference_frags_location",
                target_motif = "fixed_residues",
                copy_chain = args.ligand_chain
            )

            # calculate ligand stats
            logging.info(f"Calculating Ligand Statistics")
            backbones = ligand_clash.run(poses=backbones, prefix="rfdiffusion_ligand")
            backbones = ligand_contacts.run(poses=backbones, prefix="rfdiffusion_lig")

            # collect ligand stats into output metrics:
            output_metrics["average_ligand_contacts"] = float(np.nan_to_num(backbones.df[backbones.df["rfdiffusion_ligand_clashes"] < 1]["rfdiffusion_lig_contacts"].mean()))
            output_metrics["fraction_ligand_contacts"] = len(backbones.df[(backbones.df["rfdiffusion_ligand_clashes"] < 1) & (backbones.df["rfdiffusion_lig_contacts"] > args.min_ligand_contacts)]) / num_backbones
            output_metrics["fraction_ligand_clashes"] = len(backbones.df[backbones.df["rfdiffusion_ligand_clashes"] < 1]) / num_backbones

            # plot rfdiffusion_stats
            results_dir = os.path.join(backbones.work_dir, "results")
            if not os.path.isdir(results_dir):
                os.makedirs(results_dir, exist_ok=True)
            plots.violinplot_multiple_cols(
                dataframe = backbones.df,
                cols = ["rfdiffusion_catres_rmsd", "rfdiffusion_rog", "rfdiffusion_lig_contacts", "rfdiffusion_ligand_clashes", "rfdiffusion_plddt"],
                titles = ["Template RMSD", "ROG", "Ligand Contacts", "Ligand Clashes", "RFdiffusion pLDDT"],
                y_labels = ["RMSD [\u00C5]", "ROG [\u00C5]", "#CA", "#Clashes", "pLDDT"],
                dims = [(0,5), None, None, None, (0.8,1)],
                out_path = os.path.join(results_dir, "rfdiffusion_statistics.png"),
                show_fig = False
            )

            if args.rfdiffusion_max_clashes:
                backbones.filter_poses_by_value(score_col="rfdiffusion_ligand_clashes", value=args.rfdiffusion_max_clashes, operator="<=", prefix="rfdiffusion_ligand_clashes", plot=True)
            if args.rfdiffusion_max_rog:
                backbones.filter_poses_by_value(score_col="rfdiffusion_rog", value=args.rfdiffusion_max_rog, operator="<=", prefix="rfdiffusion_rog", plot=True)
            if args.rfdiffusion_min_ligand_contacts:
                backbones.filter_poses_by_value(score_col="rfdiffusion_lig_contacts", value=args.rfdiffusion_min_ligand_contacts, operator=">=", prefix="rfdiffusion_lig_contacts", plot=True)
            
            if len(backbones.df) == 0:
                logging.warning(f"No poses passed RFdiffusion filtering steps during {prefix}")
                prefixes.remove(prefix)
                continue
            ############################################# SEQUENCE DESIGN AND ESMFOLD ########################################################
            # run LigandMPNN
            logging.info(f"Running LigandMPNN on {len(backbones)} poses. Designing {args.screen_num_mpnn_sequences} sequences per pose.")
            backbones = ligand_mpnn.run(
                poses = backbones,
                prefix = "postdiffusion_ligandmpnn",
                nseq = args.screen_num_mpnn_sequences,
                options = ligandmpnn_options,
                model_type = "ligand_mpnn",
                fixed_res_col = "fixed_residues",
                return_seq_threaded_pdbs_as_pose=True if args.screen_mpnn_rlx_mpnn else False
            )

            if args.screen_mpnn_rlx_mpnn:
                # optimize backbones 
                backbones.df[f'screen_bbopt_opts'] = [write_bbopt_opts(row=row, cycle=1, total_cycles=5, reference_location_col="updated_reference_frags_location", cat_res_col="fixed_residues", motif_res_col="motif_residues", ligand_chain=args.ligand_chain) for _, row in backbones.df.iterrows()]
                backbones = rosetta.run(
                    poses = backbones,
                    prefix = "bbopt",
                    rosetta_application="rosetta_scripts.default.linuxgccrelease",
                    nstruct = 1,
                    options = bb_opt_options,
                    pose_options='screen_bbopt_opts'
                )

                # filter backbones down to starting backbones
                backbones.filter_poses_by_rank(n=1, score_col=f"bbopt_total_score", remove_layers=2)

                # run ligandmpnn on relaxed poses
                backbones = ligand_mpnn.run(
                    poses = backbones,
                    prefix = "mpnn",
                    nseq = args.screen_num_mpnn_sequences,
                    model_type = "ligand_mpnn",
                    options = ligandmpnn_options,
                    fixed_res_col = "fixed_residues",
                )

            # predict with ESMFold
            logging.info(f"LigandMPNN finished, now predicting {len(backbones)} sequences using ESMFold.")
            backbones = esmfold.run(
                poses = backbones,
                prefix = "esm"
            )

            ################################################ METRICS ################################################################


            # calculate ROG
            backbones = rog_calculator.run(poses=backbones, prefix="esm_rog")
            backbones.df.rename({"esm_rog_data": "esm_rog"}, inplace=True, axis=1)
            #backbones.df["esm_rog"] = [calc_rog_of_pdb(pose) for pose in backbones.poses_list()]

            # calculate RMSDs (backbone, motif, fixedres)
            logging.info(f"Prediction of {len(backbones.df.index)} sequences completed. Calculating RMSDs to rfdiffusion backbone and reference fragment.")
            backbones = catres_motif_heavy_rmsd.run(poses = backbones, prefix = "esm_catres_heavy")
            backbones = catres_motif_bb_rmsd.run(poses = backbones, prefix = "esm_catres_bb")
            backbones = bb_rmsd.run(poses = backbones, ref_col="rfdiffusion_location", prefix = "esm_backbone")

            # calculate TM-Score and get sc-tm score:
            tm_score_calculator.run(
                poses = backbones,
                prefix = "esm_tm",
                ref_col = "channel_removed_location",
            )

            # run rosetta_script to evaluate residuewise energy
            rosetta.run(
                poses = backbones,
                prefix = "fastrelax",
                rosetta_application="rosetta_scripts.default.linuxgccrelease",
                nstruct = 3,
                options = f"-parser:protocol {args.fastrelax_script} -beta"
            )

            # filter down after fastrelax
            backbones.filter_poses_by_rank(
                n = 1,
                score_col = "fastrelax_total_score",
                remove_layers = 1,
                prefix = "fastrelax_filter",
                plot = True
            )

            ############################################# BACKBONE FILTER ########################################################

            # add back ligand and determine pocket-ness!
            logging.info(f"Adding Ligand back into the structure for ligand-based pocket prediction.")
            chain_adder.superimpose_add_chain(
                poses = backbones,
                prefix = "post_prediction_ligand",
                ref_col = "updated_reference_frags_location",
                target_motif = "fixed_residues",
                copy_chain = args.ligand_chain
            )

            '''
            
            fpocket_runner.run(
                poses = backbones,
                prefix = "postrelax",
                options = f"--chain_as_ligand {args.ligand_chain}",
            )
            '''

            backbones = ligand_clash.run(poses=backbones, prefix="esm_ligand")
            backbones = ligand_contacts.run(poses=backbones, prefix="esm_lig")

            # calculate multi-scorerterm score for the final backbone filter:
            backbones.calculate_composite_score(
                name="design_composite_score",
                scoreterms=["esm_plddt", "esm_tm_TM_score_ref", "esm_catres_bb_rmsd", "esm_catres_heavy_rmsd", "esm_lig_contacts", "esm_ligand_clashes"],
                weights=[-1, -1, 4, 3, -0.5, 0.5],
                plot=True
            )

            # filter down to rfdiffusion backbones
            backbones.filter_poses_by_rank(
                n=1,
                score_col="design_composite_score",
                prefix=f"{prefix}_backbone_filter",
                plot=True,
                remove_layers=4 if args.screen_mpnn_rlx_mpnn else 2
            )

            # plot outputs
            logging.info(f"Plotting outputs.")
            cols = ["rfdiffusion_catres_rmsd", "esm_plddt", "esm_backbone_rmsd", "esm_catres_heavy_rmsd", "fastrelax_total_score", "esm_tm_sc_tm", "esm_rog", "esm_lig_contacts", "esm_ligand_clashes"]
            titles = ["RFDiffusion Motif\nBackbone RMSD", "ESMFold pLDDT", "ESMFold BB-Ca RMSD", "ESMFold Sidechain\nRMSD", "Rosetta total_score", "SC-TM Score", "Radius of Gyration", "Ligand Contacts", "Ligand Clashes"]
            y_labels = ["Angstrom", "pLDDT", "Angstrom", "Angstrom", "[REU]", "TM Score", "Angstrom", "#", "#"]
            dims = [(0,8), (0,100), (0,8), (0,8), None, (0,1), None, None, None]

            # plot results
            plots.violinplot_multiple_cols(
                dataframe = backbones.df,
                cols = cols,
                titles = titles,
                y_labels = y_labels,
                dims = dims,
                out_path = os.path.join(results_dir, "design_results.png"),
                show_fig = False
            )

            # fill up remaining output_metrics post-prediction
            output_metrics["scTM_success"] = len(backbones.df[backbones.df["esm_tm_sc_tm"] >= 0.5]) / num_backbones
            baker_success_df = backbones.df[(backbones.df["esm_plddt"] >= 75) & (backbones.df["esm_backbone_rmsd"] <= 1.5) & (backbones.df["esm_catres_bb_rmsd"] <= 1.5)]
            output_metrics["baker_success"] = len(baker_success_df) / num_backbones
            enzyme_success_df = baker_success_df[(baker_success_df["rfdiffusion_ligand_clashes"] < 1) & (baker_success_df["rfdiffusion_lig_contacts"] > args.min_ligand_contacts) & (baker_success_df["rfdiffusion_rog"] <= args.max_rog)]
            output_metrics["enzyme_success"] = len(baker_success_df[(baker_success_df["rfdiffusion_ligand_clashes"] < 1) & (baker_success_df["rfdiffusion_lig_contacts"] > args.min_ligand_contacts) & (baker_success_df["rfdiffusion_rog"] <= args.max_rog)]) / num_backbones

            # save output metrics
            output_metrics_str = "\n".join([f"\t{metric}: {value}" for metric, value in output_metrics.items()])
            logging.info(f"num_backbones = {num_backbones}")
            logging.info(f"Finished collection of output metrics.\n{output_metrics_str}\n")
            with open(os.path.join(args.output_dir, "output_metrics.json"), 'w', encoding="UTF-8") as f:
                json.dump(output_metrics, f)

            # calculate fraction of (design-successful) backbones where pocket was identified using fpocket.
            #pocket_containing_fraction = backbones.df["postrelax_top_volume"].count() / len(backbones)
            #logging.info(f"Fraction of RFdiffusion design-successful backbones that contain active-site pocket: {pocket_containing_fraction}")

            backbones.reindex_poses(prefix="reindex", remove_layers=5 if args.screen_mpnn_rlx_mpnn else 3, force_reindex=True)

            # copy filtered poses to new location
            pockets_dir = os.path.join(results_dir, "pocket_pdbs")
            backbones.save_poses(out_path=results_dir)
            backbones.save_poses(out_path=results_dir, poses_col="input_poses")
            backbones.save_scores(out_path=results_dir)

            # save pocket structures
            #backbones.save_poses(out_path=pockets_dir, poses_col="postrelax_pocket_location")

            # write pymol alignment script?
            logging.info(f"Created results/ folder and writing pymol alignment script for best backbones at {results_dir}")
            write_pymol_alignment_script(
                df=backbones.df,
                scoreterm="design_composite_score",
                top_n=np.min([len(backbones), 25]),
                path_to_script=os.path.join(results_dir, "align_results.pml"),
                ref_motif_col = "template_fixedres",
                ref_catres_col = "template_fixedres",
                target_catres_col = "fixed_residues",
                target_motif_col = "fixed_residues"
            )

            logging.info(f"Writing pymol alignment script for enzyme_success backbones at {results_dir}")
            write_pymol_alignment_script(
                df=enzyme_success_df,
                scoreterm="design_composite_score",
                top_n=len(enzyme_success_df),
                path_to_script=f"{results_dir}/enzyme_success_backbones.pml",
                ref_motif_col = "template_fixedres",
                ref_catres_col = "template_fixedres",
                target_catres_col = "fixed_residues",
                target_motif_col = "fixed_residues"
            )

            backbones.save_scores()


        scores = ["esm_plddt", "esm_tm_TM_score_ref", "esm_catres_bb_rmsd", "esm_catres_heavy_rmsd", "esm_rog", "esm_lig_contacts", "esm_ligand_clashes", "screen_passed_poses"]
        weights = [-1, -1, 4, 3, 1, -1, -1, 1]
        backbones = combine_screening_results(dir=args.output_dir, prefixes=prefixes, scores=scores, weights=weights, residue_cols=residue_cols, model=args.model)
        backbones.set_work_dir(args.output_dir)
        backbones.save_scores()

        if args.skip_refinement:
            logging.info(f"Skipping refinement. Run concluded, output can be found in {results_dir}")
            sys.exit(1)
    
    ############################################# REFINEMENT ########################################################
    if args.ref_input_json or (not args.ref_input_json and not args.eval_input_json):
        ref_prefix = f"{args.ref_prefix}_" if args.ref_prefix else ""

        if args.ref_input_json:
            logging.info(f"Reading in refinement input poses from {args.ref_input_json}!")
            backbones = protflow.poses.Poses(poses=args.ref_input_json, work_dir=args.output_dir)
            backbones.df["ligand_motif"] = [protflow.residues.from_dict(ligand_res_dict) for _ in range(len(backbones.df.index))] # TODO: Remove, just here for legacy
            for res_col in residue_cols:
                if not res_col == "ligand_motif": 
                #print([motif for motif in  backbones.df[res_col].to_list()])
                    backbones.df[res_col] = [protflow.residues.ResidueSelection(motif, from_scorefile=True) for motif in backbones.df[res_col].to_list()]

        backbones.set_work_dir(args.output_dir)

        if args.ref_input_poses_per_bb:
            logging.info(f"Filtering refinement input poses on per backbone level according to design_composite_score...")
            backbones.filter_poses_by_rank(n=args.ref_input_poses_per_bb, score_col=f'design_composite_score', remove_layers=1, prefix='refinement_input', plot=True)
        elif args.ref_input_poses:
            logging.info(f"Filtering refinement input according to design_composite_score...")
            backbones.filter_poses_by_rank(n=args.ref_input_poses, score_col=f'design_composite_score', prefix='refinement_input', plot=True)

        # use reduced motif if specified
        if args.use_reduced_motif:
            backbones.df["motif_residues"] = backbones.df.apply(lambda row: create_reduced_motif(row['fixed_residues'], row['motif_residues']), axis=1)

        # create refinement input poses dir
        refinement_input_dir = os.path.join(args.output_dir, f"{ref_prefix}refinement_input_poses")
        os.makedirs(refinement_input_dir, exist_ok=True)
        backbones.save_scores(out_path=os.path.join(refinement_input_dir, f"{ref_prefix}refinement_input_scores.json"), out_format="json")
        backbones.save_poses(out_path=refinement_input_dir)
        backbones.save_poses(out_path=refinement_input_dir, poses_col="input_poses")
        write_pymol_alignment_script(
            df=backbones.df,
            scoreterm="design_composite_score",
            top_n=np.min([len(backbones), 25]),
            path_to_script=os.path.join(refinement_input_dir, "align_poses.pml"),
            ref_motif_col = "template_fixedres",
            ref_catres_col = "template_fixedres",
            target_catres_col = "fixed_residues",
            target_motif_col = "fixed_residues"
        )

        logging.info(f"Plotting refinement input data.")
        cols = ["esm_plddt", "esm_backbone_rmsd", "esm_catres_bb_rmsd", "esm_catres_heavy_rmsd", "fastrelax_total_score", "esm_tm_sc_tm", "esm_rog", "esm_lig_contacts", "esm_ligand_clashes", "screen", "screen_decentralize_weight", "screen_decentralize_distance"]
        titles = ["ESMFold pLDDT", "ESMFold BB-Ca RMSD", "ESMFold fixed res\nBB-Ca RMSD", "ESMFold Sidechain\nRMSD", "Rosetta total_score", "SC-TM Score", "Radius of Gyration", "Ligand Contacts", "Ligand Clashes", "screen number", "decentralize weight", "decentralize distance"]
        y_labels = ["pLDDT", "Angstrom", "Angstrom", "Angstrom", "[REU]", "TM Score", "Angstrom", "#", "#", "#", "AU", "Angstrom"]
        dims = [None, None, None, None, None, None, None, None, None, None, None, None]

        plots.violinplot_multiple_cols(
            dataframe = backbones.df,
            cols = cols,
            titles = titles,
            y_labels = y_labels,
            dims = dims,
            out_path = os.path.join(backbones.plots_dir, f"{ref_prefix}refinement_input_poses.png"),
            show_fig = False
        )
        shutil.copy(os.path.join(backbones.plots_dir, f"{ref_prefix}refinement_input_poses.png"), refinement_input_dir)

        # instantiate plotting trajectories
        trajectory_plots = instantiate_trajectory_plotting(backbones.plots_dir, backbones.df)

        for cycle in range(1, args.ref_cycles+1):
            cycle_work_dir = os.path.join(args.output_dir, f"{ref_prefix}refinement_cycle_{cycle}")
            backbones.set_work_dir(cycle_work_dir)

            logging.info(f"Starting refinement cycle {cycle} in directory {cycle_work_dir}")

            # run ligandmpnn, return pdbs as poses
            backbones = ligand_mpnn.run(
                poses = backbones,
                prefix = f"cycle_{cycle}_seq_thread",
                nseq = args.ref_seq_thread_num_mpnn_seqs,
                model_type = "ligand_mpnn",
                options = ligandmpnn_options,
                fixed_res_col = "fixed_residues",
                return_seq_threaded_pdbs_as_pose=True
            )

            # optimize backbones 
            backbones.df[f'cycle_{cycle}_bbopt_opts'] = [write_bbopt_opts(row=row, cycle=cycle, total_cycles=args.ref_cycles, reference_location_col="updated_reference_frags_location", cat_res_col="fixed_residues", motif_res_col="motif_residues", ligand_chain=args.ligand_chain) for _, row in backbones.df.iterrows()]

            backbones = rosetta.run(
                poses = backbones,
                prefix = f"cycle_{cycle}_bbopt",
                rosetta_application="rosetta_scripts.default.linuxgccrelease",
                nstruct = 3,
                options = bb_opt_options,
                pose_options=f'cycle_{cycle}_bbopt_opts'
            )

            # filter backbones down to starting backbones
            backbones.filter_poses_by_rank(n=1, score_col=f"cycle_{cycle}_bbopt_total_score", remove_layers=2)

            # run ligandmpnn on optimized poses
            backbones = ligand_mpnn.run(
                poses = backbones,
                prefix = f"cycle_{cycle}_mpnn",
                nseq = args.ref_num_mpnn_seqs,
                model_type = "ligand_mpnn",
                options = ligandmpnn_options,
                fixed_res_col = "fixed_residues",
            )

            # predict structures using ESMFold
            backbones = esmfold.run(
                poses = backbones,
                prefix = f"cycle_{cycle}_esm",
            )


            # calculate rmsds, TMscores and clashes
            logging.info(f"Calculating post-ESMFold RMSDs.")
            backbones = catres_motif_heavy_rmsd.run(poses = backbones, prefix = f"cycle_{cycle}_esm_catres_heavy")
            backbones = catres_motif_bb_rmsd.run(poses = backbones, prefix = f"cycle_{cycle}_esm_catres_bb")
            backbones = bb_rmsd.run(poses = backbones, ref_col=f"cycle_{cycle}_bbopt_location", prefix = f"cycle_{cycle}_esm_backbone")
            backbones = tm_score_calculator.run(poses = backbones, prefix = f"cycle_{cycle}_esm_tm", ref_col = f"cycle_{cycle}_bbopt_location")
            
            # calculate cutoff
            plddt_cutoff = ramp_cutoff(args.ref_plddt_cutoff_start, args.ref_plddt_cutoff_end, cycle, args.ref_cycles)
            catres_bb_rmsd_cutoff = ramp_cutoff(args.ref_catres_bb_rmsd_cutoff_start, args.ref_catres_bb_rmsd_cutoff_end, cycle, args.ref_cycles)
            # apply backbone-based filters
            logging.info(f"Applying post-ESMFold backbone filters...")
            backbones.filter_poses_by_value(score_col=f"cycle_{cycle}_esm_plddt", value=plddt_cutoff, operator=">=", prefix=f"cycle_{cycle}_esm_plddt", plot=True)
            backbones.filter_poses_by_value(score_col=f"cycle_{cycle}_esm_tm_TM_score_ref", value=0.9, operator=">=", prefix=f"cycle_{cycle}_esm_TM_score", plot=True)
            backbones.filter_poses_by_value(score_col=f"cycle_{cycle}_esm_catres_bb_rmsd", value=catres_bb_rmsd_cutoff, operator="<=", prefix=f"cycle_{cycle}_esm_catres_bb", plot=True)

            # repack predictions with attnpacker, if set
            if args.attnpacker_repack:
                logging.info("Repacking ESMFold output with Attnpacker...")
                backbones = attnpacker.run(
                    poses=backbones,
                    prefix=f"cycle_{cycle}_packing"
                )

            # copy description column for merging with holo relaxed structures later
            backbones.df[f'cycle_{cycle}_rlx_description'] = backbones.df['poses_description']
            apo_backbones = copy.deepcopy(backbones)

            # relax apo poses
            logging.info("Relaxing poses without ligand present...")
            apo_backbones = rosetta.run(
                poses = apo_backbones,
                prefix = f"cycle_{cycle}_fastrelax_apo",
                rosetta_application="rosetta_scripts.default.linuxgccrelease",
                nstruct = 3,
                options = fr_options
            )

            # filter for top relaxed apo pose and merge with original dataframe
            apo_backbones.filter_poses_by_rank(n=1, score_col=f"cycle_{cycle}_fastrelax_apo_total_score", remove_layers=1)
            backbones.df = backbones.df.merge(apo_backbones.df[[f'cycle_{cycle}_rlx_description', f"cycle_{cycle}_fastrelax_apo_total_score"]], on=f'cycle_{cycle}_rlx_description')

            # add ligand to poses
            logging.info("Adding ligand to ESMFold predictions...")
            backbones = chain_adder.superimpose_add_chain(
                poses = backbones,
                prefix = f"cycle_{cycle}_ligand",
                ref_col = "updated_reference_frags_location",
                target_motif = "fixed_residues",
                copy_chain = args.ligand_chain
            )

            # calculate ligand clashes and ligand contacts
            #backbones = ligand_clash.run(poses=backbones, prefix=f"cycle_{cycle}_esm_ligand")
            #backbones = ligand_contacts.run(poses=backbones, prefix=f"cycle_{cycle}_esm_lig")

            # run rosetta_script to evaluate residuewise energy
            logging.info("Relaxing poses with ligand present...")
            backbones = rosetta.run(
                poses = backbones,
                prefix = f"cycle_{cycle}_fastrelax",
                rosetta_application="rosetta_scripts.default.linuxgccrelease",
                nstruct = 3,
                options = fr_options
            )

            # calculate RMSD on relaxed poses
            logging.info(f"Calculating RMSD of catalytic residues and ligand for relaxed poses...")
            backbones = catres_motif_heavy_rmsd.run(poses = backbones, prefix = f"cycle_{cycle}_postrelax_catres_heavy")
            backbones = catres_motif_bb_rmsd.run(poses = backbones, prefix = f"cycle_{cycle}_postrelax_catres_bb")
            backbones = ligand_rmsd.run(poses = backbones, prefix = f"cycle_{cycle}_postrelax_ligand")
            backbones= calculate_mean_scores(poses=backbones, scores=[f"cycle_{cycle}_postrelax_catres_heavy_rmsd", f"cycle_{cycle}_postrelax_catres_bb_rmsd", f"cycle_{cycle}_postrelax_ligand_rmsd"], remove_layers=1)

            # filter backbones down to relax input backbones
            backbones.filter_poses_by_rank(n=1, score_col=f"cycle_{cycle}_fastrelax_total_score", remove_layers=1)

            # ramp cutoffs during refinement
            ligand_rmsd_cutoff = ramp_cutoff(args.ref_ligand_rmsd_start, args.ref_ligand_rmsd_end, cycle, args.ref_cycles)
            # apply filters
            logging.info("Removing poses with ligand rmsd above cutoff...")
            backbones.filter_poses_by_value(score_col=f"cycle_{cycle}_postrelax_ligand_rmsd", value=ligand_rmsd_cutoff, operator="<=", prefix=f"cycle_{cycle}_ligand_rmsd", plot=True)        

            # calculate delta apo holo score
            logging.info("Calculating delta total score between relaxed poses with and without ligand present...")
            backbones.df[f'cycle_{cycle}_delta_apo_holo'] = backbones.df[f"cycle_{cycle}_fastrelax_total_score"] - backbones.df[f"cycle_{cycle}_fastrelax_apo_total_score"]

            # calculate multi-scoreterm score for the final backbone filter:
            logging.info("Calculating composite score for refinement evaluation...")
            backbones.calculate_composite_score(
                name=f"cycle_{cycle}_refinement_composite_score",
                scoreterms=[f"cycle_{cycle}_esm_plddt", f"cycle_{cycle}_esm_tm_TM_score_ref", f"cycle_{cycle}_esm_catres_bb_rmsd", f"cycle_{cycle}_esm_catres_heavy_rmsd", f"cycle_{cycle}_delta_apo_holo", f"cycle_{cycle}_postrelax_ligand_rmsd", f"cycle_{cycle}_postrelax_catres_heavy_rmsd"],
                weights=[-1, -1, 4, 4, 1, 1, 1],
                plot=True
            )

            # define number of index layers that were added during refinement cycle (higher in subsequent cycles because reindexing adds a layer)
            layers = 4
            if cycle > 1: layers += 1
            if args.attnpacker_repack: layers += 1

            # manage screen output
            backbones.reindex_poses(prefix=f"cycle_{cycle}_reindex", remove_layers=layers, force_reindex=True)

            # copy output of final round pre-filtering
            if cycle == args.ref_cycles: refinement_results = copy.deepcopy(backbones)

            # filter down to rfdiffusion backbones
            logging.info("Filtering poses according to composite score...")
            backbones.filter_poses_by_rank(
                n=args.ref_num_cycle_poses,
                score_col=f"cycle_{cycle}_refinement_composite_score",
                prefix=f"cycle_{cycle}_refinement_composite_score",
                plot=True,
                remove_layers=1
            )

            trajectory_plots = update_trajectory_plotting(trajectory_plots=trajectory_plots, df=backbones.df, cycle=cycle)
            results_dir = os.path.join(backbones.work_dir, f"cycle_{cycle}_results")
            create_ref_results_dir(poses=backbones, dir=results_dir, cycle=cycle)

        # sort output
        backbones = refinement_results
        backbones.df.sort_values(f"cycle_{cycle}_refinement_composite_score", ascending=True, inplace=True)
        backbones.df.reset_index(drop=True, inplace=True)

        refinement_results_dir = os.path.join(args.output_dir, f"{ref_prefix}refinement_results")
        create_ref_results_dir(poses=backbones, dir=refinement_results_dir, cycle=cycle)
        backbones.save_scores(out_path=os.path.join(refinement_results_dir, "evaluation_input_poses.json"), out_format="json")
        backbones.set_work_dir(args.output_dir)
        backbones.save_scores()

        if args.skip_evaluation:
            logging.info(f"Skipping evaluation. Run concluded, per-backbone output can be found in {os.path.join(backbones.work_dir, f'cycle_{cycle}_results')}. Overall results can be found in {refinement_results_dir}.")
            sys.exit(1)

    ########################### FINAL EVALUATION ###########################
    if args.eval_prefix: eval_prefix = f"{args.eval_prefix}_"
    else: eval_prefix = ref_prefix if ref_prefix else ""

    # set up poses for evaluation
    if args.eval_input_json:
        logging.info(f"Reading in evaluation input poses from {args.eval_input_json}!")
        backbones = protflow.poses.Poses(poses=args.eval_input_json)
        logging.warning(f"Using data from refinement cycle {args.ref_cycles}. Make sure this is the correct one when reading in evaluation input poses from file!")
        for res_col in residue_cols:
            backbones.df[res_col] = [protflow.residues.ResidueSelection(motif, from_scorefile=True) for motif in backbones.df[res_col].to_list()]

    backbones.set_work_dir(os.path.join(args.output_dir, f"{eval_prefix}evaluation"))

    if args.eval_input_poses_per_bb:
        logging.info(f"Filtering evaluation input poses on per backbone level according to cycle_{args.ref_cycles}_refinement_composite_score...")
        backbones.filter_poses_by_rank(n=args.eval_input_poses_per_bb, score_col=f"cycle_{args.ref_cycles}_refinement_composite_score", remove_layers=1, prefix="evaluation_input_per_bb", plot=True)
    if args.eval_input_poses: 
        logging.info(f"Filtering evaluation input poses according to cycle_{args.ref_cycles}_refinement_composite_score...")
        backbones.filter_poses_by_rank(n=args.eval_input_poses, score_col=f"cycle_{args.ref_cycles}_refinement_composite_score", prefix="evaluation_input", plot=True)

    evaluation_input_poses_dir = os.path.join(backbones.work_dir, "evaluation_input_poses")
    os.makedirs(evaluation_input_poses_dir, exist_ok=True)
    backbones.save_poses(out_path=evaluation_input_poses_dir)
    backbones.save_poses(out_path=evaluation_input_poses_dir, poses_col="input_poses")
    backbones.save_scores(out_path=evaluation_input_poses_dir)

    # write pymol alignment script
    logging.info(f"Writing pymol alignment script for evaluation input poses at {evaluation_input_poses_dir}.")
    write_pymol_alignment_script(
        df = backbones.df,
        scoreterm = f"cycle_{args.ref_cycles}_refinement_composite_score",
        top_n = np.min([len(backbones.df.index), 25]),
        path_to_script = os.path.join(evaluation_input_poses_dir, "align_input_poses.pml"),
        ref_motif_col = "template_fixedres",
        ref_catres_col = "template_fixedres",
        target_catres_col = "fixed_residues",
        target_motif_col = "fixed_residues"
    )

    backbones.convert_pdb_to_fasta(prefix="final_fasta_conversion", update_poses=True)

    # run AF2
    backbones = colabfold.run(
        poses=backbones,
        prefix="final_AF2",
        return_top_n_poses=5
    )

    if args.attnpacker_repack:
        backbones = attnpacker.run(
            poses=backbones,
            prefix=f"final_packing"
        )

    # copy description column for merging with apo relaxed structures
    backbones.df['final_relax_input_description'] = backbones.df['poses_description']
    apo_backbones = copy.deepcopy(backbones)
    apo_backbones.filter_poses_by_rank(n=1, score_col="final_AF2_plddt", remove_layers=1 if not args.attnpacker_repack else 2)

    # add ligand chain 
    backbones = chain_adder.superimpose_add_chain(
        poses = backbones,
        prefix = f"final_ligand",
        ref_col = "updated_reference_frags_location",
        target_motif = "fixed_residues",
        copy_chain = args.ligand_chain
    )

    # calculate RMSDs & TMscore
    backbones = catres_motif_heavy_rmsd.run(poses=backbones, prefix=f"final_AF2_catres_heavy")
    backbones = catres_motif_bb_rmsd.run(poses=backbones, prefix=f"final_AF2_catres_bb")
    backbones = bb_rmsd.run(poses=backbones, prefix="final_AF2_backbone", ref_col=f"cycle_{args.ref_cycles}_bbopt_location")
    backbones = bb_rmsd.run(poses=backbones, prefix="final_AF2_ESM_bb", ref_col=f"cycle_{args.ref_cycles}_esm_location")
    backbones = tm_score_calculator.run(poses=backbones, prefix=f"final_AF2_tm", ref_col=f"cycle_{args.ref_cycles}_bbopt_location")
    backbones = tm_score_calculator.run(poses=backbones, prefix=f"final_AF2_ESM_tm", ref_col=f"cycle_{args.ref_cycles}_esm_location")

    # average scores for all AF2 models
    backbones = calculate_mean_scores(poses=backbones, scores=["final_AF2_catres_heavy_rmsd", "final_AF2_catres_bb_rmsd", "final_AF2_backbone_rmsd", "final_AF2_tm_TM_score_ref"], remove_layers=1 if not args.attnpacker_repack else 2)
    
    # filter for AF2 top model
    backbones.filter_poses_by_rank(n=1, score_col="final_AF2_plddt", remove_layers=1 if not args.attnpacker_repack else 2)

    # calculate ligand clashes and ligand contacts
    backbones = ligand_clash.run(poses=backbones, prefix="final_AF2_ligand")
    backbones = ligand_contacts.run(poses=backbones, prefix="final_AF2_lig")

    # relax predictions with ligand present
    backbones = rosetta.run(
        poses = backbones,
        prefix = "final_fastrelax",
        rosetta_application="rosetta_scripts.default.linuxgccrelease",
        nstruct = 5,
        options = fr_options
    )

    # calculate RMSDs of relaxed poses
    backbones = catres_motif_heavy_rmsd.run(poses = backbones, prefix = f"final_postrelax_catres_heavy")
    backbones = catres_motif_bb_rmsd.run(poses = backbones, prefix = f"final_postrelax_catres_bb")
    backbones = ligand_rmsd.run(poses = backbones, prefix = "final_postrelax_ligand")


    # average values for all relaxed poses
    backbones = calculate_mean_scores(poses=backbones, scores=["final_postrelax_catres_heavy_rmsd", "final_postrelax_catres_bb_rmsd", "final_postrelax_ligand_rmsd"], remove_layers=1)
    
    # plot mean results
    plots.violinplot_multiple_cols(
        dataframe=backbones.df,
        cols=["final_AF2_catres_heavy_rmsd_mean", "final_AF2_catres_bb_rmsd_mean", "final_postrelax_catres_heavy_rmsd_mean", "final_postrelax_catres_bb_rmsd", "final_postrelax_ligand_rmsd"],
        y_labels=["Angstrom", "Angstrom", "Angstrom", "Angstrom", "Angstrom", "Angstrom"],
        titles=["Mean AF2\nSidechain RMSD", "Mean AF2 catres\nBB RMSD", "Mean Relaxed\nSidechain RMSD", "Mean Relaxed catres\nBB RMSD", "Mean Relaxed ligand\nRMSD"],
        out_path=os.path.join(backbones.plots_dir, "final_mean_rmsds.png"),
        show_fig=False
    )

    # filter to relaxed pose with best score
    backbones.filter_poses_by_rank(n=1, score_col="final_fastrelax_total_score", remove_layers=1)

    # relax apo poses
    apo_backbones = rosetta.run(
        poses = apo_backbones,
        prefix = "final_fastrelax_apo",
        rosetta_application="rosetta_scripts.default.linuxgccrelease",
        nstruct = 5,
        options = fr_options
    )

    # filter to relaxed pose with best score, merge dataframes
    apo_backbones.filter_poses_by_rank(n=1, score_col="final_fastrelax_apo_total_score", remove_layers=1)
    backbones.df = backbones.df.merge(apo_backbones.df[['final_relax_input_description', 'final_fastrelax_apo_total_score']], on='final_relax_input_description')
    
    # calculate delta score between apo and holo poses
    backbones.df['final_delta_apo_holo'] = backbones.df['final_fastrelax_total_score'] - backbones.df['final_fastrelax_apo_total_score']

    # calculate final composite score
    backbones.calculate_composite_score(
        name=f"final_composite_score",
        scoreterms=["final_AF2_plddt", "final_AF2_tm_TM_score_ref", "final_AF2_catres_bb_rmsd", "final_AF2_catres_heavy_rmsd", "final_delta_apo_holo", "final_postrelax_ligand_rmsd", "final_postrelax_catres_heavy_rmsd"],
        weights=[-1, -1, 4, 4, 1, 1, 1],
        plot=True
    )

    # apply filters only after calculating composite scores!
    backbones.filter_poses_by_value(score_col="final_AF2_mean_plddt", value=args.ref_plddt_cutoff_end, operator=">=", prefix="final_AF2_plddt", plot=True)
    backbones.filter_poses_by_value(score_col="final_AF2_tm_TM_score_ref", value=0.9, operator=">=", prefix=f"final_AF2_TM_score", plot=True)
    backbones.filter_poses_by_value(score_col="final_AF2_catres_bb_rmsd_mean", value=args.ref_catres_bb_rmsd_cutoff_end, operator="<=", prefix=f"final_AF2_catres_bb_rmsd", plot=True)
    backbones.filter_poses_by_value(score_col="final_postrelax_ligand_rmsd_mean", value=args.ref_ligand_rmsd_end, operator="<=", prefix="final_ligand_rmsd", plot=True)        
    backbones.filter_poses_by_value(score_col="final_AF2_ESM_bb_rmsd", value=1.5, operator="<=", prefix="final_AF2_ESM_bb_rmsd", plot=True) # check if AF2 and ESM predictions agree      

    #backbones.filter_poses_by_value(score_col="final_ligand_bb_clashes", value=1, operator="<")

    backbones.reindex_poses(prefix="final_reindex", remove_layers=2 if not args.attnpacker_repack else 3)
    #backbones.filter_poses_by_rank(n=25, score_col='final_composite_score', prefix="final_composite_score", plot=True)
    if not args.eval_input_json:
        trajectory_plots = add_final_data_to_trajectory_plots(backbones.df, trajectory_plots)
    create_final_results_dir(backbones, os.path.join(args.output_dir, f"{eval_prefix}evaluation_results"))
    backbones.save_scores()


    ########################### VARIANT GENERATION ###########################
    if args.variants_mutations_csv:

        if args.variants_prefix: variants_prefix = f"{args.variants_prefix}_"
        else: variants_prefix = eval_prefix if eval_prefix else ""

        mutations = pd.read_csv(args.variants_mutations_csv)
        if args.variants_input_json:
            logging.info(f"Reading in variant generation input poses from {args.variants_input_json}!")
            backbones = protflow.poses.Poses(poses=args.variants_input_json)
            for res_col in residue_cols:
                backbones.df[res_col] = [protflow.residues.ResidueSelection(motif, from_scorefile=True) for motif in backbones.df[res_col].to_list()]
        
        backbones.set_work_dir(os.path.join(args.output_dir, f"{variants_prefix}variants"))

        backbones.df = backbones.df.merge(mutations, on="poses_description")
        backbones.df["variants_pose_opts"] = backbones.df.apply(lambda row: omit_AAs(row['omit_AAs'], row['allow_AAs']), axis=1)

        backbones.df[f'variants_bbopt_opts'] = [write_bbopt_opts(row=row, cycle=1, total_cycles=1, reference_location_col="updated_reference_frags_location", cat_res_col="fixed_residues", motif_res_col="motif_residues", ligand_chain=args.ligand_chain) for _, row in backbones.df.iterrows()]

        backbones = rosetta.run(
            poses = backbones,
            prefix = f"variants_bbopt",
            rosetta_application="rosetta_scripts.default.linuxgccrelease",
            nstruct = 5,
            options = bb_opt_options,
            pose_options='variants_bbopt_opts'
        )

        # filter backbones down to starting backbones
        backbones.filter_poses_by_rank(n=1, score_col="variants_bbopt_total_score", remove_layers=1)

        ligandmpnn_options = f"--ligand_mpnn_use_side_chain_context 1 {args.ligandmpnn_options}"

        # optimize sequences
        backbones = ligand_mpnn.run(
            poses = backbones,
            prefix = f"variants_mpnn",
            nseq = 50,
            model_type = "ligand_mpnn",
            options = ligandmpnn_options,
            pose_options = "variants_pose_opts",
            fixed_res_col = "fixed_residues",
        )

        # predict structures using ESMFold
        backbones = esmfold.run(
            poses = backbones,
            prefix = f"variants_esm",
        )

        # repack predictions with attnpacker, if set
        if args.attnpacker_repack:
            backbones = attnpacker.run(
                poses=backbones,
                prefix=f"variants_packing"
            )

        # copy description column for merging with holo relaxed structures later
        backbones.df[f'variants_rlx_description'] = backbones.df['poses_description']
        apo_backbones = copy.deepcopy(backbones)

        # relax apo poses
        apo_backbones = rosetta.run(
            poses = apo_backbones,
            prefix = f"variants_fastrelax_apo",
            rosetta_application="rosetta_scripts.default.linuxgccrelease",
            nstruct = 3,
            options = fr_options
        )

        # filter for top relaxed apo pose and merge with original dataframe
        apo_backbones.filter_poses_by_rank(n=1, score_col="variants_fastrelax_apo_total_score", remove_layers=1)
        backbones.df = backbones.df.merge(apo_backbones.df[['variants_rlx_description', "variants_fastrelax_apo_total_score"]], on='variants_rlx_description')

        # add ligand to poses
        backbones = chain_adder.superimpose_add_chain(
            poses = backbones,
            prefix = f"variants_ligand",
            ref_col = "updated_reference_frags_location",
            target_motif = "fixed_residues",
            copy_chain = args.ligand_chain
        )

        # calculate ligand clashes and ligand contacts
        backbones = ligand_clash.run(poses=backbones, prefix=f"variants_esm_ligand")
        backbones = ligand_contacts.run(poses=backbones, prefix=f"variants_esm_lig")

        # calculate rmsds, TMscores and clashes
        backbones = catres_motif_heavy_rmsd.run(poses = backbones, prefix = f"variants_esm_catres_heavy")
        backbones = catres_motif_bb_rmsd.run(poses = backbones, prefix = f"variants_esm_catres_bb")
        backbones = bb_rmsd.run(poses = backbones, ref_col=f"variants_bbopt_location", prefix = f"variants_esm_backbone")
        backbones = tm_score_calculator.run(poses = backbones, prefix = f"variants_esm_tm", ref_col = f"variants_bbopt_location")

        # run rosetta_script to evaluate residuewise energy
        backbones = rosetta.run(
            poses = backbones,
            prefix = f"variants_fastrelax",
            rosetta_application="rosetta_scripts.default.linuxgccrelease",
            nstruct = 3,
            options = fr_options
        )

        # calculate delta apo holo score
        backbones.df[f'variants_delta_apo_holo'] = backbones.df[f"variants_fastrelax_total_score"] - backbones.df[f"variants_fastrelax_apo_total_score"]

        # calculate RMSD on relaxed poses
        logging.info(f"Relax finished. Now calculating RMSD of catalytic residues for {len(backbones)} structures.")
        backbones = catres_motif_heavy_rmsd.run(poses = backbones, prefix = "variants_postrelax_catres_heavy")
        backbones = catres_motif_bb_rmsd.run(poses = backbones, prefix = "variants_postrelax_catres_bb")
        backbones = ligand_rmsd.run(poses = backbones, prefix = "variants_postrelax_ligand")
        backbones = calculate_mean_scores(poses=backbones, scores=["variants_postrelax_catres_heavy_rmsd", "variants_postrelax_catres_bb_rmsd", "variants_postrelax_ligand_rmsd"], remove_layers=1)

        # filter backbones down to relax input backbones
        backbones.filter_poses_by_rank(n=1, score_col="variants_fastrelax_total_score", remove_layers=1)

        # calculate multi-scoreterm score for the final backbone filter:
        backbones.calculate_composite_score(
            name="variants_composite_score",
            scoreterms=["variants_esm_plddt", "variants_esm_tm_TM_score_ref", "variants_esm_catres_bb_rmsd", "variants_esm_catres_heavy_rmsd", "variants_delta_apo_holo", "variants_postrelax_ligand_rmsd", "variants_postrelax_catres_heavy_rmsd"],
            weights=[-1, -1, 4, 4, 1, 1, 1],
            plot=True
        )

        # apply filters
        backbones.filter_poses_by_value(score_col=f"cycle_{cycle}_esm_plddt", value=args.ref_plddt_cutoff_end, operator=">=", prefix=f"cycle_{cycle}_esm_plddt", plot=True)
        backbones.filter_poses_by_value(score_col=f"cycle_{cycle}_esm_tm_TM_score_ref", value=0.9, operator=">=", prefix=f"cycle_{cycle}_esm_TM_score", plot=True)
        backbones.filter_poses_by_value(score_col=f"cycle_{cycle}_esm_catres_bb_rmsd", value=args.ref_catres_bb_rmsd_cutoff_end, operator="<=", prefix=f"cycle_{cycle}_esm_catres_bb", plot=True)
        backbones.filter_poses_by_value(score_col=f"cycle_{cycle}_postrelax_ligand_rmsd", value=args.ref_ligand_rmsd_end, operator="<=", prefix=f"cycle_{cycle}_ligand_rmsd", plot=True)        

        # define number of index layers that were added during refinement cycle (higher in subsequent cycles because reindexing adds a layer)
        layers = 3
        if args.attnpacker_repack: layers += 1

        # manage screen output
        backbones.reindex_poses(prefix=f"variants_reindex", remove_layers=layers, force_reindex=True)

        # filter down to rfdiffusion backbones
        backbones.filter_poses_by_rank(
            n=args.variants_num_output_poses,
            score_col="variants_composite_score",
            prefix="variants_composite_score",
            plot=True,
            remove_layers=1
        )



if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--input_dir", type=str, required=True, help="input_directory that contains all ensemble *.pdb files to be hallucinated (max 1000 files).")
    argparser.add_argument("--output_dir", type=str, required=True, help="output_directory")

    # general optionals
    argparser.add_argument("--skip_refinement", action="store_true", help="Skip refinement and evaluation, only run screening.")
    argparser.add_argument("--skip_evaluation", action="store_true", help="Skip evaluation, only run screening and refinement.")
    argparser.add_argument("--ligand_chain", type=str, default="Z", help="Chain name of the ligand chain.")
    argparser.add_argument("--params_file", type=str, default=None, help="Path to alternative params file. Can also be multiple paths separated by ';'.")
    argparser.add_argument("--attnpacker_repack", action="store_true", help="Run attnpacker on ESM and AF2 predictions")
    argparser.add_argument("--use_reduced_motif", action="store_true", help="Instead of using the full fragments during backbone optimization, just use residues directly adjacent to fixed_residues. Also affects motif_bb_rmsd etc.")

    # jobstarter
    argparser.add_argument("--cpu_only", action="store_true", help="Should only cpu's be used during the entire pipeline run?")
    argparser.add_argument("--max_gpus", type=int, default=10, help="How many GPUs do you want to use at once?")
    argparser.add_argument("--max_cpus", type=int, default=1000, help="How many cpus do you want to use at once?")

    # refinement optionals
    argparser.add_argument("--ref_prefix", type=str, default="", help="Prefix for refinement runs for testing different settings.")
    argparser.add_argument("--ref_cycles", type=int, default=5, help="Number of Rosetta-MPNN-ESM refinement cycles.")
    argparser.add_argument("--ref_input_poses_per_bb", default=None, help="Filter the number of refinement input poses on an input-backbone level. This filter is applied before the ref_input_poses filter.")
    argparser.add_argument("--ref_input_poses", type=int, default=None, help="Maximum number of input poses for refinement cycles after initial RFDiffusion-MPNN-ESM-Rosetta run. Poses will be filtered by design_composite_score. Filter can be applied on a per-input-backbone level if using the flag --ref_input_per_backbone.")
    argparser.add_argument("--ref_num_mpnn_seqs", type=int, default=50, help="Number of sequences that should be created with LigandMPNN during refinement.")
    argparser.add_argument("--ref_catres_bb_rmsd_cutoff_end", type=float, default=0.8, help="End value for catres backbone rmsd filter after each refinement cycle. Filter will be ramped from start to end during refinement.")
    argparser.add_argument("--ref_catres_bb_rmsd_cutoff_start", type=float, default=1.5, help="Start value for catres backbone rmsd filter after each refinement cycle. Filter will be ramped from start to end during refinement.")
    argparser.add_argument("--ref_plddt_cutoff_end", type=float, default=85, help="End value for esm plddt filter after each refinement cycle. Filter will be ramped from start to end during refinement.")
    argparser.add_argument("--ref_plddt_cutoff_start", type=float, default=75, help="Start value for esm plddt filter after each refinement cycle. Filter will be ramped from start to end during refinement.")
    argparser.add_argument("--ref_ligand_rmsd_end", type=float, default=1.8, help="End value for esm plddt filter after each refinement cycle. Filter will be ramped from start to end during refinement.")
    argparser.add_argument("--ref_ligand_rmsd_start", type=float, default=3, help="Start value for esm plddt filter after each refinement cycle. Filter will be ramped from start to end during refinement.")
    argparser.add_argument("--ref_num_cycle_poses", type=int, default=3, help="Number of poses per unique diffusion backbone that should be passed on to the next refinement cycle.")
    argparser.add_argument("--ref_seq_thread_num_mpnn_seqs", type=float, default=10, help="Number of LigandMPNN output sequences during the initial, sequence-threading phase (pre-relax).")
    argparser.add_argument("--ref_input_json", type=str, default=None, help="Read in a poses json file containing input poses for refinement. Screening will be skipped.")

    # screening
    argparser.add_argument("--screen_decentralize_weights", type=str, default="10;20;30;40", help="Decentralize weights that should be tested during screening. Separated by ;. Only used if <model> is 'default'.")
    argparser.add_argument("--screen_decentralize_distances", type=str, default="0;2;4;6", help="Decentralize distances that should be tested during screening. Separated by ;. Only used if <model> is 'default'.")
    argparser.add_argument("--screen_input_poses", type=int, default=20, help="Number of input poses for screening. Will be picked randomly if --screen_from_top is not set.")
    argparser.add_argument("--screen_from_top", action="store_true", help="Instead of picking screening input poses randomly, use poses with best path score.")
    argparser.add_argument("--screen_num_rfdiffusions", type=int, default=1, help="Number of backbones to generate per input path during screening.")
    argparser.add_argument("--screen_mpnn_rlx_mpnn", action="store_true", help="Instead of running LigandMPNN on RFdiffusion output and then directly predict sequences, run a MPNN-RLX-MPNN-ESM trajectory (like in refinement).")
    argparser.add_argument("--screen_substrate_contacts_weight", type=str, default="0", help="Substrate contacts potential weights that should be tested during screening. Separated by ;. Only used if <model> is 'active_site'.")
    argparser.add_argument("--screen_rog_weight", type=str, default="2,3,4", help="Weights for ROG potential that should be tested during screening. Separated by ;. Only used if <model> is 'active_site'.")
    argparser.add_argument("--screen_num_mpnn_sequences", type=int, default=15, help="How many LigandMPNN sequences do you want to design after RFdiffusion?")

    # evaluation
    argparser.add_argument("--eval_prefix", type=str, default=None, help="Prefix for evaluation runs for testing different settings or refinement outputs.")
    argparser.add_argument("--eval_input_json", type=str, default=None, help="Read in a custom poses json containing input poses for evaluation.")
    argparser.add_argument("--eval_input_poses", type=int, default=None, help="Maximum number of input poses for evaluation with AF2 after refinement. Poses will be filtered by design_composite_score.")
    argparser.add_argument("--eval_input_poses_per_bb", type=int, default=5, help="Maximum number of input poses per unique diffusion backbone for evaluation with AF2 after refinement. Poses will be filtered by design_composite_score")

    # variant generation
    argparser.add_argument("--variants_prefix", type=str, default=None, help="Prefix for variant generation runs for testing different variants.")
    argparser.add_argument("--variants_input_json", type=str, default=None, help="Read in a custom json containing poses from evaluation output.")
    argparser.add_argument("--variants_mutations_csv", type=str, default=None, help="Read in a custom csv containing poses description and mutation columns.")
    argparser.add_argument("--variants_num_poses_per_bb", type=int, default=5, help="Read in a custom csv containing poses description and mutation columns.")

    # rfdiffusion optionals
    argparser.add_argument("--as_model_path", type=str, default="/home/mabr3112/RFdiffusion/models/ActiveSite_ckpt.pt")
    argparser.add_argument("--recenter", type=str, default=None, help="Point (xyz) in input pdb towards the diffusion should be recentered. Set strength of recentering with --decentralize_distance. example: --recenter=-13.123;34.84;2.3209")
    argparser.add_argument("--flanking", type=str, default="split", help="How flanking should be set. Always leave on split. nterm or cterm also valid options.")
    argparser.add_argument("--flanker_length", type=int, default=30, help="Set Length of Flanking regions. For active_site model: 30 (recommended at least).")
    argparser.add_argument("--total_length", type=int, default=200, help="Total length of protein to diffuse. This includes flanker, linkers and input fragments.")
    argparser.add_argument("--linker_length", type=str, default="auto", help="linker length, total length. How long should the linkers be, how long should the protein be in total?")
    argparser.add_argument("--rfdiffusion_timesteps", type=int, default=50, help="Specify how many diffusion timesteps to run. 50 recommended. don't change")
    argparser.add_argument("--model", type=str, default="default", help="{default,active_site} Choose which model to use for RFdiffusion (active site or regular model).")
    argparser.add_argument("--channel_contig", type=str, default="Q1-21", help="RFdiffusion-style contig for chain B")

    # ligandmpnn optionals
    argparser.add_argument("--ligandmpnn_options", type=str, default=None, help="Options for ligandmpnn runs.")

    # fastrelax
    argparser.add_argument("--fastrelax_script", type=str, default=f"{protflow.config.AUXILIARY_RUNNER_SCRIPTS_DIR}/fastrelax_sap.xml", help="Specify path to fastrelax script that you would like to use.")
    argparser.add_argument("--bbopt_script", type=str, default="/home/tripp/riff_diff/rosetta/fr_constrained.xml", help="Path to Rosetta xml script used during refinement.")

    # filtering options
    argparser.add_argument("--rfdiffusion_max_clashes", type=int, default=20, help="Filter rfdiffusion output for ligand-backbone clashes before passing poses to LigandMPNN.")
    argparser.add_argument("--rfdiffusion_min_ligand_contacts", type=float, default=7, help="Filter rfdiffusion output for number of ligand contacts (Ca atoms within 8A divided by number of ligand atoms) before passing poses to LigandMPNN.")
    argparser.add_argument("--rfdiffusion_max_rog", type=float, default=19, help="Filter rfdiffusion output for radius of gyration before passing poses to LigandMPNN.")
    argparser.add_argument("--max_rog", type=float, default=18, help="Maximum Radius of Gyration for the backbone to be a successful design.")
    argparser.add_argument("--min_ligand_contacts", type=float, default=3, help="Minimum number of ligand contacts per ligand heavyatom for the design to be a success.")
    #argparser.add_argument("--keep_clashing_backbones", action="store_true", help="Set this flag if you want to keep backbones that clash with the ligand.")
    argparser.add_argument("--ligand_clash_factor", type=float, default=0.8, help="Factor for determining clashes. Set to 0 if ligand clashes should be ignored.")
    #argparser.add_argument("--fpocket_composite_score_weight", type=float, default=0, help="Weight of fpocket pocket score when calculating composite score that is used for filtering. Default is 0.")

    arguments = argparser.parse_args()
    main(arguments)
