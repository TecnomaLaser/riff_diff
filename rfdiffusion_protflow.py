#!/home/tripp/anaconda3/envs/protflow/bin/python3
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
import protflow.metrics.rmsd
import protflow.metrics.tmscore
import protflow.metrics.fpocket
import protflow.tools.protein_edits
import protflow.tools.rfdiffusion
from protflow.metrics.rmsd import BackboneRMSD, MotifRMSD
import protflow.tools.rosetta
from protflow.utils.biopython_tools import renumber_pdb_by_residue_mapping, load_structure_from_pdbfile
import protflow.utils.plotting as plots
from protflow.utils.metrics import calc_rog_of_pdb, calc_ligand_clashes, calc_ligand_contacts

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

def instantiate_trajectory_plotting(plot_dir):
    # instantiate plotting trajectories:
    esm_plddt_traj = plots.PlottingTrajectory(y_label="ESMFold pLDDT", location=os.path.join(plot_dir, "esm_plddt_trajectory.png"), title="ESMFold pLDDT Trajectory", dims=(0,100))
    esm_bb_ca_rmsd_traj = plots.PlottingTrajectory(y_label="RMSD [\u00C5]", location=os.path.join(plot_dir, "esm_bb_ca_trajectory.png"), title="ESMFold BB-Ca\nRMSD Trajectory", dims=(0,5))
    esm_motif_ca_rmsd_traj = plots.PlottingTrajectory(y_label="RMSD [\u00C5]", location=os.path.join(plot_dir, "esm_motif_ca_trajectory.png"), title="ESMFold Motif-Ca\nRMSD Trajectory", dims=(0,5))
    esm_catres_rmsd_traj = plots.PlottingTrajectory(y_label="RMSD [\u00C5]", location=os.path.join(plot_dir, "esm_catres_rmsd_trajectory.png"), title="ESMFold Motif\nSidechain RMSD Trajectory", dims=(0,5))
    fastrelax_total_score_traj = plots.PlottingTrajectory(y_label="Rosetta total score [REU]", location=os.path.join(plot_dir, "rosetta_total_score_trajectory.png"), title="FastRelax Total Score Trajectory")
    postrelax_motif_ca_rmsd_traj = plots.PlottingTrajectory(y_label="RMSD [\u00C5]", location=os.path.join(plot_dir, "postrelax_motif_rmsd_trajectory.png"), title="Refinement Motif\nBB-Ca RMSD Trajectory", dims=(0,5))
    postrelax_motif_catres_rmsd_traj = plots.PlottingTrajectory(y_label="RMSD [\u00C5]", location=os.path.join(plot_dir, "postrelax_motif_catres_trajectory.png"), title="Refinement Motif\nBB-Ca RMSD Trajectory", dims=(0,5))
    delta_apo_holo_traj = plots.PlottingTrajectory(y_label="Rosetta delta total score [REU]", location=os.path.join(plot_dir, "delta_apo_holo_trajectory.png"), title="Delta apo holo total score trajectory")
    
    return {'esm_plddt': esm_plddt_traj, 'esm_backbone_rmsd': esm_bb_ca_rmsd_traj, 'esm_catres_bb_rmsd': esm_motif_ca_rmsd_traj, 'esm_catres_heavy_rmsd': esm_catres_rmsd_traj, 'fastrelax_total_score': fastrelax_total_score_traj, 'postrelax_catres_heavy_rmsd': postrelax_motif_catres_rmsd_traj, 'postrelax_catres_bb_rmsd': postrelax_motif_ca_rmsd_traj, 'delta_apo_holo': delta_apo_holo_traj}

def update_trajectory_plotting(trajectory_plots:dict, df:pd.DataFrame, cycle:int):
    for traj in trajectory_plots:
        trajectory_plots[traj].add_and_plot(df[f"cycle_{cycle}_{traj}"], f"cycle_{cycle}")
    return trajectory_plots

def create_intermediate_ref_results_dir(poses, dir:str, cycle:int):
    # plot outputs and write alignment script

    os.makedirs(dir, exist_ok=True)

    logging.info(f"Plotting outputs of cycle {cycle}.")
    cols = [f"cycle_{cycle}_esm_plddt", f"cycle_{cycle}_esm_backbone_rmsd", f"cycle_{cycle}_esm_catres_heavy_rmsd", f"cycle_{cycle}_fastrelax_total_score", f"cycle_{cycle}_postrelax_top_druggability_score", f"cycle_{cycle}_postrelax_top_volume"]
    titles = ["ESMFold pLDDT", "ESMFold BB-Ca RMSD", "ESMFold Sidechain\nRMSD", "Rosetta total_score", "FPocket\nDruggability", "FPocket\nVolume"]
    y_labels = ["pLDDT", "Angstrom", "Angstrom", "[REU]", "Druggability", "Volume [AU]"]
    dims = [(0,100), (0,5), (0,5), None, None, None]

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
    _ = write_pymol_alignment_script(
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
    cols = ["AF2_plddt", "AF2_backbone_rmsd", "AF2_catres_heavy_rmsd", "final_fastrelax_total_score", "AF2_postrelax_top_druggability_score", "AF2_postrelax_top_volume", "AF2_postrelax_catres_heavy_rmsd", "AF2_postrelax_catres_bb_rmsd", "total_delta_apo_holo"]
    titles = ["AF2 pLDDT", "AF2 BB-Ca RMSD", "AF2 Sidechain\nRMSD", "Rosetta total_score", "FPocket\nDruggability", "FPocket\nVolume", "Relaxed Sidechain\nRMSD", "Relaxed BB-Ca RMSD", "Delta Apo Holo"]
    y_labels = ["pLDDT", "Angstrom", "Angstrom", "[REU]", "Druggability", "Volume [AU]", "Angstrom", "Angstrom", "[REU]"]
    dims = [(0,100), (0,5), (0,5), None, None, None, (0,8), (0,8), None]

    # plot results
    plots.violinplot_multiple_cols(
        dataframe = poses.df,
        cols = cols,
        titles = titles,
        y_labels = y_labels,
        dims = dims,
        out_path = os.path.join(dir, f"refinement_results.png"),
        show_fig = False
    )

    poses.save_poses(out_path=dir)
    poses.save_poses(out_path=dir, poses_col="input_poses")
    poses.save_scores(out_path=dir)

    # write pymol alignment script?
    logging.info(f"Writing pymol alignment script for backbones after refinement at {dir}.")
    _ = write_pymol_alignment_script(
        df = poses.df,
        scoreterm = "final_design_composite_score",
        top_n = np.min([len(poses.df.index), 25]),
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
        

def main(args):
    '''executes everyting (duh)'''
    ################################################# INPUT PREP #########################################################
    # logging and checking of inputs
    if not os.path.isdir(args.input_dir):
        raise ValueError(f"Not a directory: {args.input_dir}.")
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        filename=f'{args.output_dir}/rfdiffusion_protflow_log.txt',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info(f"\n{'#'*50}\nRunning rfdiffusion_protflow.py on {args.input_dir}\n{'#'*50}\n")
    logging.info(f"Min ligand contacts: {args.min_ligand_contacts}")

    # setup empty dictionary for all output metrics that should go into a separate DataFrame:
    output_metrics = {"scTM_success": None, "baker_success": None, "fraction_ligand_clashes": None, "average_ligand_contacts": None, "fraction_ligand_contacts": None}

    # format path_df to be a DF readable by Poses class
    logging.info(f"Parsing inputs specified at {args.input_dir}")
    input_df = pd.read_json(f"{args.input_dir}/selected_paths.json", typ="frame")
    input_df = input_df.reset_index().rename(columns={"index": "poses_description"}) # pylint: disable=E1101
    input_df["poses"] = f"{args.input_dir}/pdb_in/" + input_df["poses_description"] + ".pdb"
    input_df["input_poses"] = input_df["poses"]
    input_df.to_json((path_df := f"{args.output_dir}/paths.poses.json"))

    # load poses
    backbones = protflow.poses.load_poses(path_df)
    backbones.set_work_dir(args.output_dir)

    # setup jobstarters
    cpu_jobstarter = SbatchArrayJobstarter(max_cores=args.max_cpus)
    small_cpu_jobstarter = SbatchArrayJobstarter(max_cores=10)
    gpu_jobstarter = cpu_jobstarter if args.cpu_only else SbatchArrayJobstarter(max_cores=args.max_gpus, gpus=1)
    real_gpu_jobstarter = SbatchArrayJobstarter(max_cores=10, gpus=1)

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

    # set motif_cols to keep after rfdiffusion:
    motif_cols = ["fixed_residues"]
    if args.model == "default":
        motif_cols.append("motif_residues")

    # store original motifs for calculation of motif RMSDs later
    backbones.df["template_motif"] = backbones.df["motif_residues"]
    backbones.df["template_fixedres"] = backbones.df["fixed_residues"]

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

    ############################################## RFDiffusion ######################################################
    # run diffusion
    logging.info(f"Running RFDiffusion on {len(backbones)} poses with {args.num_rfdiffusions} diffusions per pose.")
    if args.model == "active_site":
        diffusion_options = f"diffuser.T={str(args.rfdiffusion_timesteps)} potentials.guide_scale=5 inference.num_designs={args.num_rfdiffusions} potentials.guiding_potentials=[\\'type:substrate_contacts,weight:{args.as_substrate_contacts_weight}\\',\\'type:custom_ROG,weight:{args.rog_weight}\\'] potentials.guide_decay=quadratic"
    else:
        diffusion_options = f"diffuser.T={str(args.rfdiffusion_timesteps)} potentials.guide_scale=5 inference.num_designs={args.num_rfdiffusions} potentials.guiding_potentials=[\\'type:substrate_contacts,weight:0\\',\\'type:custom_recenter_ROG,weight:{args.decentralize_weight},rog_weight:{args.rog_weight},distance:{args.decentralize_distance}{recenter}\\'] potentials.guide_decay=quadratic"
    rfdiffusion = protflow.tools.rfdiffusion.RFdiffusion(jobstarter = gpu_jobstarter)
    backbones = rfdiffusion.run(
        poses=backbones,
        prefix="rfdiffusion",
        num_diffusions=args.num_rfdiffusions,
        options=diffusion_options,
        pose_options=backbones.df["rfdiffusion_pose_opts"].to_list(),
        update_motifs=motif_cols
    )
    num_backbones = len(backbones)

    # remove channel chain (chain B)
    logging.info(f"Diffusion completed, removing channel chain from diffusion outputs.")
    if args.channel_contig != "None":
        chain_remover = protflow.tools.protein_edits.ChainRemover(jobstarter = small_cpu_jobstarter)
        chain_remover.remove_chains(
            poses = backbones,
            prefix = "channel_removed",
            chains = "B"
        )
    else:
        backbones.df["channel_removed_location"] = backbones.df["rfdiffusion_location"]

    # create updated reference frags:
    if not os.path.isdir((updated_ref_frags_dir := f"{backbones.work_dir}/updated_reference_frags/")):
        os.makedirs(updated_ref_frags_dir)

    logging.info(f"Channel chain removeds, now renumbering reference fragments.")
    backbones.df["updated_reference_frags_location"] = update_and_copy_reference_frags(
        input_df = backbones.df,
        ref_col = "input_poses",
        desc_col = "poses_description",
        prefix = "rfdiffusion",
        out_pdb_path = updated_ref_frags_dir,
        keep_ligand_chain = args.ligand_chain
    )

    rfdiffusion_bb_rmsd = BackboneRMSD(ref_col="rfdiffusion_location", chains="A", jobstarter = small_cpu_jobstarter)
    catres_motif_bb_rmsd = MotifRMSD(ref_col = "updated_reference_frags_location", target_motif = "fixed_residues", ref_motif = "fixed_residues", atoms=["N", "CA", "C"], jobstarter=small_cpu_jobstarter)
    catres_motif_heavy_rmsd = MotifRMSD(ref_col = "updated_reference_frags_location", target_motif = "fixed_residues", ref_motif = "fixed_residues", jobstarter=small_cpu_jobstarter)

    # calculate ROG after RFDiffusion, when channel chain is already removed:
    logging.info(f"Calculating rfdiffusion_rog and rfdiffusion_catres_rmsd")
    backbones.df["rfdiffusion_rog"] = [calc_rog_of_pdb(pose) for pose in backbones.poses_list()]

    # calculate motif_rmsd of RFdiffusion (for plotting later)
    catres_motif_bb_rmsd.run(
        poses = backbones,
        prefix = "rfdiffusion_catres",
    )

    # add back the ligand:
    logging.info(f"Metrics calculated, now adding Ligand chain back into backbones.")
    chain_adder = protflow.tools.protein_edits.ChainAdder(jobstarter = cpu_jobstarter)
    chain_adder.superimpose_add_chain(
        poses = backbones,
        prefix = "post_rfdiffusion_ligand",
        ref_col = "updated_reference_frags_location",
        target_motif = "fixed_residues",
        copy_chain = args.ligand_chain
    )

    # calculate ligand stats
    logging.info(f"Calculating Ligand Statistics")
    backbones.df["rfdiffusion_ligand_contacts"] = [calc_ligand_contacts(pose, ligand_chain=args.ligand_chain, min_dist=4, max_dist=8, atoms=["CA"], excluded_elements=["H"]) for pose in backbones.poses_list()]
    backbones.df["rfdiffusion_ligand_clashes"] = [calc_ligand_clashes(pose, ligand_chain=args.ligand_chain, dist=args.ligand_clash_dist) for pose in backbones.poses_list()]

    # collect ligand stats into output metrics:
    output_metrics["average_ligand_contacts"] = float(np.nan_to_num(backbones.df[backbones.df["rfdiffusion_ligand_clashes"] < 1]["rfdiffusion_ligand_contacts"].mean()))
    output_metrics["fraction_ligand_contacts"] = len(backbones.df[(backbones.df["rfdiffusion_ligand_clashes"] < 1) & (backbones.df["rfdiffusion_ligand_contacts"] > args.min_ligand_contacts)]) / num_backbones
    output_metrics["fraction_ligand_clashes"] = len(backbones.df[backbones.df["rfdiffusion_ligand_clashes"] < 1]) / num_backbones

    # plot rfdiffusion_stats
    results_dir = backbones.work_dir + "/results/"
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    plots.violinplot_multiple_cols(
        dataframe = backbones.df,
        cols = ["rfdiffusion_catres_rmsd", "rfdiffusion_rog", "rfdiffusion_ligand_contacts", "rfdiffusion_ligand_clashes", "rfdiffusion_plddt"],
        titles = ["Template RMSD", "ROG", "Ligand Contacts", "Ligand Clashes", "RFdiffusion pLDDT"],
        y_labels = ["RMSD [\u00C5]", "ROG [\u00C5]", "#CA", "#Clashes", "pLDDT"],
        dims = [(0,5), (0,30), None, None, (0.8,1)],
        out_path = f"{results_dir}/rfdiffusion_statistics.png",
        show_fig = False
    )

    ############################################# SEQUENCE DESIGN AND ESMFOLD ########################################################
    # run LigandMPNN
    logging.info(f"Running LigandMPNN on {len(backbones)} poses. Designing {args.num_mpnn_sequences} sequences per pose.")
    ligand_mpnn = protflow.tools.ligandmpnn.LigandMPNN(jobstarter = gpu_jobstarter)
    backbones = ligand_mpnn.run(
        poses = backbones,
        prefix = "postdiffusion_ligandmpnn",
        nseq = args.num_mpnn_sequences,
        options = args.ligandmpnn_options,
        model_type = "ligand_mpnn",
        fixed_res_col = "fixed_residues"
    )

    # predict with ESMFold
    logging.info(f"LigandMPNN finished, now predicting {len(backbones)} sequences using ESMFold.")
    esmfold = protflow.tools.esmfold.ESMFold(jobstarter = real_gpu_jobstarter)
    backbones = esmfold.run(
        poses = backbones,
        prefix = "esm"
    )

    ################################################ METRICS ################################################################
    # calculate RMSDs (backbone, motif, fixedres)
    logging.info(f"Prediction of {len(backbones)} sequences completed. Calculating RMSDs to rfdiffusion backbone and reference fragment.")
    backbones = catres_motif_heavy_rmsd.run(poses = backbones, prefix = "esm_catres_heavy")
    backbones = catres_motif_bb_rmsd.run(poses = backbones, prefix = "esm_catres_bb")
    backbones = rfdiffusion_bb_rmsd.run(poses = backbones, prefix = "esm_backbone")

    # calculate TM-Score and get sc-tm score:
    logging.info(f"Calculating TM-Score between backbone and prediction using TM-Align.")
    tm_score_calculator = protflow.metrics.tmscore.TMalign(jobstarter = small_cpu_jobstarter)
    tm_score_calculator.run(
        poses = backbones,
        prefix = "esm_tm",
        ref_col = "channel_removed_location",
        overwrite = False
    )

    # run rosetta_script to evaluate residuewise energy
    logging.info(f"TMAlign finished. Now relaxing {len(backbones)} structures with Rosetta fastrelax at 5 relax runs per pose.")
    rosetta = protflow.tools.rosetta.Rosetta(jobstarter = cpu_jobstarter)
    rosetta.run(
        poses = backbones,
        prefix = "fastrelax",
        rosetta_application="rosetta_scripts.default.linuxgccrelease",
        nstruct = 5,
        options = f"-parser:protocol {args.fastrelax_script} -beta"
    )
    backbones.df["perresidue_total_score"] = backbones.df["fastrelax_total_score"] / args.total_length

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
    logging.info(f"Rosetta Relax finished. Now adding Ligand back into the structure for ligand-based pocket prediction.")
    chain_adder.superimpose_add_chain(
        poses = backbones,
        prefix = "post_prediction_ligand",
        ref_col = "updated_reference_frags_location",
        target_motif = "fixed_residues",
        copy_chain = args.ligand_chain
    )

    logging.info(f"Detecting pockets using Fpocket on {len(backbones)} backbones.")
    fpocket_runner = protflow.metrics.fpocket.FPocket(jobstarter=cpu_jobstarter)
    fpocket_runner.run(
        poses = backbones,
        prefix = "postrelax",
        options = f"--chain_as_ligand {args.ligand_chain}",
        overwrite = False
    )
    logging.info(f"Fpocket calculations completed.")

    # calculate multi-scorerterm score for the final backbone filter:
    backbones.calculate_composite_score(
        name="design_composite_score",
        scoreterms=["esm_plddt", "esm_tm_TM_score_ref", "esm_catres_bb_rmsd", "esm_catres_heavy_rmsd"],
        weights=[-0.1, -0.2, 0.4, 0.4],
        plot=True
    )

    # filter down to rfdiffusion backbones
    backbones.filter_poses_by_rank(
        n=1,
        score_col="design_composite_score",
        prefix="rfdiffusion_backbone_filter",
        plot=True,
        remove_layers=2
    )

    # plot outputs
    logging.info(f"Plotting outputs.")
    cols = ["rfdiffusion_catres_rmsd", "esm_plddt", "esm_backbone_rmsd", "esm_catres_heavy_rmsd", "fastrelax_total_score", "esm_tm_sc_tm", "postrelax_top_druggability_score", "postrelax_top_volume"]
    titles = ["RFDiffusion Motif\nBackbone RMSD", "ESMFold pLDDT", "ESMFold BB-Ca RMSD", "ESMFold Sidechain\nRMSD", "Rosetta total_score", "SC-TM Score", "FPocket\nDruggability", "FPocket\nVolume"]
    y_labels = ["Angstrom", "pLDDT", "Angstrom", "Angstrom", "[REU]", "TM Score", "Druggability", "Volume [AU]"]
    dims = [(0,8), (0,100), (0,8), (0,8), None, (0,1), None, None]

    # plot results
    plots.violinplot_multiple_cols(
        dataframe = backbones.df,
        cols = cols,
        titles = titles,
        y_labels = y_labels,
        dims = dims,
        out_path = f"{results_dir}/design_results.png",
        show_fig = False
    )

    # fill up remaining output_metrics post-prediction
    output_metrics["scTM_success"] = len(backbones.df[backbones.df["esm_tm_sc_tm"] >= 0.5]) / num_backbones
    baker_success_df = backbones.df[(backbones.df["esm_plddt"] >= 75) & (backbones.df["esm_backbone_rmsd"] <= 1.5) & (backbones.df["esm_catres_bb_rmsd"] <= 1.5)]
    output_metrics["baker_success"] = len(baker_success_df) / num_backbones
    enzyme_success_df = baker_success_df[(baker_success_df["rfdiffusion_ligand_clashes"] < 1) & (baker_success_df["rfdiffusion_ligand_contacts"] > args.min_ligand_contacts) & (baker_success_df["rfdiffusion_rog"] <= args.max_rog)]
    output_metrics["enzyme_success"] = len(baker_success_df[(baker_success_df["rfdiffusion_ligand_clashes"] < 1) & (baker_success_df["rfdiffusion_ligand_contacts"] > args.min_ligand_contacts) & (baker_success_df["rfdiffusion_rog"] <= args.max_rog)]) / num_backbones

    # save output metrics
    output_metrics_str = "\n".join([f"\t{metric}: {value}" for metric, value in output_metrics.items()])
    logging.info(f"num_backbones = {num_backbones}")
    logging.info(f"Finished collection of output metrics.\n{output_metrics_str}\n")
    with open(f"{args.output_dir}/output_metrics.json", 'w', encoding="UTF-8") as f:
        json.dump(output_metrics, f)

    # filter poses to 'baker success' (kind of):
    backbones.filter_poses_by_value(score_col="esm_plddt", value=75, operator=">=")
    backbones.filter_poses_by_value(score_col="esm_backbone_rmsd", value=1.5, operator="<=") # TODO: Better use TM score instead?
    backbones.filter_poses_by_value(score_col="esm_catres_bb_rmsd", value=1.5, operator="<=")
    #num_baker_success = num_backbones / len(backbones)

    # filter poses to 'enzyme success'
    #backbones.filter_poses_by_value(score_col="rfdiffusion_ligand_clashes", value=1, operator="<")
    #backbones.filter_poses_by_value(score_col="rfdiffusion_ligand_contacts", value=args.min_ligand_contacts, operator=">=")
    #backbones.filter_poses_by_value(score_col="rfdiffusion_rog", value=args.max_rog, operator="<=")
    #num_enzyme_success = num_backbones / len(backbones)

    # calculate fraction of (design-successful) backbones where pocket was identified using fpocket.
    pocket_containing_fraction = backbones.df["postrelax_top_volume"].count() / len(backbones)
    logging.info(f"Fraction of RFdiffusion design-successful backbones that contain active-site pocket: {pocket_containing_fraction}")

    backbones.reindex_poses(prefix=f"reindex", remove_layers=3, force_reindex=True)

    # copy filtered poses to new location
    pockets_dir = f"{results_dir}/pocket_pdbs"
    backbones.save_poses(out_path=results_dir)
    backbones.save_poses(out_path=results_dir, poses_col="input_poses")
    backbones.save_scores(out_path=results_dir)

    # save pocket structures
    backbones.save_poses(out_path=pockets_dir, poses_col="postrelax_pocket_location")

    # write pymol alignment script?
    logging.info(f"Created results/ folder and writing pymol alignment script for best backbones at {results_dir}")
    _ = write_pymol_alignment_script(
        df=backbones.df,
        scoreterm="design_composite_score",
        top_n=np.min([len(backbones), 25]),
        path_to_script=f"{results_dir}/align_results.pml",
        ref_motif_col = "template_fixedres",
        ref_catres_col = "template_fixedres",
        target_catres_col = "fixed_residues",
        target_motif_col = "fixed_residues"
    )

    logging.info(f"Writing pymol alignment script for enzyme_success backbones at {results_dir}")
    _ = write_pymol_alignment_script(
        df=enzyme_success_df,
        scoreterm="design_composite_score",
        top_n=len(enzyme_success_df),
        path_to_script=f"{results_dir}/enzyme_success_backbones.pml",
        ref_motif_col = "template_fixedres",
        ref_catres_col = "template_fixedres",
        target_catres_col = "fixed_residues",
        target_motif_col = "fixed_residues"
    )

    plots.violinplot_multiple_cols(
        dataframe = backbones.df,
        cols = cols,
        titles = titles,
        y_labels = y_labels,
        dims = dims,
        out_path = f"{results_dir}/filtered_design_results.png",
        show_fig = False
    )

    if args.skip_refinement:
        logging.info(f"Skipping refinement. Run concluded, output can be found in {results_dir}")
        sys.exit(1)
    
    ############################################# REFINEMENT ########################################################

    params_files = get_params_file(dir=args.input_dir, params=args.params_file)
    if params_files: backbones.df["params_file_path"] = ', '.join(params_files)

    # filter refinement input poses
    if args.filter_ref_input_per_backbone:
        if not args.refinement_input_poses:
            raise ValueError(f'<refinement_input_poses> must be set if filtering refinement input poses on a backbone level!')
        else:
            backbones.filter_poses_by_rank(n=args.refinement_input_poses, score_col='design_composite_score', remove_layers=2, prefix='refinement_input', plot=True)
    elif args.refinement_input_poses:
        backbones.filter_poses_by_rank(n=args.refinement_input_poses, score_col='design_composite_score', prefix='refinement_input', plot=True)

    # instantiate plotting trajectories
    trajectory_plots = instantiate_trajectory_plotting(backbones.plots_dir)

    for cycle in range(1, args.refinement_cycles+1):

        # run ligandmpnn, return pdbs as poses
        backbones = ligand_mpnn.run(
            poses = backbones,
            prefix = f"cycle_{cycle}_seq_thread",
            nseq = 5,
            model_type = "ligand_mpnn",
            options = args.ligandmpnn_options,
            fixed_res_col = "fixed_residues",
            return_seq_threaded_pdbs_as_pose=True
        )

        # optimize backbones 
        backbones.df[f'cycle_{cycle}_bbopt_opts'] = [write_bbopt_opts(row=row, cycle=cycle, total_cycles=args.refinement_cycles, reference_location_col="updated_reference_frags_location", cat_res_col="fixed_residues", motif_res_col="motif_residues", ligand_chain=args.ligand_chain) for _, row in backbones.df.iterrows()]
        options = f"-parser:protocol {args.refinement_script} -beta"
        if params_files: options = options + f" -extra_res_fa {' '.join(params_files)}"

        backbones = rosetta.run(
            poses = backbones,
            prefix = f"cycle_{cycle}_bbopt",
            rosetta_application="rosetta_scripts.default.linuxgccrelease",
            nstruct = 5,
            options = options,
            pose_options=f'cycle_{cycle}_bbopt_opts'
        )

        # filter backbones down to starting backbones
        backbones.filter_poses_by_rank(n=1, score_col=f"cycle_{cycle}_bbopt_total_score", remove_layers=2)

        # run ligandmpnn on relaxed poses
        backbones = ligand_mpnn.run(
            poses = backbones,
            prefix = f"cycle_{cycle}_mpnn",
            nseq = args.ref_num_mpnn_seqs,
            model_type = "ligand_mpnn",
            options = args.ligandmpnn_options,
            fixed_res_col = "fixed_residues",
        )

        # predict structures using ESMFold
        backbones = esmfold.run(
            poses = backbones,
            prefix = f"cycle_{cycle}_esm",
        )

        # copy description column for merging with holo relaxed structures later
        backbones.df[f'cycle_{cycle}_rlx_description'] = backbones.df['poses_description']
        apo_backbones = copy.deepcopy(backbones)

        options = f"-parser:protocol {args.fastrelax_script} -beta"
        apo_backbones = rosetta.run(
            poses = apo_backbones,
            prefix = f"cycle_{cycle}_fastrelax_apo",
            rosetta_application="rosetta_scripts.default.linuxgccrelease",
            nstruct = 5,
            options = options
        )

        apo_backbones.filter_poses_by_rank(n=1, score_col=f"cycle_{cycle}_fastrelax_apo_total_score", remove_layers=1)
        backbones.df = backbones.df.merge(apo_backbones.df[[f'cycle_{cycle}_rlx_description', f"cycle_{cycle}_fastrelax_apo_total_score"]], on=f'cycle_{cycle}_rlx_description')

        backbones = chain_adder.superimpose_add_chain(
            poses = backbones,
            prefix = f"cycle_{cycle}_ligand",
            ref_col = "updated_reference_frags_location",
            target_motif = "fixed_residues",
            copy_chain = args.ligand_chain
        )

        refinement_bb_rmsd = BackboneRMSD(ref_col=f"cycle_{cycle}_bbopt_location", chains="A", jobstarter = small_cpu_jobstarter)
        backbones = catres_motif_heavy_rmsd.run(poses = backbones, prefix = f"cycle_{cycle}_esm_catres_heavy")
        backbones = catres_motif_bb_rmsd.run(poses = backbones, prefix = f"cycle_{cycle}_esm_catres_bb")
        backbones = refinement_bb_rmsd.run(poses = backbones, prefix = f"cycle_{cycle}_esm_backbone")
        backbones.df[f"cycle_{cycle}_ligand_bb_clashes"] = [calc_ligand_clashes(pose, ligand_chain=args.ligand_chain, dist=args.ligand_clash_dist) for pose in backbones.poses_list()]

        # calculate TM-Score and get sc-tm score:
        logging.info(f"Calculating TM-Score between backbone and prediction using TM-Align.")
        backbones = tm_score_calculator.run(
            poses = backbones,
            prefix = f"cycle_{cycle}_esm_tm",
            ref_col = f"cycle_{cycle}_bbopt_location",
        )

        # run rosetta_script to evaluate residuewise energy
        logging.info(f"TMAlign finished. Now relaxing {len(backbones)} structures with Rosetta fastrelax at 5 relax runs per pose.")
        options = f"-parser:protocol {args.fastrelax_script} -beta"
        if params_files: options = options + f" -extra_res_fa {' '.join(params_files)}"
        backbones = rosetta.run(
            poses = backbones,
            prefix = f"cycle_{cycle}_fastrelax",
            rosetta_application="rosetta_scripts.default.linuxgccrelease",
            nstruct = 5,
            options = options
        )

        backbones.df[f'cycle_{cycle}_delta_apo_holo'] = backbones.df[f"cycle_{cycle}_fastrelax_total_score"] - backbones.df[f"cycle_{cycle}_fastrelax_apo_total_score"]

        # calculate RMSD on relaxed poses
        logging.info(f"Relax finished. Now calculating RMSD of catalytic residues for {len(backbones)} structures.")
        backbones = catres_motif_heavy_rmsd.run(poses = backbones, prefix = f"cycle_{cycle}_postrelax_catres_heavy")
        backbones = catres_motif_bb_rmsd.run(poses = backbones, prefix = f"cycle_{cycle}_postrelax_catres_bb")
        backbones.calculate_mean_score(name=f"cycle_{cycle}_postrelax_catres_heavy_mean_rmsd", score_col=f"cycle_{cycle}_postrelax_catres_heavy_rmsd", remove_layers=1)
        backbones.calculate_mean_score(name=f"cycle_{cycle}_postrelax_catres_bb_mean_rmsd", score_col=f"cycle_{cycle}_postrelax_catres_bb_rmsd", remove_layers=1)

        # filter backbones down to relax input backbones
        backbones.filter_poses_by_rank(n=1, score_col=f"cycle_{cycle}_fastrelax_total_score", remove_layers=1)

        backbones.df[f"cycle_{cycle}_perresidue_total_score"] = backbones.df[f"cycle_{cycle}_fastrelax_total_score"] / args.total_length

        backbones = fpocket_runner.run(
            poses = backbones,
            prefix = f"cycle_{cycle}_postrelax",
            options = f"--chain_as_ligand {args.ligand_chain}",
            overwrite = False
        )

        # calculate multi-scoreterm score for the final backbone filter:
        backbones.calculate_composite_score(
            name=f"cycle_{cycle}_refinement_composite_score",
            scoreterms=[f"cycle_{cycle}_esm_plddt", f"cycle_{cycle}_esm_tm_TM_score_ref", f"cycle_{cycle}_esm_catres_bb_rmsd", f"cycle_{cycle}_esm_catres_heavy_rmsd", f'cycle_{cycle}_delta_apo_holo'],
            weights=[-0.1, -0.1, 0.4, 0.6, 0.1],
            plot=True
        )

        # ramp cutoffs during refinement, apply filters
        plddt_cutoff = args.ref_plddt_cutoff_start + (args.ref_plddt_cutoff_end - args.ref_plddt_cutoff_start) * cycle / args.refinement_cycles
        catres_bb_rmsd_cutoff = args.ref_catres_bb_rmsd_cutoff_start - (args.ref_catres_bb_rmsd_cutoff_start - args.ref_catres_bb_rmsd_cutoff_end) * cycle / args.refinement_cycles
        backbones.filter_poses_by_value(score_col=f"cycle_{cycle}_esm_plddt", value=plddt_cutoff, operator=">=", prefix=f"cycle_{cycle}_esm_plddt", plot=True)
        backbones.filter_poses_by_value(score_col=f"cycle_{cycle}_esm_backbone_rmsd", value=1.5, operator="<=", prefix=f"cycle_{cycle}_esm_backbone_rmsd", plot=True)
        backbones.filter_poses_by_value(score_col=f"cycle_{cycle}_esm_catres_bb_rmsd", value=catres_bb_rmsd_cutoff, operator="<=", prefix=f"cycle_{cycle}_esm_catres_bb", plot=True)
        #backbones.filter_poses_by_value(score_col=f"cycle_{cycle}_ligand_bb_clashes", value=1, operator="<", prefix=f"cycle_{cycle}_esm_ligand_bb_clashes", plot=True)

        # filter down to rfdiffusion backbones
        backbones.filter_poses_by_rank(
            n=1 if cycle < args.refinement_cycles else 5,
            score_col=f"cycle_{cycle}_refinement_composite_score",
            prefix=f"cycle_{cycle}_refinement_composite_score",
            plot=True,
            remove_layers=4 if cycle == 1 else 5 # remove one more index layer in subsequent cycles because reindexing adds another one
        )

        backbones.reindex_poses(prefix=f"cycle_{cycle}_reindex", remove_layers=4 if cycle == 1 else 5, force_reindex=True) # remove one more index layer in subsequent cycles because reindexing adds another one
        trajectory_plots = update_trajectory_plotting(trajectory_plots=trajectory_plots, df=backbones.df, cycle=cycle)
        create_intermediate_ref_results_dir(poses=backbones, dir=os.path.join(backbones.work_dir, f"cycle_{cycle}_results"), cycle=cycle)


    ########################### ALPHAFOLD PREDICTION ###########################
    
    backbones.convert_pdb_to_fasta(prefix="final_fasta_conversion", update_poses=True)
    
    colabfold = protflow.tools.colabfold.Colabfold(jobstarter=real_gpu_jobstarter)
    backbones = colabfold.run(
        poses=backbones,
        prefix="AF2",
        return_top_n_poses=5
    )

    backbones.df['final_relax_input_description'] = backbones.df['poses_description']

    apo_backbones = copy.deepcopy(backbones)
    apo_backbones.filter_poses_by_rank(n=1, score_col="AF2_plddt", remove_layers=1)

    backbones = chain_adder.superimpose_add_chain(
        poses = backbones,
        prefix = f"final_ligand",
        ref_col = "updated_reference_frags_location",
        target_motif = "fixed_residues",
        copy_chain = args.ligand_chain
    )

    backbones = catres_motif_heavy_rmsd.run(poses = backbones, prefix = f"AF2_catres_heavy")
    backbones = catres_motif_bb_rmsd.run(poses = backbones, prefix = f"AF2_catres_bb")
    backbones = refinement_bb_rmsd.run(poses = backbones, prefix = f"AF2_backbone")
    #backbones.df[f"final_ligand_bb_clashes"] = [calc_ligand_clashes(pose, ligand_chain=args.ligand_chain, dist=args.ligand_clash_dist, atoms=['N', 'CA', 'C', 'O']) for pose in backbones.poses_list()]
    backbones = tm_score_calculator.run(
        poses = backbones,
        prefix = f"AF2_tm",
        ref_col = f"cycle_{cycle}_bbopt_location",
    )

    backbones.calculate_mean_score(name="AF2_catres_heavy_rmsd_mean", score_col="AF2_catres_heavy_rmsd_mean", remove_layers=1)
    backbones.calculate_mean_score(name="AF2_catres_bb_rmsd_mean", score_col="AF2_catres_bb_rmsd_mean", remove_layers=1)
    backbones.calculate_mean_score(name="AF2_backbone_rmsd_mean", score_col="AF2_backbone_rmsd_mean", remove_layers=1)
    backbones.calculate_mean_score(name="AF2_tm_TM_score_mean", score_col="AF2_tm_TM_score_ref", remove_layers=1)

    backbones.filter_poses_by_rank(n=1, score_col="AF2_plddt", remove_layers=1)

    backbones = rosetta.run(
        poses = backbones,
        prefix = "final_fastrelax",
        rosetta_application="rosetta_scripts.default.linuxgccrelease",
        nstruct = 5,
        options = f"-parser:protocol {args.fastrelax_script} -beta"
    )

    backbones = catres_motif_heavy_rmsd.run(poses = backbones, prefix = f"final_postrelax_catres_heavy")
    backbones = catres_motif_bb_rmsd.run(poses = backbones, prefix = f"final_postrelax_catres_bb")
    backbones.calculate_mean_score(name=f"final_postrelax_catres_heavy_mean_rmsd", score_col=f"final_postrelax_catres_heavy_rmsd", remove_layers=1)
    backbones.calculate_mean_score(name=f"final_postrelax_catres_bb_mean_rmsd", score_col=f"final_postrelax_catres_bb_rmsd", remove_layers=1)

    backbones.filter_poses_by_rank(n=1, score_col="final_fastrelax_total_score", remove_layers=1)

    apo_backbones = rosetta.run(
        poses = apo_backbones,
        prefix = "final_fastrelax_apo",
        rosetta_application="rosetta_scripts.default.linuxgccrelease",
        nstruct = 5,
        options = f"-parser:protocol {args.fastrelax_script} -beta"
    )

    apo_backbones.filter_poses_by_rank(n=1, score_col="final_fastrelax_apo_total_score", remove_layers=1)

    backbones.df = backbones.df.merge(apo_backbones.df[['final_relax_input_description', 'final_fastrelax_apo_total_score']], on='final_relax_input_description')
    
    backbones.df['final_delta_apo_holo'] = backbones.df['final_fastrelax_total_score'] - backbones.df['final_fastrelax_apo_total_score']

    backbones = fpocket_runner.run(
        poses = backbones,
        prefix = f"final_postrelax",
        options = f"--chain_as_ligand {args.ligand_chain}",
        overwrite = False
    )

    backbones.calculate_composite_score(
        name=f"final_design_composite_score",
        scoreterms=[f"AF2_plddt", f"AF2_tm_TM_score_ref", f"AF2_catres_bb_rmsd", f"AF2_catres_heavy_rmsd", "final_delta_apo_holo"],
        weights=[-0.1, -0.1, 0.4, 0.6, 0.1],
        plot=True
    )

    # apply filters only after calculating composite scores!
    backbones.filter_poses_by_value(score_col=f"AF2_plddt", value=args.ref_plddt_cutoff_end, operator=">=", prefix=f"AF2_plddt", plot=True)
    backbones.filter_poses_by_value(score_col=f"AF2_backbone_rmsd", value=1.5, operator="<=")
    backbones.filter_poses_by_value(score_col=f"AF2_catres_bb_rmsd", value=args.ref_catres_bb_rmsd_cutoff_end, operator="<=")
    #backbones.filter_poses_by_value(score_col="final_ligand_bb_clashes", value=1, operator="<")

    backbones = catres_motif_heavy_rmsd.run(poses = backbones, prefix = f"AF2_postrelax_catres_heavy")
    backbones = catres_motif_bb_rmsd.run(poses = backbones, prefix = f"AF2_postrelax_catres_bb")

    backbones.reindex_poses(prefix="final_reindex", remove_layers=2)
    create_final_results_dir(backbones, os.path.join(backbones.work_dir, f"refinement_results"))


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--input_dir", type=str, required=True, help="input_directory that contains all ensemble *.pdb files to be hallucinated (max 1000 files).")
    argparser.add_argument("--output_dir", type=str, required=True, help="output_directory")

    # general optionals
    argparser.add_argument("--skip_refinement", action="store_true", help="Skip refinement, only run RFdiffusion followed by a single run of LigandMPNN, ESMFold and Rosetta Relax.")
    argparser.add_argument("--ligand_chain", type=str, default="Z", help="Chain name of the ligand chain.")
    argparser.add_argument("--cpu_only", action="store_true", help="Should only cpu's be used during the entire pipeline run?")

    # refinement optionals
    argparser.add_argument("--refinement_cycles", type=int, default=5, help="Number of Rosetta-MPNN-ESM refinement cycles.")
    argparser.add_argument("--refinement_input_poses", type=int, default=None, help="Maximum number of input poses for refinement cycles after initial RFDiffusion-MPNN-ESM-Rosetta run. Poses will be filtered by design_composite_score. Filter can be applied on a per-input-backbone level if using the flag --filter_ref_input_per_backbone.")
    argparser.add_argument("--filter_ref_input_per_backbone", action="store_true", help="Filter the number of refinement input poses on an input-backbone level instead of overall filtering.")
    argparser.add_argument("--ref_num_mpnn_seqs", type=int, default=50, help="Maximum number of input poses for refinement cycles after initial RFDiffusion-MPNN-ESM-Rosetta run. Poses will be filtered by design_composite_score. Filter can be applied on a per-input-backbone level if using the flag --filter_ref_input_per_backbone.")
    argparser.add_argument("--refinement_script", type=str, default="/home/tripp/riff_diff/rosetta/fr_constrained.xml", help="Path to Rosetta xml script used during refinement.")
    argparser.add_argument("--params_file", type=str, default=None, help="Path to alternative params file. Can also be multiple paths separated by ';'.")
    argparser.add_argument("--ref_catres_bb_rmsd_cutoff_end", type=float, default=1, help="End value for catres backbone rmsd filter after each refinement cycle. Filter will be ramped from start to end during refinement.")
    argparser.add_argument("--ref_catres_bb_rmsd_cutoff_start", type=float, default=1.5, help="Start value for catres backbone rmsd filter after each refinement cycle. Filter will be ramped from start to end during refinement.")
    argparser.add_argument("--ref_plddt_cutoff_end", type=float, default=85, help="End value for esm plddt filter after each refinement cycle. Filter will be ramped from start to end during refinement.")
    argparser.add_argument("--ref_plddt_cutoff_start", type=float, default=75, help="Start value for esm plddt filter after each refinement cycle. Filter will be ramped from start to end during refinement.")


    # jobstarter
    argparser.add_argument("--max_gpus", type=int, default=10, help="How many GPUs do you want to use at once?")
    argparser.add_argument("--max_cpus", type=int, default=1000, help="How many cpus do you want to use at once?")

    # rfdiffusion optionals
    argparser.add_argument("--as_model_path", type=str, default="/home/mabr3112/RFdiffusion/models/ActiveSite_ckpt.pt")
    argparser.add_argument("--num_rfdiffusions", type=int, default=10, help="Number of backbones to generate per input path.")
    argparser.add_argument("--recenter", type=str, default=None, help="Point (xyz) in input pdb towards the diffusion should be recentered. Set strength of recentering with --decentralize_distance. example: --recenter=-13.123;34.84;2.3209")
    argparser.add_argument("--decentralize_distance", type=float, default=6, help="Default Distance to decentralize from diffusion center. Default direction of decentralization is away from the substrate.")
    argparser.add_argument("--rog_weight", type=float, default=3, help="Strength of ROG weight of auxiliary potential. Adjust to desired ROG. be aware that it might decrease diffusion performance.")
    argparser.add_argument("--flanking", type=str, default="split", help="How flanking should be set. Always leave on split. nterm or cterm also valid options.")
    argparser.add_argument("--flanker_length", type=int, default=30, help="Set Length of Flanking regions. For active_site model: 30 (recommended at least).")
    argparser.add_argument("--total_length", type=int, default=200, help="Total length of protein to diffuse. This includes flanker, linkers and input fragments.")
    argparser.add_argument("--linker_length", type=str, default="auto", help="linker length, total length. How long should the linkers be, how long should the protein be in total?")
    argparser.add_argument("--rfdiffusion_timesteps", type=int, default=50, help="Specify how many diffusion timesteps to run. 50 recommended. don't change")
    argparser.add_argument("--model", type=str, default="default", help="{default,active_site} Choose which model to use for RFdiffusion (active site or regular model).")
    argparser.add_argument("--channel_contig", type=str, default="Q1-21", help="RFdiffusion-style contig for chain B")
    argparser.add_argument("--decentralize_weight", type=float, default=15, help="Weight of decentralization potential for RFdiffusion.")
    argparser.add_argument("--as_substrate_contacts_weight", type=float, default=0, help="Weight of default substrate_contacts potential in RFdiffusion.")

    # ligandmpnn optionals
    argparser.add_argument("--num_mpnn_sequences", type=int, default=8, help="How many LigandMPNN sequences do you want to design after RFdiffusion?")
    argparser.add_argument("--ligandmpnn_options", type=str, default=None, help="Options for ligandmpnn runs.")

    # fastrelax
    argparser.add_argument("--fastrelax_script", type=str, default=f"{protflow.config.AUXILIARY_RUNNER_SCRIPTS_DIR}/fastrelax_sap.xml", help="Specify path to fastrelax script that you would like to use.")

    # filtering options
    argparser.add_argument("--max_rog", type=float, default=18, help="Maximum Radius of Gyration for the backbone to be a successful design.")
    argparser.add_argument("--min_ligand_contacts", type=float, default=3, help="Minimum number of ligand contacts per ligand heavyatom for the design to be a success.")
    argparser.add_argument("--keep_clashing_backbones", action="store_true", help="Set this flag if you want to keep backbones that clash with the ligand.")
    argparser.add_argument("--ligand_clash_dist", type=float, default=3.5, help="Distance threshold to consider a backbone heavyatom - ligand distance a clash.")
    #argparser.add_argument("--fpocket_composite_score_weight", type=float, default=0, help="Weight of fpocket pocket score when calculating composite score that is used for filtering. Default is 0.")

    arguments = argparser.parse_args()
    main(arguments)
