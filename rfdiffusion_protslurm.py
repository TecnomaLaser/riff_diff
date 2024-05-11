#!/home/mabr3112/anaconda3/envs/protslurm/bin/python
'''
Script to run RFdiffusion active-site model on artificial motif libraries.
'''
import logging
import os
import re

# dependency
import pandas as pd
import matplotlib

# custom
import protslurm
import protslurm.config
from protslurm.jobstarters import SbatchArrayJobstarter
import protslurm.residues
import protslurm.tools
import protslurm.tools.alphafold2
import protslurm.tools.esmfold
import protslurm.tools.ligandmpnn
import protslurm.tools.metrics.rmsd
import protslurm.tools.protein_edits
import protslurm.tools.rfdiffusion
from protslurm.tools.metrics.rmsd import BackboneRMSD, MotifRMSD
import protslurm.tools.rosetta
from protslurm.utils.biopython_tools import renumber_pdb_by_residue_mapping

# local
import protslurm.utils.plotting as plots

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

def update_and_copy_reference_frags(input_df: pd.DataFrame, ref_col:str, desc_col:str, prefix: str, out_pdb_path=None, keep_ligand_chain:str="") -> list[str]:
    '''Updates reference fragments (input_pdbs) to the motifs that were set during diffusion.'''
    # create residue mappings {old: new} for renaming
    list_of_mappings = [protslurm.tools.rfdiffusion.get_residue_mapping(ref_motif, inp_motif) for ref_motif, inp_motif in zip(input_df[f"{prefix}_con_ref_pdb_idx"].to_list(), input_df[f"{prefix}_con_hal_pdb_idx"].to_list())]

    # compile list of output filenames
    output_pdb_names_list = [f"{out_pdb_path}/{desc}.pdb" for desc in input_df[desc_col].to_list()]

    # renumber
    return [renumber_pdb_by_residue_mapping(ref_frag, res_mapping, out_pdb_path=pdb_output, keep_chain=keep_ligand_chain) for ref_frag, res_mapping, pdb_output in zip(input_df[ref_col].to_list(), list_of_mappings, output_pdb_names_list)]

def active_site_pose_opts(input_opt: str, motif: protslurm.residues.ResidueSelection) -> str:
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
    return " ".join(opts_l)

def replace_number_with_10(input_string):
    '''Replaces minimum linker length in rfdiffusion contig (from x-50 to 10-50)'''
    # This regex matches any sequence of digits followed by '-50'
    pattern = r'\d+-50'
    # Replace found patterns with '10-50'
    return re.sub(pattern, '10-50', input_string)

def main(args):
    '''executes everyting (duh)'''
    # logging and checking of inputs
    if not os.path.isdir(args.input_dir):
        raise ValueError(f"Not a directory: {args.input_dir}.")
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"\n{'#'*50}\nRunning rfdiffusion_protslurm.py on {args.input_dir}\n{'#'*50}\n")

    # format path_df to be a DF readable by Poses class
    input_df = pd.read_json(f"{args.input_dir}/selected_paths.json", typ="frame")
    input_df = input_df.reset_index().rename(columns={"index": "poses_description"})
    input_df["poses"] = f"{args.input_dir}/pdb_in/" + input_df["poses_description"] + ".pdb"
    input_df["input_poses"] = input_df["poses"]
    input_df.to_json((path_df := f"{args.output_dir}/paths.poses.json"))

    # load poses
    backbones = protslurm.poses.load_poses(path_df)
    backbones.set_work_dir(args.output_dir)

    # setup jobstarters
    cpu_jobstarter = SbatchArrayJobstarter(max_cores=args.max_cpus)
    small_cpu_jobstarter = SbatchArrayJobstarter(max_cores=10)
    gpu_jobstarter = cpu_jobstarter if args.cpu_only else SbatchArrayJobstarter(max_cores=args.max_gpus, gpus=1)

    # change flanker lengths of rfdiffusion motif contigs
    if args.flanking:
        backbones.df["rfdiffusion_pose_opts"] = [adjust_flanking(rfdiffusion_pose_opts_str, "split", args.flanker_length) for rfdiffusion_pose_opts_str in backbones.df["rfdiffusion_pose_opts"].to_list()]
    elif args.flanker_length:
        raise ValueError(f"Argument 'total_flanker_length' was given, but not 'flanking'! Both args have to be provided.")

    # adjust linkers
    linker_length, total_length = [int(x) for x in args.overwrite_linker_lengths.split(",")]
    backbones.df["rfdiffusion_pose_opts"] = [overwrite_linker_length(pose_opts, total_length, linker_length) for pose_opts in backbones.df["rfdiffusion_pose_opts"].to_list()]

    # convert motifs from dict to ResidueSelection
    backbones.df["fixed_residues"] = [protslurm.residues.from_dict(motif) for motif in backbones.df["fixed_residues"].to_list()]
    backbones.df["motif_residues"] = [protslurm.residues.from_dict(motif) for motif in backbones.df["motif_residues"].to_list()]

    # set motif_cols to keep after rfdiffusion:
    motif_cols = ["fixed_residues"]
    if args.model == "default":
        motif_cols.append("motif_residues")

    # store original motifs for calculation of motif RMSDs later
    backbones.df["template_motif"] = backbones.df["motif_residues"]
    backbones.df["template_fixedres"] = backbones.df["fixed_residues"]

    # setup rfdiffusion options:
    if args.recenter:
        if len(args.recenter.split(";")) != 3:
            raise ValueError(f"--recenter needs to be semicolon separated coordinates. E.g. --recenter=31.123;-12.123;-0.342")
        recenter = f",recenter_xyz:{args.recenter}"
    else:
        recenter = ""

    # change pose_opts according to model being used:
    if args.model == "active_site":
        logging.info("Using Active Site Model. Changing contig strings from pose_options.")
        backbones.df["rfdiffusion_pose_opts"] = [active_site_pose_opts(row["rfdiffusion_pose_opts"], row["template_fixedres"]) for row in backbones]

    # load channel_contig
    backbones.df["rfdiffusion_pose_opts"] = backbones.df["rfdiffusion_pose_opts"].str.replace("contigmap.contigs=[", f"contigmap.contigs=[{args.channel_contig}/0 ")


    # run diffusion
    diffusion_options = f"diffuser.T={str(args.rfdiffusion_timesteps)} potentials.guide_scale=5 inference.num_designs={args.num_rfdiffusions} potentials.guiding_potentials=[\\'type:substrate_contacts,weight:0\\',\\'type:custom_ROG,weight:{args.rog_weight}\\',\\'type:custom_recenter,weight:10,distance:{args.decentralize_distance}{recenter}\\'] potentials.guide_decay=quadratic"
    rfdiffusion = protslurm.tools.rfdiffusion.RFdiffusion(jobstarter = gpu_jobstarter)
    backbones = rfdiffusion.run(
        poses=backbones,
        prefix="rfdiffusion",
        num_diffusions=args.num_rfdiffusions,
        options=diffusion_options,
        pose_options=backbones.df["rfdiffusion_pose_opts"].to_list(),
        update_motifs=motif_cols
    )

    # remove channel chain (chain B)
    chain_remover = protslurm.tools.protein_edits.ChainRemover(jobstarter = small_cpu_jobstarter)
    chain_remover.remove_chains(
        poses = backbones,
        prefix = "channel_removed",
        chains = "B"
    )

    # create updated reference frags:
    if not os.path.isdir((updated_ref_frags_dir := f"{backbones.work_dir}/updated_reference_frags/")):
        os.makedirs(updated_ref_frags_dir)

    backbones.df["updated_reference_frags_location"] = update_and_copy_reference_frags(
        input_df = backbones.df,
        ref_col = "input_poses",
        desc_col = "poses_description",
        prefix = "rfdiffusion",
        out_pdb_path = updated_ref_frags_dir,
        keep_ligand_chain = args.ligand_chain
    )

    rfdiffusion_bb_rmsd = BackboneRMSD(ref_col="rfdiffusion_location", chains="A", jobstarter = small_cpu_jobstarter)
    catres_ca_rmsd = MotifRMSD(ref_col = "updated_reference_frags_location", target_motif = "fixed_residues", ref_motif = "fixed_residues", jobstarter=small_cpu_jobstarter)

    # add back the ligand:
    chain_adder = protslurm.tools.protein_edits.ChainAdder(jobstarter = cpu_jobstarter)
    chain_adder.superimpose_add_chain(
        poses = backbones,
        prefix = "post_rfdiffusion_ligand",
        ref_col = "updated_reference_frags_location",
        target_motif = "fixed_residues",
        copy_chain = args.ligand_chain
    )

    # run LigandMPNN
    ligand_mpnn = protslurm.tools.ligandmpnn.LigandMPNN(jobstarter = gpu_jobstarter)
    backbones = ligand_mpnn.run(
        poses = backbones,
        prefix = "postdiffusion_ligandmpnn",
        nseq = args.num_mpnn_sequences,
        model_type = "ligand_mpnn",
        fixed_res_col = "fixed_residues"
    )

    # predict with ESMFold
    esmfold = protslurm.tools.esmfold.ESMFold(jobstarter = gpu_jobstarter)
    backbones = esmfold.run(
        poses = backbones,
        prefix = "esm",
    )

    # calculate RMSD (backbone, motif, fixedres)
    catres_ca_rmsd.calc_rmsd(poses = backbones, prefix = "esm_catres")
    rfdiffusion_bb_rmsd.calc_rmsd(poses = backbones, prefix = "esm_backbone")

    # run rosetta_script to evaluate residuewiese energy
    rosetta = protslurm.tools.rosetta.Rosetta(jobstarter = cpu_jobstarter)
    rosetta.run(
        poses = backbones,
        prefix = "fastrelax",
        rosetta_application="rosetta_scripts.default.linuxgccrelease",
        nstruct = 5,
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
    # determine pocket-ness!

    # plot outputs
    cols = [
        #"rfdiffusion_catres_rmsd",
        "esm_plddt",
        "esm_backbone_rmsd",
        "esm_catres_heavy_atom_rmsd",
        "fastrelax_total_score"
    ]

    titles = [
        #"RFDiffusion Sidechain\nRMSD",
        "ESMFold pLDDT",
        "ESMFold BB-Ca RMSD",
        "ESMFold Sidechain\nRMSD",
        "Rosetta total_score"
    ]

    y_labels = [
        #"Angstrom",
        "pLDDT",
        "Angstrom",
        "Angstrom",
        "[REU]"
    ]

    dims = [
        #(0,8),
        (0,100),
        (0,8),
        (0,8),
        None
    ]

    # plot results
    plots.violinplot_multiple_cols(
        df = backbones.df,
        cols = cols,
        titles = titles,
        y_labels = y_labels,
        dims = dims,
        out_path = f"{backbones.work_dir}/design_results.png"
    )

    # write pymol alignment script?

if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--input_dir", type=str, required=True, help="input_directory that contains all ensemble *.pdb files to be hallucinated (max 1000 files).")
    argparser.add_argument("--output_dir", type=str, required=True, help="output_directory")

    # general optionals
    argparser.add_argument("--ligand_chain", type=str, default="Z", help="Chain name of the ligand chain.")
    argparser.add_argument("--cpu_only", action="store_true", help="Should only cpu's be used during the entire pipeline run?")

    # jobstarter
    argparser.add_argument("--max_gpus", type=int, default=10, help="How many GPUs do you want to use at once?")
    argparser.add_argument("--max_cpus", type=int, default=1000, help="How many cpus do you want to use at once?")

    # rfdiffusion optionals
    argparser.add_argument("--num_rfdiffusions", type=int, default=10, help="Number of backbones to generate per input path.")
    argparser.add_argument("--recenter", type=str, default=None, help="Point (xyz) in input pdb towards the diffusion should be recentered. Set strength of recentering with --decentralize_distance. example: --recenter=-13.123;34.84;2.3209")
    argparser.add_argument("--decentralize_distance", type=float, default=20, help="Default Distance to decentralize from diffusion center. Default direction of decentralization is away from the substrate.")
    argparser.add_argument("--rog_weight", type=float, default=16, help="Strength of ROG weight of auxiliary potential. Adjust to desired ROG. be aware that it might decrease diffusion performance.")
    argparser.add_argument("--flanking", type=str, default="split", help="How flanking should be set. Always leave on split. nterm or cterm also valid options.")
    argparser.add_argument("--flanker_length", type=int, default=30, help="Set Length of Flanking regions. For active_site model: 30 (recommended at least).")
    argparser.add_argument("--overwrite_linker_lengths", type=str, default="50,200", help="linker length, total length. How long should the linkers be, how long should the protein be in total?")
    argparser.add_argument("--rfdiffusion_timesteps", type=int, default=50, help="Specify how many diffusion timesteps to run. 50 recommended. don't change")
    argparser.add_argument("--model", type=str, default="default", help="{default,active_site} Choose which model to use for RFdiffusion (active site or regular model).")
    argparser.add_argument("--channel_contig", type=str, default="Q1-21", help="RFdiffusion-style contig for chain B")

    # ligandmpnn optionals
    argparser.add_argument("--num_mpnn_sequences", type=int, default=8, help="How many LigandMPNN sequences do you want to design after RFdiffusion?")

    # fastrelax
    argparser.add_argument("--fastrelax_script", type=str, default=f"{protslurm.config.AUXILIARY_RUNNER_SCRIPTS_DIR}/fastrelax_sap.xml", help="Specify path to fastrelax script that you would like to use.")

    arguments = argparser.parse_args()

    main(arguments)
