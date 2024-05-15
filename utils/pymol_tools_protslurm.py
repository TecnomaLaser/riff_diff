#############################
#
#   Tools to present inpainting results with PyMol
#
#
#

# dependencies
import pandas as pd
from protslurm.residues import ResidueSelection

def write_pymol_alignment_script(df:pd.DataFrame, scoreterm: str, top_n:int, path_to_script: str, ascending=True, use_original_location=False,
                                 ref_motif_col: str = "template_motif", target_motif_col: str = "motif_residues",
                                 ref_catres_col: str = "template_fixedres", target_catres_col: str = "fixed_residues"
                                 ) -> str:
    '''
    Writes .pml script for automated pymol alignment.
    '''
    cmds = []
    for index in df.sort_values(scoreterm, ascending=ascending).head(top_n).index:
        cmd = write_align_cmds(
            input_data=df.loc[index],
            use_original_location=use_original_location,
            ref_motif_col=ref_motif_col,
            target_motif_col=target_motif_col,
            ref_catres_col=ref_catres_col,
            target_catres_col=target_catres_col
        )
        cmds.append(cmd)

    with open(path_to_script, 'w', encoding="UTF-8") as f:
        f.write("\n".join(cmds))
    return path_to_script

def pymol_alignment_scriptwriter(df: pd.DataFrame, scoreterm: str, top_n:int, path_to_script: str, ascending=True, pose_col="poses_description", ref_pose_col="input_poses", motif_res_col="motif_residues", fixed_res_col="fixed_residues", ref_motif_res_col="template_motif", ref_fixed_res_col="template_fixedres"):
    ''''''
    top_df = df.sort_values(scoreterm, ascending=ascending).head(top_n)
    cmds = [write_align_cmds_v2(top_df.loc[index], pose_col=pose_col, ref_pose_col=ref_pose_col, motif_res_col=motif_res_col, fixed_res_col=fixed_res_col, ref_motif_res_col=ref_motif_res_col, ref_fixed_res_col=ref_fixed_res_col) for index in top_df.index]

    with open(path_to_script, 'w', encoding = "UTF-8") as f:
        f.write("\n".join(cmds))
    return path_to_script

def write_pymol_motif_selection(obj: str, motif: dict) -> str:
    '''AAA'''
    if isinstance(motif, ResidueSelection):
        motif = motif.to_dict()

    residues = [f"chain {chain} and resi {'+'.join([str(x) for x in res_ids])}" for chain, res_ids in motif.items()]
    pymol_selection = ' or '.join([f"{obj} and {resis}" for resis in residues])
    return pymol_selection

def write_align_cmds(input_data: pd.Series, use_original_location=False, ref_motif_col: str = "template_motif", target_motif_col: str = "motif_residues", ref_catres_col: str = "template_fixedres", target_catres_col: str = "fixed_residues"):
    '''AAA'''
    cmds = list()
    if use_original_location: 
        ref_pose = input_data["input_poses"].replace(".pdb", "")
        pose = input_data["esm_location"]
    else: 
        ref_pose = input_data["input_poses"].split("/")[-1].replace(".pdb", "")
        pose = input_data["poses_description"] + ".pdb"

    # load pose and reference
    cmds.append(f"load {pose}")
    ref_pose_name = input_data['poses_description'] + "_ref"
    cmds.append(f"load {ref_pose}.pdb, {ref_pose_name}")

    # basecolor
    cmds.append(f"color violetpurple, {input_data['poses_description']}")
    cmds.append(f"color yelloworange, {ref_pose_name}")

    # select inpaint_motif residues
    cmds.append(f"select temp_motif_res, {write_pymol_motif_selection(input_data['poses_description'], input_data[target_motif_col])}")
    cmds.append(f"select temp_ref_res, {write_pymol_motif_selection(ref_pose_name, input_data[ref_motif_col])}")

    # alignimpose inpaint_motif_residues:
    cmds.append(f"align temp_ref_res, temp_motif_res")

    # select fixed residues, show sticks and color
    cmds.append(f"select temp_cat_res, {write_pymol_motif_selection(input_data['poses_description'], input_data[target_catres_col])}")
    cmds.append(f"select temp_refcat_res, {write_pymol_motif_selection(ref_pose_name, input_data[ref_catres_col])}")
    cmds.append(f"show sticks, temp_cat_res")
    cmds.append(f"show sticks, temp_refcat_res")
    cmds.append(f"hide sticks, hydrogens")
    cmds.append(f"color atomic, (not elem C)")

    # store scene, delete selection and disable object:
    cmds.append(f"center temp_motif_res")
    cmds.append(f"scene {input_data['poses_description']}, store")
    cmds.append(f"disable {input_data['poses_description']}")
    cmds.append(f"disable {ref_pose_name}")
    cmds.append(f"delete temp_cat_res")
    cmds.append(f"delete temp_refcat_res")
    cmds.append(f"delete temp_motif_res")
    cmds.append(f"delete temp_ref_res")
    return "\n".join(cmds)

def write_align_cmds_v2(input_data: pd.Series, pose_col="poses_description", ref_pose_col="input_poses", motif_res_col="motif_residues", fixed_res_col="fixed_residues", ref_motif_res_col="template_motif", ref_fixed_res_col="template_fixedres"):
    '''AAA'''
    cmds = list()
    ref_pose = input_data[ref_pose_col].split("/")[-1].replace(".pdb", "")
    pose_desc = input_data[pose_col]
    pose = pose_desc + ".pdb"

    # load pose and reference
    cmds.append(f"load {pose}, {pose_desc}")
    ref_pose_name = pose_desc + "_ref"
    cmds.append(f"load {ref_pose}.pdb, {ref_pose_name}")

    # basecolor
    cmds.append(f"color violetpurple, {pose_desc}")
    cmds.append(f"color yelloworange, {ref_pose_name}")

    # select inpaint_motif residues
    cmds.append(f"select temp_motif_res, {write_pymol_motif_selection(input_data[pose_col], input_data[motif_res_col])}")
    cmds.append(f"select temp_ref_res, {write_pymol_motif_selection(ref_pose_name, input_data[ref_motif_res_col])}")

    # alignimpose inpaint_motif_residues:
    cmds.append(f"align temp_ref_res, temp_motif_res")

    # select fixed residues, show sticks and color
    cmds.append(f"select temp_cat_res, {write_pymol_motif_selection(input_data[pose_col], input_data[fixed_res_col])}")
    cmds.append(f"select temp_refcat_res, {write_pymol_motif_selection(ref_pose_name, input_data[ref_fixed_res_col])}")
    cmds.append(f"show sticks, temp_cat_res")
    cmds.append(f"show sticks, temp_refcat_res")
    cmds.append(f"hide sticks, hydrogens")
    cmds.append(f"color atomic, (not elem C)")

    # store scene, delete selection and disable object:
    cmds.append(f"center temp_motif_res")
    cmds.append(f"scene {input_data[pose_col]}, store")
    cmds.append(f"disable {input_data[pose_col]}")
    cmds.append(f"disable {ref_pose_name}")
    cmds.append(f"delete temp_cat_res")
    cmds.append(f"delete temp_refcat_res")
    cmds.append(f"delete temp_motif_res")
    cmds.append(f"delete temp_ref_res")
    return "\n".join(cmds)
