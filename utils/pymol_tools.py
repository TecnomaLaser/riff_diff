#############################
#
#   Tools to present inpainting results with PyMol
#
#
#
import pandas as pd

def write_pymol_alignment_script(df:pd.DataFrame, scoreterm: str, top_n:int, path_to_script: str, ascending=True, use_original_location=False) -> str:
    '''
    '''
    cmds = [write_align_cmds(df.loc[index], use_original_location=use_original_location) for index in df.sort_values(scoreterm, ascending=ascending).head(top_n).index]
    
    with open(path_to_script, 'w') as f:
        f.write("\n".join(cmds))
    return path_to_script

def write_pymol_motif_selection(obj: str, motif: dict) -> str:
    '''AAA'''
    residues = [f"chain {chain} and resi {'+'.join([str(x) for x in res_ids])}" for chain, res_ids in motif.items()]
    pymol_selection = ' or '.join([f"{obj} and {resis}" for resis in residues])
    return pymol_selection

def write_align_cmds(input_data: pd.Series, use_original_location=False):
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
    cmds.append(f"select temp_motif_res, {write_pymol_motif_selection(input_data['poses_description'], input_data['motif_residues'])}")
    cmds.append(f"select temp_ref_res, {write_pymol_motif_selection(ref_pose_name, input_data['template_motif'])}")

    # superimpose inpaint_motif_residues:
    cmds.append(f"super temp_ref_res, temp_motif_res")

    # select fixed residues, show sticks and color
    cmds.append(f"select temp_cat_res, {write_pymol_motif_selection(input_data['poses_description'], input_data['fixed_residues'])}")
    cmds.append(f"select temp_refcat_res, {write_pymol_motif_selection(ref_pose_name, input_data['template_fixedres'])}")
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
