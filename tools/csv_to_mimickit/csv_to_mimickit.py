import argparse
import pandas as pd
import numpy as np
import sys
import torch
import os
import xml.etree.ElementTree as ET
from typing import List, Optional

# 确保可以将当前目录添加到路径中以导入 mimickit
sys.path.append(os.getcwd())

try:
    from mimickit.anim.motion import Motion, LoopMode
    from mimickit.util.torch_util import quat_to_exp_map
except ImportError:
    print("Error: Could not import 'mimickit'. Please run this script from the root of the MimicKit repository.")
    sys.exit(1)


ROOT_POS_COLS = ["root pos x", "root pos y", "root pos z"]
ROOT_QUAT_COLS = ["root rot x", "root rot y", "root rot z", "root rot w"]


def _read_joint_order_from_txt(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        joints: List[str] = []
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            joints.append(s)
    return joints


def _read_joint_order_from_mjcf_xml(xml_path: str) -> List[str]:
    """
    从 MJCF/XML 里按出现顺序收集 <joint name="...">。
    注意：这里只取 name 且过滤 freejoint root（通常叫 root）。
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    joints: List[str] = []
    for j in root.iter("joint"):
        name = j.attrib.get("name")
        if not name:
            continue
        if name == "root":
            continue
        joints.append(name)
    return joints


def _validate_required_columns(df: pd.DataFrame, cols: List[str], what: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {what} columns in CSV: {missing}")


def convert_csv_to_mimickit(
    csv_file_path: str,
    output_file_path: str,
    loop_mode: str,
    fps: int,
    xml_path: Optional[str] = None,
    joint_order_path: Optional[str] = None,
):
    
    """
    Convert CSV motion data to MimicKit format (.pkl).
    """
    print(f"Reading CSV: {csv_file_path}")
    df = pd.read_csv(csv_file_path)

    _validate_required_columns(df, ROOT_POS_COLS, "root position")
    _validate_required_columns(df, ROOT_QUAT_COLS, "root rotation(quat)")

    # Root pos (3)
    root_pos = df[ROOT_POS_COLS].to_numpy(dtype=np.float32)

    # Root quat -> expmap (3)
    root_rot_quat = df[ROOT_QUAT_COLS].to_numpy(dtype=np.float32)
    root_rot_exp = quat_to_exp_map(torch.tensor(root_rot_quat)).numpy().astype(np.float32)

    # 确定 joint 顺序
    joint_names: List[str] = []
    if joint_order_path:
        joint_names = _read_joint_order_from_txt(joint_order_path)
        if not joint_names:
            raise ValueError(f"Empty joint order file: {joint_order_path}")
    elif xml_path:
        joint_names = _read_joint_order_from_mjcf_xml(xml_path)
        if not joint_names:
            raise ValueError(f"No <joint> found in xml: {xml_path}")
    else:
        # 退化：保持你原来的逻辑，但强烈不推荐
        print("WARNING: no --xml/--joint_order provided; falling back to df.iloc[:, 8:] order.")
        joint_data = df.iloc[:, 8:].to_numpy(dtype=np.float32)
        frames = np.concatenate([root_pos, root_rot_exp, joint_data], axis=1)
        loop_mode_out = LoopMode.WRAP if loop_mode == "wrap" else LoopMode.CLAMP
        motion = Motion(loop_mode=loop_mode_out, fps=fps, frames=frames)
        motion.save(output_file_path)
        print(f"Saved: {output_file_path}, frames={frames.shape}")
        return

    # 用 joint 名称从 CSV 中取列（最关键：重排）
    missing_joint_cols = [j for j in joint_names if j not in df.columns]
    if missing_joint_cols:
        # 给一点辅助信息：列名里可能是 *_joint 或去掉了 _joint
        sample_cols = list(df.columns[:50])
        raise ValueError(
            "Some joint columns are missing in CSV.\n"
            f"Missing({len(missing_joint_cols)}): {missing_joint_cols[:20]}...\n"
            "Hint: check whether CSV joint column names match MJCF joint names exactly.\n"
            f"CSV columns sample: {sample_cols}"
        )

    joint_data = df[joint_names].to_numpy(dtype=np.float32)

    # 拼接 frames: [root_pos(3), root_rot_exp(3), joints(N)]
    frames = np.concatenate([root_pos, root_rot_exp, joint_data], axis=1)

    # loop mode
    if loop_mode == "wrap":
        loop_mode_out = LoopMode.WRAP
    elif loop_mode == "clamp":
        loop_mode_out = LoopMode.CLAMP
    else:
        loop_mode_out = LoopMode.WRAP

    motion = Motion(loop_mode=loop_mode_out, fps=fps, frames=frames)
    motion.save(output_file_path)

    print("=" * 60)
    print("CONVERSION SUCCESSFUL")
    print("=" * 60)
    print(f"Input:  {csv_file_path}")
    print(f"Output: {output_file_path}")
    print(f"Frames Shape: {frames.shape} (Frames x Dofs)")
    print(f"  - Root Pos: {root_pos.shape}")
    print(f"  - Root Rot Exp: {root_rot_exp.shape}")
    print(f"  - Joints: {joint_data.shape} (ordered)")
    print(f"FPS: {fps}")
    print(f"Joint count: {len(joint_names)}")
    print("First 10 joints:", joint_names[:10])
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Convert CSV motion data to MimicKit format (.pkl).")
    parser.add_argument("--input_file", required=True, help="Path to the input CSV file")
    parser.add_argument("--output_file", required=True, help="Path to the output .pkl file")
    parser.add_argument("--fps", type=int, default=24, help="Frame rate of the motion (default: 24)")
    parser.add_argument("--loop", default="wrap", choices=["wrap", "clamp"], help="Loop mode (default: wrap)")

    parser.add_argument("--xml", default=None, help="MJCF/XML path used to infer joint order (recommended).")
    parser.add_argument("--joint_order", default=None, help="Text file containing joint names (one per line). Overrides --xml.")

    args = parser.parse_args()

    convert_csv_to_mimickit(
        args.input_file,
        args.output_file,
        args.loop,
        args.fps,
        xml_path=args.xml,
        joint_order_path=args.joint_order,
    )


if __name__ == "__main__":
    main()

