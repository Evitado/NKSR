import torch
import nksr
import numpy as np
import typer
import sys
import open3d as o3d
from typing_extensions import Annotated
from pathlib import Path
from pycg import vis

app = typer.Typer()


def load_pcd(filename):
    #  read pcl
    pcd = o3d.io.read_point_cloud(filename)
    # estimate normals
    pcd.estimate_normals()
    return pcd


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@app.command()
def main(
    filename: Annotated[Path, typer.Argument()],
    device: Annotated[str, typer.Argument(default_factory=get_device)],
    detail_level: float = typer.Option(
        1.0,
        "--detail_level",
        "-d",
        help="[Optional] read paper about detail level lies in 0 and 1",
    ),
    visualize: bool = typer.Option(
        False, "--visualize", "-v", help="[Optional] Open an online visualization"
    ),
    save_path: Path = typer.Option(
        1.0,
        "--save_path",
        "-s",
        help="[Optional] path to save",
    ),
):
    """
    provide it with device, by default its on cpu
    """
    if filename is None:
        sys.exit()
    pcd = load_pcd(str(filename.absolute()))
    input_xyz = torch.from_numpy(np.asarray(pcd.points)).float().to(device)
    input_normal = torch.from_numpy(np.asarray(pcd.normals)).float().to(device)
    reconstructor = nksr.Reconstructor(device)
    field = reconstructor.reconstruct(
        input_xyz, input_normal, detail_level=detail_level
    )
    mesh = field.extract_dual_mesh(mise_iter=1)

    if visualize:
        mesh = vis.mesh(mesh.v, mesh.f)
        vis.show_3d([mesh])
    if save_path:
        vis.to_file(mesh, str(save_path))
        print(f"saved file to {str(save_path.absolute())}")


if __name__ == "__main__":
    app()
