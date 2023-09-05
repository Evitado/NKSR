from common import load_pcd
import torch
import nksr
import numpy as np
from typing import Optional
import typer

app = typer.Typer()


@app.command()
def main(device:str="cpu", detail_level:float=1.0):
    """
    provide it with device, by default its on cpu
    """
    pcd = load_pcd("abc")
    input_xyz = torch.from_numpy(np.asarray(pcd.points)).float().to(device)
    input_normal = torch.from_numpy(np.asarray(pcd.normals)).float().to(device)
    reconstructor = nksr.Reconstructor(device)
    field = reconstructor.reconstruct(input_xyz, input_normal, detail_level=detail_level)
    mesh = field.extract_dual_mesh(mise_iter=1)
    pass


if __name__ == "__main__":
    app()
