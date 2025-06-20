from pathlib import Path

import torch

# override mps backend check - dgl is not compatible.
torch.backends.mps.is_available = lambda: False

import dgl
import nfflr
import alignn.pretrained


class ALIGNNTransform(torch.nn.Module):
    """KShell and line graph transform compatible with reference ALIGNN."""

    def __init__(
        self, neighbor_transform: torch.nn.Module = nfflr.nn.PeriodicKShellGraph()
    ):
        super().__init__()
        self.neighbor_transform = neighbor_transform
        self.atom_embedding = nfflr.nn.layers.atomfeatures._get_attribute_lookup(
            "cgcnn"
        )

    def collate(self, inputs):
        inputs, targets = map(list, zip(*inputs))
        graphs, line_graphs = map(list, zip(*inputs))
        batched_inputs = (dgl.batch(graphs), dgl.batch(line_graphs))
        return batched_inputs, torch.tensor(targets)

    def prepare_batch(self, batch, device=None, non_blocking=False):
        inputs, targets = batch
        graphs, line_graphs = inputs

        batch = (
            (
                graphs.to(device, non_blocking=non_blocking),
                line_graphs.to(device, non_blocking=non_blocking),
                None,  # add an extra item to tuple if alignn expects cell here
            ),
            targets.to(device),
        )
        return batch

    def forward(self, atoms: nfflr.Atoms):
        g = self.neighbor_transform(atoms)
        g.ndata["atom_features"] = self.atom_embedding(g.ndata["atomic_number"])
        lg = dgl.line_graph(g, shared=True)
        lg.apply_edges(nfflr.data.graph.compute_bond_cosines)
        return g, lg


epochs = 50
criterion = torch.nn.MSELoss()
args = nfflr.train.TrainingConfig(
    experiment_dir=Path(__file__).parent.resolve(),
    random_seed=42,
    dataloader_workers=0,
    progress=True,
    criterion=criterion,
    epochs=epochs,
    per_device_batch_size=32,
    weight_decay=1e-2,
    learning_rate=1e-4,
    warmup_steps=1 / epochs,  # 1 epochs warmup
    diskcache=None,
)

# load a pretrained ALIGNN model
transform = ALIGNNTransform()
target = "jv_formation_energy_peratom_alignn"
model = alignn.pretrained.get_figshare_model(model_name=target)
dataset = nfflr.AtomsDataset(
    "dft_3d",
    transform=transform,
    diskcache=args.diskcache,
    custom_collate_fn=transform.collate,
    custom_prepare_batch_fn=transform.prepare_batch,
)
