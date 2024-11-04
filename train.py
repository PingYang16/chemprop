from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import pandas as pd
from chemprop import data, featurizers, models, nn

# Load data
train_path = '/work/pi_pengbai_umass_edu/pinyang_umass_edu-conda/chemprop/tests/data/regression/mol/qm9/train.csv'
test_path = '/work/pi_pengbai_umass_edu/pinyang_umass_edu-conda/chemprop/tests/data/regression/mol/qm9/test.csv'
val_path = '/work/pi_pengbai_umass_edu/pinyang_umass_edu-conda/chemprop/tests/data/regression/mol/qm9/val.csv'
num_workers = 0
smiles_column = 'smiles'
target_columns = ['u0_atom']

train_input = pd.read_csv(train_path)
test_input = pd.read_csv(test_path)
val_input = pd.read_csv(val_path)
data_input = pd.concat([train_input, test_input, val_input], axis=0)

smis = data_input.loc[:, smiles_column].values
ys = data_input.loc[:, target_columns].values

all_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, ys)]

mols = [d.mol for d in all_data]
train_indices, val_indices, test_indices = data.make_split_indices(mols, "random", (0.8,0.1,0.1))
train_data, val_data, test_data = data.split_data_by_indices(
    all_data, train_indices, val_indices, test_indices
)

# Featurize data
featurizers = featurizers.SimpleMoleculeMolGraphFeaturizer()

train_dset = data.MoleculeDataset(train_data[0], featurizers)
scaler = train_dset.normalize_targets()
val_dset = data.MoleculeDataset(val_data[0], featurizers)
val_dset.normalize_targets(scaler)
test_dset = data.MoleculeDataset(test_data[0], featurizers)

train_loader = data.build_dataloader(train_dset, num_workers=num_workers)
val_loader = data.build_dataloader(val_dset, num_workers=num_workers, shuffle=False)
test_loader = data.build_dataloader(test_dset, num_workers=num_workers, shuffle=False)

mp = nn.BondMessagePassing()
agg = nn.NormAggregation()
output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
ffn = nn.RegressionFFN(output_transform=output_transform)

batch_norm = True
metric_list = [nn.metrics.RMSEMetric(), nn.metrics.MAEMetric()]

mpnn = models.MPNN(mp, agg, ffn, batch_norm, metric_list)

checkpointing = ModelCheckpoint(
    "checkpoints",
    "best-{epoch}-{val_loss:.2f}",
    "val_loss",
    mode="min",
    save_last=True,
)

trainer = pl.Trainer(
    logger=False,
    enable_checkpointing=True,
    enable_progress_bar=True,
    accelerator="auto",
    devices=1,
    max_epochs=50,
    callbacks=[checkpointing],
)

trainer.fit(mpnn, train_loader, val_loader)

results = trainer.test(dataloaders=test_loader)