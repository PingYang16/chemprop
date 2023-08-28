from argparse import ArgumentError, ArgumentParser, Namespace
import logging
from pathlib import Path
import sys
import warnings

from lightning import pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import torch

from chemprop.v2 import data
from chemprop.v2.data.utils import split_data
from chemprop.v2.models import MetricRegistry
from chemprop.v2.featurizers.utils import ReactionMode
from chemprop.v2.models.loss import LossFunctionRegistry
from chemprop.v2.models.model import MPNN
from chemprop.v2.models.modules.agg import AggregationRegistry

from chemprop.v2.cli.utils import Subcommand, RegistryAction
from chemprop.v2.cli.utils_ import build_data_from_files, make_dataset
from chemprop.v2.models.modules.message_passing.molecule import AtomMessageBlock, BondMessageBlock
from chemprop.v2.models.modules.readout import ReadoutRegistry, RegressionFFN
from chemprop.v2.utils.registry import Factory

logger = logging.getLogger(__name__)


class TrainSubcommand(Subcommand):
    COMMAND = "train"
    HELP = "train a chemprop model"

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> ArgumentParser:
        return add_args(parser)

    @classmethod
    def func(cls, args: Namespace):
        process_args(args)
        main(args)


def add_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "-i",
        "--input",
        "--data-path",
        help="path to an input CSV containing SMILES and associated target values",
    )
    parser.add_argument("-o", "--output-dir")
    parser.add_argument(
        "--logdir",
        nargs="?",
        const="chemprop_logs",
        help="runs will be logged to {logdir}/chemprop_{time}.log. If unspecified, will use 'output_dir'. If only the flag is given (i.e., '--logdir'), then will write to 'chemprop_logs'",
    )

    mp_args = parser.add_argument_group("message passing")
    mp_args.add_argument(
        "--message-hidden-dim", type=int, default=300, help="hidden dimension of the messages"
    )
    mp_args.add_argument(
        "--message-bias", action="store_true", help="add bias to the message passing layers"
    )
    mp_args.add_argument(
        "--depth", type=int, default=3, help="the number of message passing layers to stack"
    )
    mp_args.add_argument(
        "--undirected", action="store_true", help="pass messages on undirected bonds"
    )
    mp_args.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="dropout probability in message passing/FFN layers",
    )
    mp_args.add_argument(
        "--activation", default="relu", help="activation function in message passing/FFN layers"
    )
    mp_args.add_argument(
        "--aggregation",
        "--agg",
        default="mean",
        choices=RegistryAction(AggregationRegistry),
        help="the aggregation mode to use during graph readout",
    )
    mp_args.add_argument(
        "--norm", type=float, default=100, help="normalization factor to use for 'norm' aggregation"
    )
    mp_args.add_argument(
        "--atom-messages", action="store_true", help="pass messages on atoms rather than bonds"
    )

    ffn_args = parser.add_argument_group("FFN args")
    ffn_args.add_argument(
        "--ffn-hidden-dim", type=int, default=300, help="hidden dimension in the FFN top model"
    )
    ffn_args.add_argument(
        "--ffn-num-layers", type=int, default=1, help="number of layers in FFN top model"
    )

    exta_mpnn_args = parser.add_argument_group("extra MPNN args")
    exta_mpnn_args.add_argument(
        "--num-classes", type=int, help="the number of classes to predict in multiclass settings"
    )
    exta_mpnn_args.add_argument("--spectral-activation", default="exp", choices=["softplus", "exp"])

    data_args = parser.add_argument_group("input data parsing args")
    data_args.add_argument(
        "-t", "--task", default="regression", action=RegistryAction(ReadoutRegistry)
    )
    data_args.add_argument(
        "--no-header-row", action="store_true", help="if there is no header in the input data CSV"
    )
    data_args.add_argument(
        "-s",
        "--smiles-columns",
        nargs="+",
        type=int,
        default=[0],
        help="the columns in the CSV containing the SMILES strings of the inputs",
    )
    data_args.add_argument(
        "-t",
        "--target-columns",
        nargs="+",
        type=int,
        default=[1],
        help="the columns in the CSV containing the target values of the inputs",
    )

    data_args.add_argument(
        "--rxn-idxs",
        nargs="+",
        type=int,
        default=list(),
        help="the indices in the input SMILES containing reactions. Unless specified, each input is assumed to be a molecule. Should be a number in `[0, N)`, where `N` is the number of `--smiles-columns` specified",
    )
    data_args.add_argument("--phase-mask-path")
    data_args.add_argument(
        "--data-weights-path",
        help="a plaintext file that is parallel to the input data file and contains a single float per line that corresponds to the weight of the respective input weight during training.",
    )
    data_args.add_argument("--val-path")
    data_args.add_argument("--val-features-path")
    data_args.add_argument("--val-atom-features-path")
    data_args.add_argument("--val-bond-features-path")
    data_args.add_argument("--val-atom-descriptors-path")
    data_args.add_argument("--test-path")
    data_args.add_argument("--test-features-path")
    data_args.add_argument("--test-atom-features-path")
    data_args.add_argument("--test-bond-features-path")
    data_args.add_argument("--test-atom-descriptors-path")

    featurization_args = parser.add_argument_group("featurization args")
    featurization_args.add_argument("--rxn-mode", choices=ReactionMode.choices, default="reac_diff")
    featurization_args.add_argument(
        "--atom-features-path",
        help="the path to a .npy file containing a _list_ of `N` 2D arrays, where the `i`th array contains the atom features for the `i`th molecule in the input data file. NOTE: each 2D array *must* have correct ordering with respect to the corresponding molecule in the data file. I.e., row `j` contains the atom features of the `j`th atom in the molecule.",
    )
    featurization_args.add_argument(
        "--bond-features-path",
        help="the path to a .npy file containing a _list_ of `N` arrays, where the `i`th array contains the bond features for the `i`th molecule in the input data file. NOTE: each 2D array *must* have correct ordering with respect to the corresponding molecule in the data file. I.e., row `j` contains the bond features of the `j`th bond in the molecule.",
    )
    featurization_args.add_argument(
        "--atom-descriptors-path",
        help="the path to a .npy file containing a _list_ of `N` arrays, where the `i`th array contains the atom descriptors for the `i`th molecule in the input data file. NOTE: each 2D array *must* have correct ordering with respect to the corresponding molecule in the data file. I.e., row `j` contains the atom descriptors of the `j`th atom in the molecule.",
    )
    featurization_args.add_argument("--features-generators", nargs="+")
    featurization_args.add_argument("--features-path")
    featurization_args.add_argument("--explicit-h", action="store_true")
    featurization_args.add_argument("--add-h", action="store_true")

    train_args = parser.add_argument_group("training args")
    train_args.add_argument("-b", "--batch-size", type=int, default=64)
    train_args.add_argument("--target-weights", type=float, nargs="+")
    train_args.add_argument("-l", "--loss-function", action=RegistryAction(LossFunctionRegistry))
    train_args.add_argument(
        "--v-kl", type=float, default=0.2, help="evidential/dirichlet regularization term weight"
    )
    train_args.add_argument(
        "--eps", type=float, default=1e-8, help="evidential regularization epsilon"
    )
    train_args.add_argument("-T", "--threshold", type=float, help="spectral threshold limit")
    train_args.add_argument(
        "--metrics",
        nargs="+",
        choices=RegistryAction(MetricRegistry),
        help="evaluation metrics. If unspecified, will use the following metrics for given dataset types: regression->rmse, classification->roc, multiclass->ce ('cross entropy'), spectral->sid. If multiple metrics are provided, the 0th one will be used for early stopping and checkpointing",
    )
    train_args.add_argument(
        "-tw",
        "--task-weights",
        nargs="+",
        type=float,
        help="the weight to apply to an individual task in the overall loss",
    )
    train_args.add_argument("--warmup-epochs", type=int, default=2)
    train_args.add_argument("--num-lrs", type=int, default=1)
    train_args.add_argument("--init-lr", type=float, default=1e-4)
    train_args.add_argument("--max-lr", type=float, default=1e-3)
    train_args.add_argument("--final-lr", type=float, default=1e-4)

    parser.add_argument("--epochs", type=int, default=30, help="the number of epochs to train over")

    parser.add_argument("--split", "--split-type", default="random")
    parser.add_argument("--split-sizes", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    parser.add_argument("-k", "--num-folds", type=int, default=1)
    parser.add_argument("--save-splits", action="store_true")

    parser.add_argument("-g", "--n-gpu", type=int, default=1, help="the number of GPU(s) to use")
    parser.add_argument(
        "-c",
        "--n-cpu",
        "--num-workers",
        type=int,
        default=0,
        help="the number of CPUs over which to parallelize data loading",
    )

    return parser


def process_args(args: Namespace):
    args.input = Path(args.input)
    args.output_dir = Path(args.output_dir or Path.cwd() / args.input.stem)
    args.logdir = Path(args.logdir or args.output_dir / "logs")

    args.output_dir.mkdir(exist_ok=True, parents=True)
    args.logdir.mkdir(exist_ok=True, parents=True)


def validate_args(args):
    pass


def main(args):
    bond_messages = not args.atom_messages
    n_components = len(args.smiles_columns)
    n_tasks = len(args.target_columns)
    bounded = args.loss_function is not None and "bounded" in args.loss_function

    if n_components > 1:
        warnings.warn(
            "Multicomponent input is not supported at this time! Using only the 1st input..."
        )

    format_kwargs = dict(
        no_header_row=args.no_header_row,
        smiles_columns=args.smiles_columns,
        target_columns=args.target_columns,
        bounded=bounded,
    )
    featurization_kwargs = dict(
        features_generators=args.features_generators,
        explicit_h=args.explicit_h,
        add_h=args.add_h,
        reaction=0 in args.rxn_idxs,
    )

    all_data = build_data_from_files(
        args.input,
        **format_kwargs,
        p_features=args.features_path,
        p_atom_feats=args.atom_features_path,
        p_bond_feats=args.bond_features_path,
        p_atom_descs=args.atom_descriptors_path,
        **featurization_kwargs,
    )

    if args.val_path is None and args.test_path is None:
        train_data, val_data, test_data = split_data(all_data, args.split, args.split_sizes)
    elif args.test_path is not None:
        test_data = build_data_from_files(
            args.test_path,
            p_features=args.test_features_path,
            p_atom_feats=args.test_atom_features_path,
            p_bond_feats=args.test_bond_features_path,
            p_atom_descs=args.test_atom_descriptors_path,
            **format_kwargs,
            **featurization_kwargs,
        )
        if args.val_path is not None:
            val_data = build_data_from_files(
                args.val_path,
                p_features=args.val_features_path,
                p_atom_feats=args.val_atom_features_path,
                p_bond_feats=args.val_bond_features_path,
                p_atom_descs=args.val_atom_descriptors_path,
                **format_kwargs,
                **featurization_kwargs,
            )
            train_data = all_data
        else:
            train_data, val_data, _ = split_data(all_data, args.split, args.split_sizes)
    else:
        raise ArgumentError("'val_path' must be specified is 'test_path' is provided!")
    logger.info(f"train/val/test sizes: {len(train_data)}/{len(val_data)}/{len(test_data)}")

    train_dset = make_dataset(train_data, bond_messages, args.rxn_mode)
    val_dset = make_dataset(val_data, bond_messages, args.rxn_mode)

    mp_cls = BondMessageBlock if bond_messages else AtomMessageBlock
    mp_block = mp_cls(
        train_dset.featurizer.atom_fdim,
        train_dset.featurizer.bond_fdim,
        args.message_hidden_dim,
        args.message_bias,
        args.depth,
        args.undirected,
        args.dropout,
        args.activation,
    )
    agg = Factory.build(AggregationRegistry[args.aggregation], norm=args.norm)
    readout_cls = ReadoutRegistry[args.readout]

    if args.loss_function is not None:
        criterion = Factory.build(
            LossFunctionRegistry[args.loss_function],
            v_kl=args.v_kl,
            threshold=args.threshold,
            eps=args.eps,
        )
    else:
        logger.info(
            f"No loss function specified, will use class default: {readout_cls._default_criterion}"
        )
        criterion = readout_cls._default_criterion

    readout_ffn = Factory.build(
        readout_cls,
        input_dim=mp_block.output_dim,
        n_tasks=args.n_tasks,
        hidden_dim=args.ffn_hidden_dim,
        n_layers=args.ffn_num_layers,
        dropout=args.dropout,
        activation=args.activation,
        criterion=criterion,
        num_classes=args.num_classes,
        spectral_activation=args.spectral_activation,
    )

    if isinstance(readout_ffn, RegressionFFN):
        scaler = train_dset.normalize_targets()
        val_dset.normalize_targets(scaler)
        logger.info(f"Train data: loc = {scaler.mean_}, scale = {scaler.scale_}")
    else:
        scaler = None

    train_loader = data.MolGraphDataLoader(train_dset, args.batch_size, args.n_cpu)
    val_loader = data.MolGraphDataLoader(val_dset, args.batch_size, args.n_cpu, shuffle=False)
    if len(test_data) > 0:
        test_dset = make_dataset(test_data, bond_messages, args.rxn_mode)
        test_loader = data.MolGraphDataLoader(test_dset, args.batch_size, args.n_cpu, shuffle=False)
    else:
        test_loader = None

    model = MPNN(
        mp_block,
        agg,
        readout_ffn,
        None,
        args.task_weights,
        args.warmup_epochs,
        args.num_lrs,
        args.init_lr,
        args.max_lr,
        args.final_lr,
    )
    logger.info(model)

    monitor_mode = "min" if model.metrics[0].minimize else "max"
    logger.debug(f"Evaluation metric: '{model.metrics[0].alias}', mode: '{monitor_mode}'")

    tb_logger = TensorBoardLogger(args.output_dir, "tb_logs")
    checkpointing = ModelCheckpoint(
        args.output_dir / "chkpts",
        "{epoch}-{val_loss:.2f}",
        "val_loss",
        mode=monitor_mode,
        save_last=True,
    )
    early_stopping = EarlyStopping("val_loss", patience=5, mode=monitor_mode)

    trainer = pl.Trainer(
        logger=tb_logger,
        enable_progress_bar=True,
        accelerator="auto",
        devices=args.n_gpu if torch.cuda.is_available() else 1,
        max_epochs=args.epochs,
        callbacks=[checkpointing, early_stopping],
    )
    trainer.fit(model, train_loader, val_loader)

    if test_loader is not None:
        if args.dataset_type == "regression":
            model.loc, model.scale = float(scaler.mean_), float(scaler.scale_)
        results = trainer.test(model, test_loader)[0]
        logger.info(f"Test results: {results}")

    p_model = args.output / "model.pt"
    torch.save(model.state_dict(), p_model)
    logger.info(f"model state dict saved to '{p_model}'")


if __name__ == "__main__":
    parser = ArgumentParser()
    add_args(parser)

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)
    args = parser.parse_args()
    process_args(args)

    main(args)