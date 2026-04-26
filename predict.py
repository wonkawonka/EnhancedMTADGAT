import argparse
import datetime
import json

from args import get_parser, str2bool
from model_factory import build_model
from prediction import Predictor
from utils import *

if __name__ == "__main__":

    parser = get_parser()
    parser.add_argument("--model_id", type=str, default=None,
                        help="ID (datetime) of pretrained model to use, '-1' for latest, '-2' for second latest, etc")
    parser.add_argument("--load_scores", type=str2bool, default=False, help="To use already computed anomaly scores")
    parser.add_argument("--save_output", type=str2bool, default=False)
    args = parser.parse_args()
    print(args)

    dataset = args.dataset
    if args.model_id is None:
        if dataset == 'SMD':
            dir_path = f"./output/{dataset}/{args.group}"
        elif dataset in ['CALCE', 'CALCE2']:
            # For CALCE datasets, check if universal_model directory exists
            base_path = f"./output/{dataset}"
            if os.path.exists(f"{base_path}/universal_model"):
                dir_path = f"{base_path}/universal_model"
            else:
                dir_path = base_path
        else:
            dir_path = f"./output/{dataset}"
        dir_content = os.listdir(dir_path)
        subfolders = [subf for subf in dir_content if os.path.isdir(f"{dir_path}/{subf}") and subf != "logs"]
        # Filter out non-datetime directories like 'universal_model'
        date_times = []
        for subf in subfolders:
            try:
                dt = datetime.datetime.strptime(subf, '%d%m%Y_%H%M%S')
                date_times.append(dt)
            except ValueError:
                # Skip directories that don't match the datetime format
                continue
        if not date_times:
            raise Exception(f"No valid datetime directories found in {dir_path}")
        date_times.sort()
        model_datetime = date_times[-1]
        model_id = model_datetime.strftime('%d%m%Y_%H%M%S')

    else:
        model_id = args.model_id

    if dataset == "SMD":
        model_path = f"./output/{dataset}/{args.group}/{model_id}"
    elif dataset in ['MSL', 'SMAP', 'NASA', 'BMS']:
        model_path = f"./output/{dataset}/{model_id}"
    elif dataset in ['CALCE', 'CALCE2']:
        # For CALCE datasets, check if universal_model directory exists
        base_path = f"./output/{dataset}"
        if os.path.exists(f"{base_path}/universal_model"):
            model_path = f"{base_path}/universal_model/{model_id}"
        else:
            model_path = f"{base_path}/{model_id}"
    else:
        raise Exception(f'Dataset "{dataset}" not available.')

    # Check that model exist
    if not os.path.isfile(f"{model_path}/model.pt"):
        raise Exception(f"<{model_path}/model.pt> does not exist.")

    # Get configs of model
    print(f'Using model from {model_path}')
    model_parser = argparse.ArgumentParser()
    model_args, unknown = model_parser.parse_known_args()
    
    # For CALCE datasets, config is in the main model directory
    model_args_path = f"{model_path}/config.txt"
    with open(model_args_path, "r") as f:
        model_args.__dict__ = json.load(f)
    window_size = model_args.lookback

    # Check that model is trained on specified dataset
    if args.dataset.lower() != model_args.dataset.lower():
        raise Exception(f"Model trained on {model_args.dataset}, but asked to predict {args.dataset}.")

    elif args.dataset == "SMD" and args.group != model_args.group:
        print(f"Model trained on SMD group {model_args.group}, but asked to predict SMD group {args.group}.")

    window_size = model_args.lookback
    normalize = model_args.normalize
    n_epochs = model_args.epochs
    batch_size = model_args.bs
    init_lr = model_args.init_lr
    val_split = model_args.val_split
    shuffle_dataset = model_args.shuffle_dataset
    use_cuda = model_args.use_cuda
    print_every = model_args.print_every
    group_index = model_args.group[0]
    index = model_args.group[2:]
    args_summary = str(model_args.__dict__)

    if dataset == "SMD":
        (x_train, _), (x_test, y_test) = get_data(f"machine-{group_index}-{index}", normalize=normalize)
    elif dataset == "NASA":
        (x_train, _), (x_test, y_test) = get_nasa_battery_data(
            normalize=normalize,
            nasa_battery_id=model_args.nasa_battery_id if hasattr(model_args, "nasa_battery_id") else "",
            nasa_train_batteries=model_args.nasa_train_batteries if hasattr(model_args, "nasa_train_batteries") else "",
            nasa_test_batteries=model_args.nasa_test_batteries if hasattr(model_args, "nasa_test_batteries") else "",
        )
    else:
        (x_train, _), (x_test, y_test) = get_data(args.dataset, normalize=normalize)

    nasa_train_tensors = None
    if dataset == "NASA" and isinstance(x_train, dict):
        nasa_train_tensors = {battery_name: torch.from_numpy(battery_data).float()
                              for battery_name, battery_data in x_train.items()}
        first_train_battery = next(iter(nasa_train_tensors))
        x_train = nasa_train_tensors[first_train_battery]
    else:
        x_train = torch.from_numpy(x_train).float()

    nasa_test_tensors = None
    if dataset == "NASA" and isinstance(x_test, dict):
        nasa_test_tensors = {battery_name: torch.from_numpy(battery_data).float()
                             for battery_name, battery_data in x_test.items()}
        first_test_battery = next(iter(nasa_test_tensors))
        x_test = nasa_test_tensors[first_test_battery]
    else:
        x_test = torch.from_numpy(x_test).float()
    n_features = x_train.shape[1]

    target_dims = get_target_dims(args.dataset)
    if target_dims is None:
        out_dim = n_features
    elif type(target_dims) == int:
        out_dim = 1
    else:
        out_dim = len(target_dims)

    train_dataset = SlidingWindowDataset(x_train, window_size, target_dims)
    test_dataset = SlidingWindowDataset(x_test, window_size, target_dims)

    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, batch_size, val_split, shuffle_dataset, test_dataset=test_dataset
    )

    train_dataset = SlidingWindowDataset(x_train, window_size, target_dims)
    test_dataset = SlidingWindowDataset(x_test, window_size, target_dims)

    model = build_model(model_args, n_features, window_size, out_dim)

    device = "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    load(model, f"{model_path}/model.pt", device=device)
    model.to(device)

    # Some suggestions for POT args
    level_q_dict = {
        "SMAP": (0.90, 0.005),
        "MSL": (0.90, 0.001),
        "SMD-1": (0.9950, 0.001),
        "SMD-2": (0.9925, 0.001),
        "SMD-3": (0.9999, 0.001),
        "NASA": (0.99, 0.001),
        "CALCE": (0.99, 0.001),
        "CALCE2": (0.99, 0.001),
        "BMS": (0.99, 0.001)
    }
    key = "SMD-" + args.group[0] if args.dataset == "SMD" else args.dataset
    level, q = level_q_dict[key]
    if args.level is not None:
        level = args.level
    if args.q is not None:
        q = args.q

    # Some suggestions for Epsilon args
    reg_level_dict = {"SMAP": 0, "MSL": 0, "SMD-1": 1, "SMD-2": 1, "SMD-3": 1, "NASA": 0, "CALCE": 0, "CALCE2": 0, "BMS": 0}
    key = "SMD-" + args.group[0] if dataset == "SMD" else dataset
    reg_level = reg_level_dict[key]

    # For CALCE datasets, we need to determine the test entity to create appropriate save path
    save_path = model_path
    if dataset in ['CALCE', 'CALCE2']:
        # Check if there are entity directories
        entity_dirs = [d for d in os.listdir(model_path) 
                      if os.path.isdir(os.path.join(model_path, d)) and d.isdigit()]
        if entity_dirs:
            # Use the first entity directory for saving results (in predict mode)
            first_entity = sorted(entity_dirs, key=int)[0]
            save_path = os.path.join(model_path, first_entity)

    prediction_args = {
        'dataset': dataset,
        "target_dims": target_dims,
        'scale_scores': args.scale_scores,
        "level": level,
        "q": q,
        'dynamic_pot': args.dynamic_pot,
        "use_mov_av": args.use_mov_av,
        "gamma": args.gamma,
        "reg_level": reg_level,
        "save_path": save_path,
    }

    # Creating a new summary-file each time when new prediction are made with a pre-trained model
    count = 0
    for filename in os.listdir(save_path):
        if filename.startswith("summary"):
            count += 1
    if count == 0:
        summary_file_name = "summary.txt"
    else:
        summary_file_name = f"summary_{count}.txt"

    if dataset == "NASA" and nasa_test_tensors is not None:
        train_reference = nasa_train_tensors if nasa_train_tensors is not None else x_train
        for battery_name, battery_tensor in nasa_test_tensors.items():
            battery_save_path = save_path if len(nasa_test_tensors) == 1 else os.path.join(save_path, f"battery_{battery_name}")
            if len(nasa_test_tensors) > 1:
                os.makedirs(battery_save_path, exist_ok=True)

            battery_prediction_args = dict(prediction_args)
            battery_prediction_args["save_path"] = battery_save_path

            battery_summary_count = 0
            for filename in os.listdir(battery_save_path):
                if filename.startswith("summary"):
                    battery_summary_count += 1
            if battery_summary_count == 0:
                battery_summary_name = "summary.txt"
            else:
                battery_summary_name = f"summary_{battery_summary_count}.txt"

            predictor = Predictor(model, window_size, n_features, battery_prediction_args, summary_file_name=battery_summary_name)
            battery_label = None
            if isinstance(y_test, dict):
                raw_label = y_test.get(battery_name)
                if raw_label is not None:
                    battery_label = raw_label[window_size:]

            predictor.predict_anomalies(
                train_reference,
                battery_tensor,
                battery_label,
                load_scores=args.load_scores,
                save_output=args.save_output,
            )
    else:
        label = y_test[window_size:] if y_test is not None else None
        predictor = Predictor(model, window_size, n_features, prediction_args, summary_file_name=summary_file_name)
        predictor.predict_anomalies(x_train, x_test, label,
                                    load_scores=args.load_scores,
                                    save_output=args.save_output)
