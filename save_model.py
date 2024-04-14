import pickle
import sys

from explore_model import load_run

if __name__ == "__main__":
    wandb_model = sys.argv[1]
    model_path = sys.argv[2]
    config, model = load_run(wandb_model)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
