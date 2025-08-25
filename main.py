from data_loader import get_generators
from model import create_model
from train import train_model
from evaluate import evaluate_model

DATASET_DIR = "/impacs/sad64/SLURM/dataset/ClassroomActivity"

train_gen, test_gen = get_generators(DATASET_DIR)
model = create_model(25, 224, 224, 10)
history = train_model(model, train_gen, test_gen)
evaluate_model(model, test_gen)
