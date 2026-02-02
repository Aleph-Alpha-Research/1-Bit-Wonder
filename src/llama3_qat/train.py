from torchtitan.train import main, Trainer
import llama3_qat.experiment  # noqa: F401

if __name__ == "__main__":
    main(Trainer)