try:
    from .model import GPT2, GPT2Config
    from .train_demo import train_demo
except ImportError:
    from model import GPT2, GPT2Config
    from train_demo import train_demo

__all__ = ["GPT2", "GPT2Config", "train_demo"]


if __name__ == "__main__":
    train_demo()
