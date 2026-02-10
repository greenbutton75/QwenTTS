import logging

from .worker import run_loop


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_loop()


if __name__ == "__main__":
    main()
