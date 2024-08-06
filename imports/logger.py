import logging
import os

def configure_logging(log_level=logging.INFO):
    """Configures logging to a file and console."""
    log_file_name = "aif56.log"  # Set the log file name
    log_format = '%(asctime)s - %(levelname)s - %(filename)s - %(lineno)d - %(message)s'
    logging.basicConfig(
        filename=log_file_name,
        level=log_level,
        format=log_format,
        filemode='w'  # Overwrite the log file on each run
    )
    # Also log to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(console_handler)