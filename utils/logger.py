import sys
import logging

def get_logger(name, fname='default', level=logging.DEBUG, is_save=True):
    fmt='%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    datefmt='%Y-%m-%d %H:%M:%S'
    if name == "__main__":
        name = "main"
    else:
        name = "main" + "." + name
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if name != "main":
        return logger
    s_handler = logging.StreamHandler(sys.stdout)
    s_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    s_handler.setLevel(level)
    logger.addHandler(s_handler)
    if is_save:
        filename = fname +'.log'
        f_handler = logging.FileHandler(filename=filename, mode='w')
        f_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        f_handler.setLevel(level)
        logger.addHandler(f_handler)
    return logger
