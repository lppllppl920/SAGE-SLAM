import logging

logger = logging.getLogger('sage-slam')
formatter = logging.Formatter(
    fmt='[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s', datefmt='%m-%d %H:%M:%S')
