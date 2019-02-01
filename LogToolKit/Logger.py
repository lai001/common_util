import functools
import logging
import os


def singleton(cls):
    """ Use class as singleton. """

    cls.__new_original__ = cls.__new__

    @functools.wraps(cls.__new__)
    def singleton_new(cls, *args, **kw):
        it = cls.__dict__.get('__it__')
        if it is not None:
            return it

        cls.__it__ = it = cls.__new_original__(cls, *args, **kw)
        it.__init_original__(*args, **kw)
        return it

    cls.__new__ = singleton_new
    cls.__init_original__ = cls.__init__
    cls.__init__ = object.__init__

    return cls


class Level:
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@singleton
class Logger:

    def __init__(self, file_path: str = os.path.join(os.getcwd(), 'Python_All.log'), level: Level = Level.DEBUG,
                 format: str = '%(asctime)s - %(levelname)s: \n%(message)s\n', LOG_DIR_PATH: str = os.getcwd()):
        if not os.path.exists(LOG_DIR_PATH):
            os.mkdir(LOG_DIR_PATH)
        self.__logger = None
        self.__logger = logging.getLogger(file_path)
        self.__logger.setLevel(level)
        format = logging.Formatter(format)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(format)
        file_handler = logging.FileHandler(file_path, encoding='utf-8')
        file_handler.setFormatter(format)
        self.__logger.addHandler(stream_handler)
        self.__logger.addHandler(file_handler)

    def debug(self, mes: str):
        self.__logger.debug(mes)

    def info(self, mes: str):
        self.__logger.info(mes)

    def warning(self, mes: str):
        self.__logger.warning(mes)

    def error(self, mes: str):
        self.__logger.error(mes)

    def critical(self, mes: str):
        self.__logger.critical(mes)


logger = Logger()
