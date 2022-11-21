import logging 
import os

#https://alexandra-zaharia.github.io/posts/make-your-own-custom-color-formatter-with-python-logging/
class ColoredFormatter(logging.Formatter):
    reset = '\x1b[0m'
    blue = '\x1b[38;5;39m'
    yellow = '\x1b[38;5;226m'
    red = '\x1b[38;5;196m'
    bold_red = '\x1b[31;1m'
    white = '\x1b[38;5;231m'

    def __init__(self, format):
        logging.Formatter.__init__(self, format)
        self.fmt = format

        self.FORMATS = {
            logging.DEBUG: self.blue + self.fmt + self.reset, 
            logging.INFO: self.white + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset, 
            logging.ERROR: self.red + self.fmt + self.reset, 
            logging.WARNING: self.yellow + self.fmt + self.reset
        }



    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class ActivationLogger(object):
    def __init__(self, logger_name, log_level = logging.DEBUG, show_logger_name = True, show_time = False):
        self._logger = logging.getLogger(logger_name)
        self._logger.setLevel(log_level)
        console = logging.StreamHandler()
        console.setLevel(log_level)

        messageFormat = self.setFormatter(show_logger_name, show_time)

        if os.name != 'nt':
            console.setFormatter(ColoredFormatter(messageFormat))
        if os.name == 'nt':
            console.setFormatter(messageFormat)

        self._logger.addHandler(console)


    def debug(self, msg):
        self._logger.debug(msg)

    def warn(self, msg):
        self._logger.warn(msg)

    def info(self, msg):
        self._logger.info(msg)

    def error(self, msg):
        self._logger.error(msg)

    def critical(self, msg):
        self._logger.critical(msg)

    def setFormatter(self, show_logger_name, show_time):
        format = ''

        if show_logger_name:
            format = '%(name)s'

        if show_time:
            if len(format) > 0:
                format = format + ' | '
            
            format = format + '%(asctime)s'

        if len(format) > 0:
            format = format + ' | '

        format = format + '%(filename)s | %(lineno)d | %(message)s'

        return format