import os
import json
from collections import defaultdict

from log import get_logger

logger = get_logger(__name__)


class TrainLog:
    class LogRecord:
        def __init__(self, epoch, num_batches_processed, train_err, train_err_running_avg, valid_err, time_elapsed_sec):
            self.time_elapsed_sec = time_elapsed_sec
            self.valid_err = valid_err
            self.train_err_running_avg = train_err_running_avg
            self.train_err = train_err
            self.num_batches_processed = num_batches_processed
            self.epoch = epoch

        def to_string(self):
            m, s = divmod(self.time_elapsed_sec, 60)
            h, m = divmod(m, 60)
            return 'Epoch{ep:3} Batch{ba:5} Training err. {te:.5f} Training err. RA {tera:.5f} Valid. err. {ve:.5f}'.format(
                ep=self.epoch, ba=self.num_batches_processed, te=self.train_err, tera=self.train_err_running_avg,
                ve=self.valid_err) + ' Time elapsed: {h:02d}:{m:02d}:{s:02d}'.format(h=int(h), m=int(m), s=int(s))

    def __init__(self, epochs=None, nums_batches_processed=None, train_errs=None, train_err_running_avgs=None,
                 valid_errs=None, times_elapsed_sec=None):
        self.times_elapsed_sec = times_elapsed_sec or []
        self.valid_errs = valid_errs or []
        self.train_err_running_avgs = train_err_running_avgs or []
        self.train_errs = train_errs or []
        self.nums_batches_processed = nums_batches_processed or []
        self.epochs = epochs or []

    def _add_record(self, record):
        self.times_elapsed_sec.append(record.time_elapsed_sec)
        self.epochs.append(record.epoch)
        self.nums_batches_processed.append(record.num_batches_processed)
        self.train_errs.append(record.train_err)
        self.train_err_running_avgs.append(record.train_err_running_avg)
        self.valid_errs.append(record.valid_err)

    def log_record(self, record, record_logger, log=True):
        self._add_record(record)
        if log:
            if logger is None:
                logger.warning('Trying to log a record but no logger is provided')
            else:
                record_logger.info(record.to_string())

    def _logs_to_dict(self):
        result = {
            'times': self.times_elapsed_sec,
            'epochs' : self.epochs,
            'batches': self.nums_batches_processed,
            'train_errs': self.train_errs,
            'train_errs_ra': self.train_err_running_avgs,
            'valid_errs': self.valid_errs
        }

        return result

    def dump_to_json(self, filename):
        try:
            if os.path.exists(filename):
                os.remove(filename)
            with open(filename, 'w+') as fp:
                json.dump(self._logs_to_dict(), fp, indent=2)
        except PermissionError:
            logger.warning('Cannot delete or write to {0}'.format(filename))

    @staticmethod
    def from_json(self, filename):
        if not os.path.exists(filename):
            logger.warning('Trying to load a log from a non-existing file(will return empty log): {0}'.format(filename))
            return TrainLog()
        with open(filename, 'r') as fp:
            log_dict = json.load(fp)

        log_dict_def = defaultdict(list)
        for k, v in log_dict.items():
            log_dict_def[k] = v

        return TrainLog(
            epochs=log_dict_def['epochs'],
            nums_batches_processed=log_dict_def['batches'],
            train_errs=log_dict_def['train_errs'],
            train_errs_running_avgs=log_dict_def['train_errs_ra'],
            valid_errs=log_dict_def['valid_errs'],
            times_elapsed_sec=log_dict_def['times']
        )

