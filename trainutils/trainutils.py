import os
import json
from collections import defaultdict

from utils.logging import slack_logging
from log import get_logger

logger = get_logger(__name__)


class TrainLog:
    class LogRecord:
        def __init__(self, epoch, num_batches_processed, train_err, valid_err, total_grad_norm, time_elapsed_sec):
            self.time_elapsed_sec = time_elapsed_sec
            self.valid_err = valid_err
            self.train_err = train_err
            self.total_grad = total_grad_norm
            self.num_batches_processed = num_batches_processed
            self.epoch = epoch

        def to_string(self):
            m, s = divmod(self.time_elapsed_sec, 60)
            h, m = divmod(m, 60)
            return 'Epoch{ep:3} Batch{ba:5} Training err. {te:.5f} Valid. err. {ve:.5f}'.format(
                ep=self.epoch, ba=self.num_batches_processed, te=self.train_err,
                ve=self.valid_err) + ' Time elapsed: {h:02d}:{m:02d}:{s:02d}'.format(h=int(h), m=int(m), s=int(s))

    def __init__(self, epochs=None, nums_batches_processed=None, train_errs=None, valid_errs=None, grad_norms=None,
                 times_elapsed_sec=None, batches_per_epoch=None):
        self.times_elapsed_sec = times_elapsed_sec or []
        self.valid_errs = valid_errs or []
        self.train_errs = train_errs or []
        self.grad_norms = grad_norms or []
        self.nums_batches_processed = nums_batches_processed or []
        self.epochs = epochs or []
        self.batches_per_epoch = batches_per_epoch

    def _add_record(self, record: LogRecord):
        self.times_elapsed_sec.append(record.time_elapsed_sec)
        self.epochs.append(record.epoch)
        self.nums_batches_processed.append(record.num_batches_processed)
        self.train_errs.append(record.train_err)
        self.valid_errs.append(record.valid_err)
        self.grad_norms.append(record.total_grad)

    def log_record(self, record, record_logger, log=True, experiment_name=''):
        self._add_record(record)
        if log:
            if logger is None:
                logger.warning('Trying to log a record but no logger is provided')
            else:
                record_logger.info(experiment_name + ': ' + record.to_string())
            if slack_logging.SlackClient is not None:
                slack_logging.send_message(experiment_name[:slack_logging.MAX_CHANNEL_NAME_LENGTH],
                                           slack_logging.generate_train_stats_message(record))

    def get_number_of_records(self):
        return len(self.valid_errs)

    def _logs_to_dict(self):
        result = {
            'times': self.times_elapsed_sec,
            'epochs' : self.epochs,
            'batches': self.nums_batches_processed,
            'train_errs': self.train_errs,
            'valid_errs': self.valid_errs,
            'grad_norms': self.grad_norms,
            'batches_per_epoch': self.batches_per_epoch
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
    def from_json(filename):
        if not os.path.exists(filename):
            logger.warning('Trying to load a log from a non-existing file(will return empty log): {0}'.format(filename))
            return TrainLog()
        with open(filename, 'r') as fp:
            log_dict = json.load(fp)

        log_dict_def = defaultdict(list)
        for k, v in log_dict.items():
            log_dict_def[k] = v

        batches_per_epoch = None
        if 'batches_per_epoch' in log_dict_def:
            batches_per_epoch = log_dict_def['batches_per_epoch']

        return TrainLog(
            epochs=log_dict_def['epochs'],
            nums_batches_processed=log_dict_def['batches'],
            train_errs=log_dict_def['train_errs'],
            valid_errs=log_dict_def['valid_errs'],
            grad_norms=log_dict_def['grad_norms'],
            times_elapsed_sec=log_dict_def['times'],
            batches_per_epoch=batches_per_epoch
        )
