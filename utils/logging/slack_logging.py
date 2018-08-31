import os
import json
import traceback
from datetime import timedelta

from typing import Optional, Dict, List, NewType
from slackclient import SlackClient

from models.utils import RNN_Hyperparameters

from log import get_logger

logger = get_logger(__name__)


Username = NewType('Username', str)
ChannelName = NewType('ChannelName', str)

UserId = NewType('UserId', str)
ChannelId = NewType('ChannelId', str)


# Global Slack client
g_slack_client: SlackClient = None

# Map from Slack channel name to ID
g_channel_map: Dict[ChannelName, ChannelId] = dict()

# All of the users in the workspace
g_user_ids: List[UserId]

# If > 1, log every time stats are logged, otherwise only at end of epoch
g_verbosity: int = 1


def init_slack_client(verbosity:int=1):
    global g_slack_client, g_channel_map, g_verbosity, g_user_ids

    if 'SLACK_API_TOKEN' not in os.environ:
        logger.warning(f'Cannot initialize Slack client -- SLACK_API_TOKEN env var not present')
        return
    try:
        g_slack_client = SlackClient(os.environ['SLACK_API_TOKEN'])
    except Exception as e:
        logger.warning(f'Cannot initialize Slack client: {str(e)}')

    g_channel_map = _get_available_channels()
    g_user_ids = _get_all_user_ids()
    g_verbosity = verbosity


def _get_available_channels() -> Dict[ChannelName, ChannelId]:
    if g_slack_client is not None:
        resp = g_slack_client.api_call('channels.list')
        if not resp['ok']:
            logger.warning(f'Failed to retreive slack channels list: {resp["error"]}')
            return {}
        return dict((channel['name'], channel['id']) for channel in resp['channels'])


def _get_all_user_ids() -> List[UserId]:
    if g_slack_client is not None:
        try:
            resp = g_slack_client.api_call('users.list')
            if not resp['ok']:
                logger.warning(f'Failed to retreive slack users list: {resp["error"]}')
                return []
            return [entry['id'] for entry in resp['members']]
        except Exception as e:
            logger.warning(f'Something went wrong while fetching user list: {str(e)}')


def create_channel(channel_name: str, channel_topic: Optional[str] = None) -> None:
    if g_slack_client is not None:
        channel_id = None
        if channel_name in g_channel_map:
            channel_id = g_channel_map[channel_name]
        else:
            try:
                resp = g_slack_client.api_call('channels.create', name=channel_name, is_private=False)
                if not resp['ok']:
                    logger.warning(f'Something went wrong while tying to not create Slack channel {channel_name}:'
                                   f' {resp["error"]}')
                    return

                channel_id = resp['channel']['id']
                g_channel_map[channel_name] = resp['']
            except Exception as e:
                logger.info(f'Could not create Slack channel {channel_name}: {str(e)}')

        if channel_topic and channel_id:
            try:
                resp = g_slack_client.api_call('channels.setTopic', chanel=channel_id, topic=channel_topic)
                if not resp['ok']:
                    logger.warning(f'Could not set channel topic for channel {channel_name}')
                    return
            except Exception as e:
                logger.warning(f'Something went wrong while tying to set topic for channel {channel_name}: {str(e)}')

        # Invite all users to the channel
        if channel_id:
            for uid in g_user_ids:
                try:
                    resp = g_slack_client.api_call('channels.invite', channel=channel_id, user=uid)
                    if not resp['ok'] and not resp['error'] in {'already_in_channel', 'cant_invite_self'}:
                        logger.warning(f'Could not invite {uid} to {channel_name}: {resp["error"]}')
                except Exception as e:
                    logger.warning(f'Something went wrong while tying to invite {uid} to {channel_name}: {str(e)}')


def send_message(channel_name: str, message: dict) -> None:
    if g_slack_client is not None and message is not None:
        if channel_name not in g_channel_map:
            logger.warning(f'No such Slack channel: {channel_name}')
            return
        try:
            resp = g_slack_client.api_call('chat.postMessage', channel=g_channel_map[channel_name], as_user=True,
                                           **message)
            if not resp['ok']:
                logger.warning(f'Failed to post message to {channel_name}: {resp["error"]}')
        except Exception as e:
            logger.warning(f'Something went wrong while tying to to post message to {channel_name}: {str(e)}')


def upload_file(channel_name: str, filename: str, **kwargs: dict) -> None:
    if g_slack_client is not None and filename is not None:
        if channel_name not in g_channel_map:
            logger.warning(f'No such Slack channel: {channel_name}')
            return
        try:
            with open(filename, 'rb') as fp:
                resp = g_slack_client.api_call('files.upload', channels=g_channel_map[channel_name], file=fp, **kwargs, as_bot=True)
            if not resp['ok']:
                logger.warning(f'Something went wrong while tying to to upload {filename} to {channel_name}: '
                               f'{resp["error"]}')
        except Exception as e:
            logger.warning(f'Something went wrong while tying to to upload {filename} to {channel_name}: {str(e)}')


def generate_experiment_start_message(current_config:int, total_config: int, hyperparams: RNN_Hyperparameters,
                                      resuming: bool=False):
    resume_text = 'Resuming ' if resuming else ''
    attachment = {
        'title': 'Configuration',
        'text': f'```{hyperparams.to_string()}```',
        'color': '0000ff',
        'mrkdwn': True
    }
    return {
        'text': f'*{resume_text}{current_config} out of {total_config}:*',
        'attachments': [attachment],
        'mrkdwn': True
    }


def generate_experiment_resume_message(resume_spec:dict):
    # TODO: Prettify this -- collabsible hyperparams, title, lines ...
    attachment = {
        'title': 'Resuming',
        'text': f'```{json.dumps(resume_spec, indent=2)}```',
        'color': '00ffff',
        'mrkdwn': True
    }
    return {'attachments': [attachment]}


def generate_train_stats_message(log_record):
    if g_verbosity < 2:
        return None
    # TODO: Prettify this -- title, lines ...
    return {'text': log_record.to_string()}


def generate_epoch_end_message(epoch_that_ended: int, total_epochs:int, validation_loss: float, time_elapsed_sec: float):
    text = f'*Epoch* [{epoch_that_ended}/{total_epochs}] *Val.loss*: {validation_loss:.4f}' \
           f' *Time*: {str(timedelta(seconds=time_elapsed_sec // 1))}'
    attachment = {
        'text': text,
        'color': '06ff06',
        'mrkdwn': True
    }
    return {'attachments': [attachment]}


def generate_plot_message(filename):
    return {
        'filename': filename,
        'title': 'Error plot',
        'initial_comment': '<!here> Training finished for configuration'
    }


def generate_results_message(filename):
    return {
        'filename': filename,
        'title': 'Results file',
        'initial_comment': '<!here> Experiment finished'
    }


def generate_unexpected_error_message(stacktrace: str, args: Optional[dict]=None):
    args_str = ''
    if args is not None:
        args_str = f'\nParams:```{str(args)}``` Stacktrace: '
    attachment = {
        'text': f'Something went wrong:{args_str}```{stacktrace}```',
        "title": "ERROR",
        'color': '#ff0000',
        'mrkdwn': True
    }
    return {'text': '<!here>', 'attachments': [attachment]}