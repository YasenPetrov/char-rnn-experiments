# A PyTorch-based language modelling experimentation framework

## Logging

### Slack logging

The framework can post training logs to a Slack workspace if supplied with a
[Slack test token](https://api.slack.com/custom-integrations/legacy-tokens). A new channel is added to the workspace
and every member is added to the channel. 
#### How?
Pass the `--slack` flag to experiment.py to enable Slack messages

#### Requrements
A [Slack test token](https://api.slack.com/custom-integrations/legacy-tokens) must be present in an environment variable
called `SLACK_API_TOKEN`

#### What is logged?
 - Training start
 - The validation loss and time taken at end of each epoch.
 - *Optional* If the `-v` flag is set, more verbose messages are logged at every `batches_between_stats` batches, as
 specified by the experiment's `spec.json`
 - A plot of the losses at the end of training for each configuration
 - The `results.json` file, summarizing results at the end of training
