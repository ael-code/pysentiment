import json
import os
import logging
import click

from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud import WatsonException
import watson_developer_cloud.natural_language_understanding.features.v1 as \
    features


env_prefix = "NLU"
api_version = '2017-02-27',

user = None
pwd = None

nlu = None


def load_credentials_from_file(cred_file):
    with open(cred_file, 'r') as f:
        cred_json = json.load(f)
    return cred_json['username'], cred_json['password']


def analyze_text_sentiment(nlu, text):
    response = nlu.analyze(
        text=text,
        features=[features.Sentiment()])
    return response['sentiment']['document']['score']


@click.command()
@click.option('-c', '--cred-file',
        type=click.Path(exists=True, dir_okay=False, readable=True),
        help="file from which load credential (BLUEMIX json format)")
@click.option('--user')
@click.option('--password')
@click.option('-v', '--verbose', is_flag=True)
@click.argument('data', envvar='DATA')
def main(cred_file, user, password, verbose, data):

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    if cred_file:
        logging.debug("Loading credentials from file '{}'".format(cred_file))
        user, pwd = load_credentials_from_file(cred_file)
    elif user and password:
        logging.debug("Loding credentials from env variables")
        user, pwd = (user, password)
    else:
        logging.critical("Unable to setup credentials")
        exit(1)

    nlu = NaturalLanguageUnderstandingV1(
            version=api_version,
            username=user,
            password=pwd)

    if os.path.exists(data):
        logging.debug("Treating input data as a file system path")
        raise NotImplemented()
    else:
        try:
            print("score: {}".format(analyze_text_sentiment(nlu, data)))
        except WatsonException as e:
            click.secho(str(e), fg='red')
            exit(1)


if __name__ == "__main__":
    main(auto_envvar_prefix=env_prefix)
