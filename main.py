import json
import os
import logging
import click
import re

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


def analyze_dataset(nlu, path):
    fname_pattern = re.compile(r"^(\d+)_(\d+)\.txt$")
    logging.debug("Analyzing dataset in '{}'".format(path))
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".watson"):
                continue
            file_path = os.path.join(root, file)

            match = fname_pattern.match(file)
            if not match:
                logging.info("Skipping unmatching file: '{}'".format(file_path))
                continue

            output_file = file_path + ".watson"
            if os.path.exists(output_file):
                continue

            logging.info("Analyzing file: '{}'".format(file_path))
            ID = match.group(1)
            vote = match.group(2)
            try:
                score = analyze_file_sentiment(nlu, file_path)
            except WatsonException as we:
                if "limit exceeded" in str(we):
                    logging.error(we)
                    exit(2)
            except Exception as e:
                logging.error("Error while analyzing file: '{}'. {}".format(file, str(e)))
                continue

            with open(output_file, 'w') as f:
                f.write(str(score))


def dataset_stats(path):
    fname_pattern = re.compile(r"^(\d+)_(\d+)\.txt$")
    not_analyzed = 0
    analyzed = 0

    logging.debug("Analyzing dataset in '{}'".format(path))
    for root, dirs, files in os.walk(path):
        curr_not_analyzed = 0
        curr_analyzed = 0
        for file in files:
            if file.endswith(".watson"):
                continue
            file_path = os.path.join(root, file)

            match = fname_pattern.match(file)
            if not match:
                logging.info("Skipping unmatching file: '{}'".format(file_path))
                continue

            output_file = file_path + ".watson"
            if not os.path.exists(output_file):
                curr_not_analyzed += 1
                continue

            curr_analyzed += 1
        print("Folder: '{}', analyzed: {}, not_scored: {}".format(root, curr_analyzed, curr_not_analyzed))
        not_analyzed += curr_not_analyzed
        analyzed += curr_analyzed
    print("Stats, analyzed: {}, not_scored: {}".format(analyzed, not_analyzed))


def analyze_file_sentiment(nlu, file_path):
    with open(file_path, 'r') as f:
        text = f.read()
    return analyze_text_sentiment(nlu, text)


def analyze_text_sentiment(nlu, text):
    response = nlu.analyze(
        text=text,
        features=[features.Sentiment()])
    # logging.debug(json.dumps(response))
    return response['sentiment']['document']['score']


@click.command()
@click.option('-c', '--cred-file',
        type=click.Path(exists=True, dir_okay=False, readable=True),
        help="file from which load credential (BLUEMIX json format) [NLU_CRED_FILE]")
@click.option('--user')
@click.option('--password')
@click.option('-v', '--verbose', is_flag=True)
@click.option('-d', '--dataset-dir',
        type=click.Path(exists=True, dir_okay=True, file_okay=False, readable=True),
        help="path to the Large Movie Review Dataset (from Andrew Maas)")
@click.option('-s', '--stats', is_flag=True, help="do not analyze dataset, just print statistics")
@click.option('-t', '--text')
@click.option('-f', '--file', type=click.Path(exists=True, dir_okay=False, readable=True))
def main(cred_file, user, password, verbose, dataset_dir, stats, text, file):

    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

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

    try:
        if dataset_dir:
            if stats:
                dataset_stats(dataset_dir)
            else:
                analyze_dataset(nlu, dataset_dir)
            exit(0)

        if file:
            print("score: {}".format(analyze_file_sentiment(nlu, file)))
            exit(0)

        if text:
            print("score: {}".format(analyze_text_sentiment(nlu, text)))
            exit(0)

    except WatsonException as e:
        click.secho(str(e), fg='red')
        exit(1)


if __name__ == "__main__":
    main(auto_envvar_prefix=env_prefix)
