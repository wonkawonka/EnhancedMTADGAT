"""Run Kaggle CLI with an explicit, verified account; never print credentials."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys


def authenticate_account(account):
    project = Path(__file__).resolve().parents[1]
    for name in ['KAGGLE_USERNAME', 'KAGGLE_KEY', 'KAGGLE_API_TOKEN']:
        os.environ.pop(name, None)
    if account == 'daisychen2':
        os.environ['KAGGLE_CONFIG_DIR'] = str(project/'.kaggle')
        credential_file = project/'.kaggle/kaggle.json'
        credentials = json.loads(credential_file.read_text())
        token_file = project/'.kaggle/access_token'
        token = credentials.get('key', '')
        if token_file.exists():
            os.environ['KAGGLE_API_TOKEN'] = str(token_file)
        elif token.startswith('KGAT_'):
            os.environ['KAGGLE_API_TOKEN'] = token
        # Kaggle 2.2 prioritizes ~/.kaggle/access_token over KAGGLE_CONFIG_DIR.
        # Disable that lookup for this legacy-key account only.
        if 'KAGGLE_API_TOKEN' not in os.environ:
            import kagglesdk
            import kagglesdk.kaggle_env
            import kagglesdk.kaggle_http_client
            kagglesdk.get_access_token_from_env = lambda: (None, None)
            kagglesdk.kaggle_env.get_access_token_from_env = lambda: (None, None)
            kagglesdk.kaggle_http_client.get_access_token_from_env = lambda: (None, None)
    else:
        directory = Path('/home/chenmj/.kaggle')
        os.environ['KAGGLE_CONFIG_DIR'] = str(directory)
        os.environ['KAGGLE_API_TOKEN'] = str(directory/'access_token')
    import kaggle
    if not kaggle.api._authenticated:
        kaggle.api.authenticate()
    actual = kaggle.api.config_values.get('username')
    if actual != account:
        raise RuntimeError(f'Account mismatch: expected {account}, authenticated {actual}')
    print(f'VERIFIED_KAGGLE_ACCOUNT {actual}', flush=True)
    return kaggle.api


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--account', required=True, choices=['chenmjdaisy', 'daisychen2'])
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    authenticate_account(args.account)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    sys.argv = ['kaggle', *command]
    runpy.run_module('kaggle', run_name='__main__')


if __name__ == '__main__':
    main()
