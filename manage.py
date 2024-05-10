#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys

from producer import wait_for_kafka_broker
from utils.download_models import download_kprn_model


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
    # wait_for_kafka_broker('[Broker waiting in manage.py]')
    download_kprn_model()

    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
