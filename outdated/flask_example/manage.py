#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
from pathlib import Path


def main():
    """Run administrative tasks."""
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc

    p = Path("logs/backend")
    p.mkdir(parents=True, exist_ok=True)
    p = Path("logs/frontend")
    p.mkdir(parents=True, exist_ok=True)

    try:
        if sys.argv[2] == 'react':
            os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.development_settings')
            project_root = os.getcwd()
            os.chdir(os.path.join(project_root, "frontend"))
            os.system("yarn run build")
            os.chdir(project_root)
            sys.argv.pop(2)
        elif sys.argv[2] == 'production':
            os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.production_settings')
            sys.argv.pop(2)
    except IndexError:
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.development_settings')
        execute_from_command_line(sys.argv)        
    else:
        execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
