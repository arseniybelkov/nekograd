scriptDir=$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")
cd "$scriptDir" || exit 1

autoflake --remove-all-unused-imports \
          --ignore-init-module-imports --in-place -r .
black .; isort .; flake8 nekograd