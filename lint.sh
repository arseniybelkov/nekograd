scriptDir=$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")
cd "$scriptDir" || exit 1

black nekograd; isort nekograd; flake8 nekograd