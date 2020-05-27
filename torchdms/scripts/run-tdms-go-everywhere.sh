set -eu

for CONFIG in $(find $PWD -name "config.json")
do
    cd $(dirname $CONFIG)
    if test -n "$(find . -maxdepth 1 -name '*sentinel' -print -quit)"
    then
        echo "\n\n*** Skipping $CONFIG because sentinel file found."
        continue
    fi
    echo "\n\n*** Starting $CONFIG"
    tdms go --config $CONFIG
    echo "*** $CONFIG ran successfully"
done
