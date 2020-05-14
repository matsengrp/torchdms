set -eu

for CONFIG in $(find $PWD -name "config.json")
do
    cd $(dirname $CONFIG)
    echo "\n\n*** Starting $CONFIG"
    tdms go --config $CONFIG
    echo "*** $CONFIG ran successfully"
done
