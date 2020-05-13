set -eux

run() {
    FLAGS=$@
    for CONFIG in $(find $PWD -name "config.json")
    do
        cd $(dirname $CONFIG)
        tdms $FLAGS go --config $CONFIG
    done
}

run --dry-run

read -p "Do want to execute this run? [n] " yn
case $yn in
    [Yy]* ) run; break;;
    * ) break;;
esac
