import click


@click.group()
def cli():
    pass


@cli.command()
@click.argument("model_json", type=click.Path(exists=True))
def run(model_json):
    print(model_json)


if __name__ == "__main__":
    cli()
