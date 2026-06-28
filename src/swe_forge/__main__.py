import typer

# Imported before the sibling CLI modules below (some call dotenv.load_dotenv()
# at import time) so it can snapshot the real process environment first.
from swe_forge.forge import runtime_env
from swe_forge.cli.harness import harness
from swe_forge.cli.mine import app as mine_app
from swe_forge.cli.validate import app as validate_app
from swe_forge.cli.export import app as export_app
from swe_forge.cli.benchmark import benchmark
from swe_forge.cli.publish import publish
from swe_forge.cli.synthetic import app as synthetic_app
from swe_forge.forge.cli import app as forge_app

# Remove forge credentials that an implicit .env load injected during the imports
# above, so `env -u TEACHER_LLM_API_KEY swe-forge forge llm-check` fails fast.
runtime_env.scrub_injected_forge_env()

app = typer.Typer(name="swe-forge", help="SWE-bench dataset generator")


@app.command()
def version():
    from swe_forge import __version__

    typer.echo(f"swe-forge version {__version__}")


app.command(name="harness")(harness)
app.command(name="benchmark")(benchmark)
app.command(name="publish")(publish)
app.add_typer(mine_app, name="mine")
app.add_typer(validate_app, name="validate")
app.add_typer(export_app, name="export")
app.add_typer(synthetic_app, name="synthetic")
app.add_typer(forge_app, name="forge")


def main():
    app()


if __name__ == "__main__":
    main()
