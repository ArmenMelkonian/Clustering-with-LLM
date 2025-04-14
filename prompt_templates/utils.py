from pathlib import Path
import jinja2

_CURRENT_DIR = Path(__file__).parent


def read_prompt_template(name: str) -> jinja2.Template:
    with open(_CURRENT_DIR / f"{name}.j2") as f:
        return jinja2.Template(f.read())
