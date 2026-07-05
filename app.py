"""Point d'entrée Gradio."""
from __future__ import annotations

from src.ui import build_ui


def main() -> None:
    demo = build_ui()
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, show_error=True)


if __name__ == "__main__":
    main()
