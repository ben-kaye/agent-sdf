"""Launch the WebGL2 viewer for an SDF scene."""


def main():
    from .glsl_viewer import main as run_glsl_viewer

    run_glsl_viewer()


if __name__ == "__main__":
    main()
