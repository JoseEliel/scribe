from scribe.config import launch_settings, load_config
from scribe.ui import build_demo


demo = build_demo(load_config())


if __name__ == "__main__":
    try:
        demo = demo.queue()
    except Exception:
        pass
    demo.launch(**launch_settings())
