def test_attention_visualization_uses_headless_backend() -> None:
    from trkh.evaluation import attention_viz

    assert attention_viz.matplotlib.get_backend().lower() == "agg"
