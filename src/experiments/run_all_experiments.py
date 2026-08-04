"""Run the main entrypoint of every maintained experiment."""

from src.experiments import (
    experiment_graph_size_visibility as graph_size_visibility,
    experiment_model_improvements as model_improvements,
    experiment_topologies as topologies,
    experiment_vertex_visibility_spline_transfer as spline_transfer,
    multiple_psd_exp,
)

if __name__ == "__main__":
    graph_size_visibility.main()
    model_improvements.main()
    topologies.main()
    spline_transfer.main()
    multiple_psd_exp.main()
