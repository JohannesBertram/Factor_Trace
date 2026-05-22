from .models import SimpleMLP
from .training import correct, label_transform_even_odd, train_epoch, evaluate
from .data_utils import (get_mnist_loaders, collect_activations,
                          collect_layer_inputs, collect_layer_inputs_generic,
                          collect_layer_data)
from .bft import (run_nmf, normalize_factors, sort_by_lambda, full_nmf_pipeline,
                  auto_nmf_pipeline, compute_joint_arbors_normalized,
                  compute_conv_joint_arbors, trace_single_layer, bft)
from .r1d import run_r1d, r1d, rec_err_curve
# Backward-compat: moved to old_code.py (gitignored) but still importable
from .old_code import select_k_from_lambdas, nmf_component_sweep
from .neuron_analysis import (
    compute_synaptic_arbors,
    pca_decoding,
    plot_pca_scatter,
    plot_pca_alpha,
    plot_mean_arbors,
    plot_pos_mean_arbors,
    plot_activation_weighted_arbors,
    plot_activation_hist,
    single_neuron_report,
)
from .plot_utils import (plot_nmf_component, plot_factor_graph, plot_nmf_scree,
                         plot_neuron_nmf_component, plot_neuron_nmf_scatter,
                         plot_scaffold_graph)
from .checkpoint import (
    save_experiment,
    load_experiment,
    get_transform,
    get_loaders_from_config,
    MODEL_REGISTRY,
    TRANSFORM_REGISTRY,
    DATASET_REGISTRY,
)
from .scaffold_utils import (bft_node_vals, build_scaffold_edges,
                             scaffold_loading_from_edges, scaffold_layer_sizes_from_edges)
from .stimulus_utils import (
    extract_tree_nodes,
    compute_node_activations,
    plot_factor_tree,
    extract_factor_fingerprint,
    extract_fingerprint_matrix,
    compute_stimulus_similarity,
    project_stimuli_onto_tree,
    extract_factor_tree_nodes,
    compute_factor_activations,
    nodes_at_layer,
    top_stimuli_factor_activations,
)
from .ablation_utils import (
    assign_factors_to_classes,
    path_from_child,
    extract_importance_scores,
    all_weight_keys,
    ablate_model,
    per_class_accuracy,
    magnitude_scores,
    act_magnitude_scores,
    taylor_scores,
    run_ablation_sweep,
)
