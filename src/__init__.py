from .types import BFTNode, BFTResult, FingerprintResult
from .models import SimpleMLP, SmallCNN, PatchEmbedder, TransformerBlock, TinyViT
from .training import correct, label_transform_even_odd, train_epoch, evaluate
from .data_utils import (get_mnist_loaders, get_cifar10_loaders, get_imagenet_loaders,
                          collect_activations,
                          collect_layer_inputs, collect_layer_inputs_generic)
from .bft import (run_nmf_minibatch,
                  normalize_factors, sort_by_lambda,
                  full_nmf_pipeline, auto_nmf_pipeline,
                  compute_joint_arbors_normalized,
                  compute_conv_joint_arbors, compute_attn_joint_arbors,
                  trace_single_layer, collect_layer_dicts, bft,
                  nodes_at_layer)
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
                         plot_scaffold_graph,
                         plot_factor_overview_panel, plot_input_layer_factors,
                         plot_factor_gallery, plot_pruning_results,
                         plot_embedding_comparison,
                         extract_tree_nodes, compute_node_activations,
                         plot_factor_tree, extract_factor_tree_nodes,
                         compute_factor_activations, top_stimuli_factor_activations)
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
from .fingerprint_utils import (
    extract_factor_fingerprint,
    extract_fingerprint_matrix,
    compute_stimulus_similarity,
    compute_fingerprints,
    project_stimuli_onto_tree,
    project_onto_bft,
)
from .ablation_utils import (
    assign_factors_to_classes,
    path_from_child,
    extract_importance_scores,
    normalize_scores_per_layer,
    all_weight_keys,
    ablate_model,
    per_class_accuracy,
    magnitude_scores,
    act_magnitude_scores,
    taylor_scores,
    run_ablation_sweep,
    select_class_circuit,
)
from .robustness_utils import (
    compute_nmf_stability,
    align_factors,
    compute_k_sensitivity,
    plot_nmf_stability_figure,
    plot_k_sensitivity_figure,
    plot_robustness_summary,
)
