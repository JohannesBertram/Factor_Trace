from .models import SimpleMLP
from .training import correct, label_transform_even_odd, train_epoch, evaluate
from .data_utils import get_mnist_loaders, collect_activations, collect_layer_inputs
from .factorization import (run_nmf, normalize_factors, sort_by_lambda, full_nmf_pipeline,
                            select_k_from_lambdas, auto_nmf_pipeline,
                            nmf_component_sweep, compute_stimulus_loading,
                            compute_effective_arbors, reconstruct_from_components,
                            compute_joint_arbors_normalized, trace_single_layer, dfs_trace,
                            tree_trace)
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
