import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  #save to file, no display
import matplotlib.pyplot as plt

from training.simclr_training import train_simclr
from training.feature_extraction import extract_features
from training.classifier import train_classifier
from training.linear_probe import extract_test_features, train_linear_probe
from training.semisupervised import train_semisupervised
from typiclust.selection import typiclustselect_round, randomselect_round
from typiclust.baselines import uncertaintyselect_round

#plot style config

_LABEL  = {'typiclust': 'TPC_RP', 'random': 'Random',
           'uncertainty': 'Uncertainty', 'margin': 'Margin',
           'entropy': 'Entropy', 'hybrid': 'Hybrid (TC→Unc)'}
_COLOR  = {'typiclust': 'tab:blue', 'random': 'black',
           'uncertainty': 'tab:red', 'margin': 'tab:green',
           'entropy': 'tab:purple', 'hybrid': 'tab:orange'}
_MARKER = {'typiclust': 'o', 'random': 's', 'uncertainty': '^',
           'margin': 'D', 'entropy': 'v', 'hybrid': 'P'}


def select_round(strategy, features, labels_gt, labeled_indices, budget, device, classifier_epochs, round_idx=0):
    
    n_total = len(labels_gt)
    
    if strategy == 'typiclust':
        return typiclustselect_round(features, labeled_indices, budget)
    
    elif strategy == 'random':
        return randomselect_round(n_total, labeled_indices, budget)
    
    elif strategy in ('uncertainty', 'margin', 'entropy'):
        return uncertaintyselect_round(
            labeled_indices, budget, n_total,
            strategy=strategy, device=device, epochs=classifier_epochs
        )
    
    elif strategy == 'hybrid':
        from typiclust.selection import hybridselect_round
        return hybridselect_round(
            features, labeled_indices, budget, round_idx,
            n_total=n_total, device=device, classifier_epochs=classifier_epochs
        )
    
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")


def print_table(results, num_rounds, budget_per_round, title):
    strats = list(results.keys())
    
    print(f"\n=== {title} ===")
    
    header = f"{'budget':<8}" + "".join(f"{s:>18}" for s in strats)
    
    print(header)
    print("-" * len(header))
    
    for r in range(num_rounds):
        budget = (r + 1) * budget_per_round
        row = f"{budget:<8}"
        
        for s in strats:
            vals = results[s][r]
            mean = np.mean(vals)
            se   = np.std(vals) / np.sqrt(len(vals))
            row += f"{mean:>10.2f}±{se:<6.2f}"
        print(row)


def plot_results(results, num_rounds, budget_per_round, title, save_path):
    
    budgets = [(r + 1) * budget_per_round for r in range(num_rounds)]
    plt.figure(figsize=(7, 5))
    
    for strategy, round_data in results.items():
        means = [np.mean(round_data[r]) for r in range(num_rounds)]
        ses   = [np.std(round_data[r]) / np.sqrt(len(round_data[r]))
                 for r in range(num_rounds)]
        plt.errorbar(budgets, means, yerr=ses,
                     label=_LABEL.get(strategy, strategy),
                     color=_COLOR.get(strategy),
                     marker=_MARKER.get(strategy, 'o'),
                     capsize=3, linewidth=1.5)
    plt.xlabel("Cumulative budget (labeled examples)")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Plot saved → {save_path}")


def run_experiment(
        
    #hyperparams
    simclr_epochs=500,
    simclr_batch_size=512,
    budget_per_round=10,       
    num_rounds=5,
    classifier_epochs=100,      
    num_seeds=10,               
    device='cuda',
    checkpoint_path='simclr_checkpoint.pt',

    #dont forget hybird and ',' even if 1 item
    strategies=('typiclust', 'random', 'uncertainty', 'margin', 'entropy', 'hybrid'),
):
    device = device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    #simclr pre training
    simclr_model = train_simclr(epochs=simclr_epochs, batch_size=simclr_batch_size, device=device, checkpoint_path=checkpoint_path)

    #feature extraction
    train_features, train_labels = extract_features(simclr_model, device=device)
    test_features,  test_labels  = extract_test_features(simclr_model, device=device)
    print(f"  test features: {test_features.shape}\n")

    #accumulators
    sup_results    = {s: {r: [] for r in range(num_rounds)} for s in strategies}
    probe_results  = {s: {r: [] for r in range(num_rounds)} for s in strategies}
    semisup_results = {s: {r: [] for r in range(num_rounds)} for s in strategies}

    for seed in range(num_seeds):

        print(f"\nSeed {seed + 1}/{num_seeds}\n")
        np.random.seed(seed)
        torch.manual_seed(seed)

        for strategy in strategies:
            labeled_indices = []
            for round_idx in range(num_rounds):

                new_queries = select_round(
                    strategy, train_features, train_labels,
                    labeled_indices, budget_per_round, device, classifier_epochs,
                    round_idx=round_idx
                )

                labeled_indices = labeled_indices + new_queries
                budget = len(labeled_indices)

                #fully supervised
                acc = train_classifier(
                    labeled_indices, device=device, epochs=classifier_epochs
                )

                #linear probe on simclr features
                probe_acc = train_linear_probe(
                    train_features[labeled_indices], train_labels[labeled_indices],
                    test_features, test_labels,
                    device=device, supervised_epochs=classifier_epochs
                )

                semisup_acc = train_semisupervised(
                    train_features, train_labels, labeled_indices,
                    test_features, test_labels,
                    device=device, supervised_epochs=classifier_epochs,
                )

                sup_results[strategy][round_idx].append(acc)
                probe_results[strategy][round_idx].append(probe_acc)
                semisup_results[strategy][round_idx].append(semisup_acc)

                print(f"  [{strategy:10s}] round {round_idx + 1} "
                      f"n={budget:3d}: sup={acc:.1f}%  probe={probe_acc:.1f}%  semisup={semisup_acc:.1f}%")

    print_table(sup_results,    num_rounds, budget_per_round, "Fully Supervised")
    print_table(probe_results,  num_rounds, budget_per_round,
                 "Self-Supervised Embedding (Linear Probe)")
    print_table(semisup_results, num_rounds, budget_per_round,
                 "Semi-Supervised (Pseudo-Labelling)")

    plot_results(sup_results, num_rounds, budget_per_round,
                  title="Fully Supervised — CIFAR-10 (low budget)",
                  save_path="results_supervised.png")
    plot_results(probe_results, num_rounds, budget_per_round,
                  title="Linear Probe on SimCLR features — CIFAR-10 (low budget)",
                  save_path="results_probe.png")
    plot_results(semisup_results, num_rounds, budget_per_round,
                  title="Semi-Supervised (Pseudo-Labelling) — CIFAR-10 (low budget)",
                  save_path="results_semisup.png")

    return sup_results, probe_results, semisup_results


if __name__ == "__main__":
    run_experiment(
        simclr_epochs=500,
        simclr_batch_size=512,
        budget_per_round=10,        #b = m (number of classes)
        num_rounds=5,
        classifier_epochs=100,
        num_seeds=10,
        strategies=('hybrid',), #run only hybrid, uses cache
    )
