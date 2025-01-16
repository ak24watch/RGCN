import torch
import torch.nn.functional as F
from RgcnModel import RgcnModel
from trainUtility import trainInfo
import argparse
import plotly.graph_objects as go
import plotly.io as pio


def train_and_evaluate(dataset_name):
    g, category_id, num_classes, num_rels, labels, train_idx, test_idx, target_idx = (
        trainInfo(dataset_name)
    )

    hidden_dim = 16

    model = RgcnModel(
        hiden_dim=hidden_dim,
        out_dim=num_classes,
        num_rels=num_rels,
        regularizer="basis",
        num_bases=num_rels,
        activation=F.relu,
        dropout=0.05,
        layer_norm=True,
        g=g,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    print(f"Start training on {dataset_name}...")
    model.train()
    best_val_acc = 0
    for epoch in range(200):
        optimizer.zero_grad()
        logits = model.forward()
        logits = logits[target_idx]

        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        loss.backward()

        optimizer.step()
        train_acc = torch.sum(logits[train_idx].argmax(dim=1) == labels[train_idx])
        train_acc = train_acc.item() / len(train_idx)
        val_loss = F.cross_entropy(logits[test_idx], labels[test_idx])
        val_acc = torch.sum(logits[test_idx].argmax(dim=1) == labels[test_idx])
        val_acc = val_acc.item() / len(test_idx)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        print(
            "Epoch {:05d} | ".format(epoch)
            + "Train Accuracy: {:.4f} | Train Loss: {:.4f} | ".format(
                train_acc, loss.item()
            )
            + "Validation Accuracy: {:.4f} | Validation loss: {:.4f}".format(
                val_acc, val_loss.item()
            ),
        )
    return best_val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RGCN for entity classification")
    parser.add_argument(
        "--dataset",
        type=str,
        default="aifb",
        help="Dataset name ('aifb', 'mutag', 'bgs', 'am').",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the best validation accuracies for all datasets.",
    )
    parser.add_argument(
        "--save_plot",
        type=str,
        default="best_validation_accuracies.png",
        help="File name to save the plot.",
    )
    args = parser.parse_args()

    if args.plot:
        datasets = ["aifb", "mutag", "bgs", "am"]
        best_accuracies = []
        for dataset in datasets:
            best_acc = train_and_evaluate(dataset)
            best_accuracies.append(best_acc)

        fig = go.Figure(
            data=[
                go.Bar(name="Best Validation Accuracy", x=datasets, y=best_accuracies)
            ]
        )
        fig.update_layout(
            title="Best Validation Accuracy for Each Dataset",
            xaxis_title="Dataset",
            yaxis_title="Accuracy",
        )
        pio.write_image(fig, args.save_plot)
    else:
        train_and_evaluate(args.dataset)
