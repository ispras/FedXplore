import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Cifar10ModelTrainer:
    def __init__(self, cfg):
        # Always remember that you most likely
        # won't be able to change any variables in `context`,
        # since python is passed a copy of this object.
        # The exception would be mutable fields in the `context`.
        pass

    def train_fn(self, context):
        # context.model must be a torch.nn.Module,
        # or other mutable object, for this function to work

        context.model.train()
        for _ in range(context.local_epochs):
            for batch in context.train_loader:
                _, (input, targets) = batch

                inp = input[0].to(context.device)
                targets = targets.to(context.device)

                context.optimizer.zero_grad()
                outputs = context.model(inp)

                loss = context.get_loss_value(outputs, targets)
                loss.backward()

                context.optimizer.step()

                inp = input[0].to("cpu")
                targets = targets.to("cpu")

    def client_eval_fn(self, context):
        context.model.eval()
        val_loss = 0
        fin_targets = []
        fin_outputs = []

        with torch.no_grad():
            for _, batch in enumerate(context.valid_loader):
                _, (input, targets) = batch

                inp = input[0].to(context.device)
                targets = targets.to(context.device)

                outputs = context.model(inp)

                val_loss += context.criterion(outputs, targets).detach().item()

                fin_targets.extend(targets.tolist())
                fin_outputs.extend(outputs.tolist())

                inp = input[0].to("cpu")
                targets = targets.to("cpu")

        client_metrics = self.calculate_metrics(fin_targets, fin_outputs)
        return val_loss / len(context.valid_loader), client_metrics

    def test_fn(self, context):
        fin_targets, fin_outputs, test_loss = self.server_eval_fn(context)
        metrics = self.calculate_metrics(
            fin_targets,
            fin_outputs,
            verbose=True,
        )
        return metrics, test_loss

    def server_eval_fn(self, context):
        context.global_model.to(context.device)
        context.global_model.eval()

        test_loss = 0
        fin_targets = []
        fin_outputs = []

        with torch.no_grad():
            for _, batch in enumerate(context.test_loader):
                _, (input, targets) = batch

                inp = input[0].to(context.device)
                targets = targets.to(context.device)
                outputs = context.global_model(inp)

                test_loss += context.criterion(outputs, targets)
                fin_targets.extend(targets.tolist())
                fin_outputs.extend(outputs.tolist())

        test_loss /= len(context.test_loader)
        return fin_targets, fin_outputs, test_loss

    def calculate_metrics(self, fin_targets, fin_outputs, verbose=False):
        # Get results
        softmax = torch.nn.Softmax(dim=1)
        results = softmax(torch.as_tensor(fin_outputs)).max(dim=1)[1]
        fin_targets = torch.as_tensor(fin_targets)

        # Calc metrics
        df = pd.DataFrame(
            columns=["cifar"],
            index=[
                "Accuracy",
                "Precision",
                "Recall",
                "f1-score",
            ],
        )
        df.loc["Accuracy", "cifar"] = accuracy_score(fin_targets, results)
        df.loc["Precision", "cifar"] = precision_score(
            fin_targets, results, average="macro", zero_division=0
        )
        df.loc["Recall", "cifar"] = recall_score(
            fin_targets, results, average="macro", zero_division=0
        )
        df.loc["f1-score", "cifar"] = f1_score(
            fin_targets, results, average="macro", zero_division=0
        )
        if verbose:
            print(df)

        return df
