import sys
import subprocess
from hydra.core.hydra_config import HydraConfig
import torch
from ecglib.version import COMMIT_HASH


def get_repository_info(do_print: bool = False) -> dict:
    """Create dict with information about current repository state
    to be reproduced later

    :param do_print: Whether to print repository info

    :return: Dictionary with commit hash and diff
    """

    def sh(cmd):
        return subprocess.check_output(cmd, shell=True, text=True).strip()

    git_info = {
        "commit_hash": sh("git rev-parse HEAD"),
        "repo_name": sh("git rev-parse --show-toplevel").split("/")[-1],
        "remote_url": sh("git config --get remote.origin.url || echo '(no origin)'"),
        "branch_name": sh("git rev-parse --abbrev-ref HEAD"),
        "staged_changes": [
            line for line in sh("git diff --name-only --cached").splitlines() if line
        ],
        "unstaged_changes": [
            line for line in sh("git diff --name-only").splitlines() if line
        ],
        "untracked_changes": [
            line
            for line in sh("git ls-files --others --exclude-standard").splitlines()
            if line
        ],
        "ecglib_commit_hash": COMMIT_HASH,
        # "diff": subprocess.run("git diff".split(), capture_output=True).stdout.decode(
        #     "utf-8"
        # ),
    }

    unstaged_diffs = {}
    repo_root = sh("git rev-parse --show-toplevel")
    for path in git_info["unstaged_changes"]:
        diff_text = subprocess.check_output(
            ["git", "-C", repo_root, "-c", "color.ui=false", "diff", "--", path],
            text=True,
        )
        lines = diff_text.splitlines()
        # Extract header lines (everything before the first @@)
        header_lines = []
        i = 0
        for i, line in enumerate(lines):
            if line.startswith("@@ "):
                break
            header_lines.append(line)
        else:
            # If no @@ found, the whole thing is header (unlikely for diff)
            i = len(lines)

        # Extract hunks
        hunks = []
        current_hunk = []
        for line in lines[i:]:
            if line.startswith("@@ "):
                if current_hunk:
                    hunks.append(current_hunk)
                current_hunk = [line]
            else:
                if current_hunk:  # Avoid adding lines before first hunk
                    current_hunk.append(line)
        if current_hunk:
            hunks.append(current_hunk)

        # Store as structured dict with lists of lines
        unstaged_diffs[path] = {"header": header_lines, "hunks": hunks}
    git_info["unstaged_diffs"] = unstaged_diffs

    if do_print:
        print("commit:", git_info["commit_hash"])
        print("branch:", git_info["branch_name"])
        print("unstaged files:", git_info["unstaged_changes"])
        for p, hunks in git_info["unstaged_diffs"].items():
            print(f"\n--- DIFF for {p} ({len(hunks)} hunks) ---")
            for i, h in enumerate(hunks, 1):
                print(f"\n--- HUNK {i} ---")
                print(h)

    return git_info


def get_run_command():
    script_name = sys.argv[0]
    args = " ".join(sys.argv[1:])
    run_dir = HydraConfig.get().run.dir
    run_command = f"python {script_name} {args} > {run_dir}/train_log.txt &"
    return run_command


def create_model_info(
    model_state, valid_metrics, valid_loss, test_metrics, test_loss, cfg
):
    model_info = {
        "git": get_repository_info(),
        "run_command": get_run_command(),
        "model": model_state,
        "metrics": {
            "valid_metrics": valid_metrics,
            "valid_loss": valid_loss,
            "test_metrics": test_metrics,
            "test_loss": test_loss,
        },
        "config_file": cfg,
    }

    return model_info


def generate_confluence_report(
    run_dir,
    checkpoint_path=None,
    git_info=None,
    run_command=None,
):
    """Generate header report of current experiment in confluence_wiki format

    Args:
        run_dir (str): path to directory with current logs/metrics etc.
        checkpoint_path (str): path to model checkpoint
        git_info (dict): repository info (name, hash, branch)
        run_command (str): command to run federated learning

    Returns:
        typing.TextIO: file pointer to continue writing a report
    """

    filename = f"{run_dir}/experiment_report.txt"
    f = open(filename, "w")
    if checkpoint_path is not None:
        ckpt = torch.load(
            checkpoint_path, map_location=torch.device("cpu"), weights_only=False
        )
        git_info = ckpt["git"]
        run_command = ckpt["run_command"]
        metrics = ckpt["metrics"]
    else:
        assert (
            git_info is not None and run_command is not None
        ), f"if we don't have global checkpoint in FL, we need to provide git_info and run_command."

    f.write(
        f"- Ветка проекта _{git_info['repo_name']}_:\n*{git_info['branch_name']}*\n"
    )
    f.write(f"- Хеш проекта _{git_info['repo_name']}_:\n*{git_info['commit_hash']}*\n")
    f.write(f"- Хеш проекта _ecglib_:\n*{git_info['ecglib_commit_hash']}*\n")
    f.write(f"- Чекпоинт модели:\n_{checkpoint_path}_\n")
    f.write(f"- Код запуска FL:\n{{code:language=python}}{run_command}{{code}}\n")

    # report metrics
    if metrics is not None:
        if metrics["valid_metrics"] is not None:
            f.write("\n- Server Valid Metrics\n\n")
            f = convert_df_to_table(metrics["valid_metrics"], f)
        if metrics["valid_loss"] is not None:
            f.write(f"- Server Valid Loss: *{metrics['valid_loss']}*\n\n")
        if metrics["test_metrics"] is not None:
            f.write("\n- Server Test Metrics\n\n")
            f = convert_df_to_table(metrics["test_metrics"], f)
        if metrics["test_loss"] is not None:
            f.write(f"- Server Test Loss: *{metrics['test_loss']}*")

    return f


def convert_df_to_table(df, f):
    """Convert dataframe to confluence_wiki format and write this content to file pointer

    Args:
        df (pandas.core.frame.DataFrame): dataframe with some metrics
        f (typing.TextIO): file pointer to report txt

    Returns:
        f (typing.TextIO): file pointer to report txt
    """
    f.write("|| ||")
    for col in df.columns:
        f.write(f"{col}||")
    f.write("\n")
    for index, row in df.iterrows():
        f.write(f"||{index}|")
        f.write("|".join([str(round(val, 4)) for val in row]))
        f.write("|\n")
    return f
