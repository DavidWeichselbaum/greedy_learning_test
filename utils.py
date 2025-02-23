import git


def get_commit_hash():
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    return sha


def add_gradient_debug_hook(model):
    for name, param in model.named_parameters():
        param.register_hook(lambda grad, name=name: print(f"Gradient for {name} with shape {grad.shape}"))
