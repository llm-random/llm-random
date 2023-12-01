import multiprocessing as mp

# n_blanks: int = 0
# n_blanks = mp.Value("i", 0)
curriculum_step: int = 5000

train_dataloader = None
eval_dataloader = None


def set_n_blanks(n_blanks: int) -> None:
    global train_dataloader
    global eval_dataloader
    g1 = train_dataloader.dataloader.dataset.n_blanks_glob
    g2 = eval_dataloader.dataloader.dataset.n_blanks_glob
    with g1.get_lock():
        with g2.get_lock():
            g1.value = g2.value = n_blanks

    # train_dataloader.dataloader.dataset.n_blanks_glob.value = n_blanks
    # eval_dataloader.dataloader.dataset.n_blanks_glob.value = n_blanks


def get_n_blanks() -> int:
    global train_dataloader
    global eval_dataloader
    assert (
        train_dataloader.dataloader.dataset.n_blanks_glob.value
        == eval_dataloader.dataloader.dataset.n_blanks_glob.value
    )
    return train_dataloader.dataloader.dataset.n_blanks_glob.value
