"""
Functions used for model optimization.
"""

import numpy as np


def step_lr_schedule(
    optimizer,
    i_iter: int,
    scheduler_cus: dict,
    warm_up: int = 0,
    use_lr_schedule: bool = True,
):
    """
    Step learning rate schedule.
    ### Parameters:
    - optimizer: torch.optim.Optimizer, optimizer.
    - i_iter: int, current iteration.
    - scheduler_cus: dict, scheduler configuration.
    - warm_up: int, warm up iterations.
    - use_lr_schedule: bool, whether to use learning rate schedule.
    """

    if use_lr_schedule:
        if (warm_up > 0) and (i_iter < warm_up):
            lr = (i_iter + 1) / warm_up * scheduler_cus["lr"]
            # set learning rate
            for g in optimizer.param_groups:
                g["lr"] = lr

        if i_iter >= warm_up:
            if (i_iter + 1 - warm_up) % scheduler_cus["every"] == 0:
                lr = scheduler_cus["lr"] * (
                    scheduler_cus["rate"]
                    ** ((i_iter + 1 - warm_up) // scheduler_cus["every"])
                )
                lr = np.maximum(lr, scheduler_cus["min"])
                for g in optimizer.param_groups:
                    g["lr"] = lr
    else:
        if (warm_up > 0) and (i_iter < warm_up):
            lr = (i_iter + 1) / warm_up * scheduler_cus["lr"]
            for g in optimizer.param_groups:
                g["lr"] = lr

        # if i_iter >= warm_up:
        #     for g in optimizer.param_groups:
        #         g["lr"] = scheduler_cus["lr"]
