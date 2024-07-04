import numpy as np


class SchedulerInterval:
    def __init__(self, percent, schedule_type, multipliers, seq_len, total_steps):
        multipliers = [float(m) for m in multipliers.split("-")]
        percent = int(percent)
        self.validate_params(percent, schedule_type, multipliers)

        self.steps = round(int(percent) / 100 * total_steps)
        self.schedule_type = schedule_type
        self.interval_endpoints = [round(m * seq_len) - seq_len for m in multipliers]

    def validate_params(self, percent, schedule_type, multipliers):
        if not (0 < percent <= 100):
            raise ValueError(f"Percent must be between 1 and 100, got {percent}")

        if schedule_type == "const" and len(multipliers) != 1:
            raise ValueError(
                f"Const type should have exactly one multiplier, got {len(multipliers)}"
            )
        elif schedule_type in ["lin", "cos"] and len(multipliers) != 2:
            raise ValueError(
                f"{schedule_type.value} type should have exactly two multipliers, got {len(multipliers)}"
            )
        elif schedule_type not in ["const", "lin", "cos"]:
            raise ValueError(
                f"Unknown schedule type: {schedule_type}. Should be one of 'const', 'lin', 'cos'"
            )

        for m in multipliers:
            assert m >= 1 and m <= 5, "Multiplier should be between 1 and 5"


class TokenReductionScheduler:

    def __init__(self, total_steps, seq_len, schedule_str):
        self.total_steps = total_steps
        self.seq_len = seq_len
        self.schedule_steps = self._parse_schedule(schedule_str)
        self.current_step = 0

    def _parse_schedule(self, schedule_str):
        schedule_steps = []
        intervals = schedule_str.split(";")

        total_percent = 0

        for interval in intervals:
            split_interval = interval.split("_")

            if len(split_interval) != 3:
                raise ValueError(
                    f"Interval parameter should have form <percentage>_<type>_<multiplier/s>: {interval}"
                )

            percent, schedule_type, multipliers = split_interval
            schedule_step = SchedulerInterval(
                percent, schedule_type, multipliers, self.seq_len, self.total_steps
            )
            schedule_steps.append(schedule_step)
            total_percent += int(percent)

        if total_percent != 100:
            raise ValueError("Total percent of all intervals must equal 100")

        return schedule_steps

    def set_step(self, new_step):
        self.current_step = new_step

    def _linear_schedule(self, progress, interval_endpoints):
        return round(
            interval_endpoints[0]
            + progress * (interval_endpoints[1] - interval_endpoints[0])
        )

    def _cosine_schedule(self, progress, interval_endpoints):
        return round(
            interval_endpoints[0]
            + (1 - np.cos(progress * np.pi))
            / 2
            * (interval_endpoints[1] - interval_endpoints[0])
        )

    @property
    def value(self):
        sum_steps = 0
        for schedule_step in self.schedule_steps:
            if self.current_step < sum_steps + schedule_step.steps:
                progress = (self.current_step - sum_steps) / schedule_step.steps
                if schedule_step.schedule_type == "const":
                    return schedule_step.interval_endpoints[0]
                elif schedule_step.schedule_type == "lin":
                    return self._linear_schedule(
                        progress, schedule_step.interval_endpoints
                    )
                elif schedule_step.schedule_type == "cos":
                    return self._cosine_schedule(
                        progress, schedule_step.interval_endpoints
                    )
            sum_steps += schedule_step.steps

        return schedule_step.interval_endpoints[-1]

    def get_max_tokens_reduced(self):
        max_tokens_reduced = 0
        for schedule_step in self.schedule_steps:
            max_tokens_reduced = max(
                max_tokens_reduced, max(schedule_step.interval_endpoints)
            )
        return max_tokens_reduced
