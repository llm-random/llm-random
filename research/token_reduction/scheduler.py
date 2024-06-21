class TokenReductionScheduler:
    def __init__(self, ranges):
        """
        Initialize the scheduler with specified ranges.

        :param ranges: List of tuples, each tuple defines a range.
                       A tuple can be of two forms:
                        - (start_step, end_step, start_val, end_val): for a linear change
                        - (start_step, end_step, constant_val): for a constant value
        """
        self.ranges = ranges
        self.currrent_step = 0
        self._validate_ranges()

    def _validate_ranges(self):
        """
        Validate that ranges are continuous and non-overlapping.
        """
        assert len(self.ranges) > 0, "At least one range must be provided"
        previous_end = -1
        for r in self.ranges:
            if len(r) not in [3, 4]:
                raise ValueError(f"Invalid range format: {r}")

            start_step, end_step = r[0], r[1]
            if previous_end != -1 and start_step != previous_end + 1:
                raise ValueError(
                    f"Ranges are not continuous between steps {previous_end} and {start_step}"
                )

            previous_end = end_step

    def set_step(self, new_step):
        self.current_step = new_step

    def step(self):
        self.current_step += 1

    @property
    def value(self):
        if self.current_step < self.ranges[0][0]:
            return self.ranges[0][2]
        for r in self.ranges:
            start_step, end_step = r[0], r[1]
            if start_step <= self.current_step <= end_step:
                if len(r) == 3:  # Constant Value
                    return r[2]
                elif len(r) == 4:  # Linear Change
                    start_val, end_val = r[2], r[3]
                    relative_step = self.current_step - start_step
                    total_steps = end_step - start_step
                    return round(
                        start_val + (end_val - start_val) * relative_step / total_steps
                    )
        return self.ranges[-1][2] if len(self.ranges[-1]) == 3 else self.ranges[-1][3]
