class EarlyStopper:
    def __init__(self, min_delta=0, tolerance=5):
        self.min_delta = min_delta
        self.tolerance = tolerance        
        self.min_error = float('inf')
        self.error_counter = 0
        self.max_score = float('-inf')
        self.score_counter = 0

    def get_error(self):
        return self.min_error

    def get_score(self):
        return self.max_score

    def set_error(self, value):
        if value < self.min_delta:
            self.min_error = value
            return True
        return False

    def set_score(self, value):
        if value > self.max_score:
            self.max_score = value
            return True
        return False
        
    def stop(self, current_error=None, current_score=None):
        assert (current_score is not None) or (current_error is not None)
        if current_error is not None:
            if current_error < self.min_error:
                self.min_error = current_error
                self.error_counter = 0
            elif current_error > (self.min_error + self.min_delta):
                self.error_counter += 1
                if self.error_counter >= self.tolerance:
                    return True
        if current_score is not None:
            if current_score > self.max_score:
                self.max_score = current_score
                self.score_counter = 0
            elif current_score < (self.max_score - self.min_delta):
                self.score_counter += 1
                if self.score_counter >= self.tolerance:
                    return True
        return False



if __name__ == '__main__':
    early_stopping = EarlyStopper(min_delta=5, tolerance=2)

    train_loss = [
        642.14990234,
        601.29278564,
        561.98400879,
        530.01501465,
        497.1098938,
        466.92709351,
        438.2364502,
        413.76028442,
        391.5090332,
        370.79074097,
    ]
    validate_loss = [
        509.13619995,
        497.3125,
        506.17315674,
        497.68960571,
        505.69918823,
        459.78610229,
        480.25592041,
        418.08630371,
        446.42675781,
        372.09902954,
    ]

    for (idx,val_loss) in enumerate(validate_loss):
        print(f"val_loss: {validate_loss[idx]}")
        if early_stopping.stop(val_loss):
            print("We are at epoch:", idx)
            break
