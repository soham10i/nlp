class Evaluator:
    """
    Handles the evaluation of the Question Answering system's predictions.
    Currently, it calculates accuracy based on predicted answer option indices
    against ground truth labels.
    """
    def evaluate(self, predictions: list[int], ground_truth_labels: list[int]) -> float:
        """
        Calculates the accuracy of predictions against ground truth labels.

        Args:
            predictions (list[int]): A list of predicted answer option indices (e.g., 0-3).
            ground_truth_labels (list[int]): A list of true answer option indices (e.g., 0-3).

        Returns:
            float: The calculated accuracy, a value between 0.0 and 1.0.

        Raises:
            ValueError: If the lengths of `predictions` and `ground_truth_labels` lists
                        do not match.
        """
        if not isinstance(predictions, list) or not isinstance(ground_truth_labels, list):
            raise TypeError("Predictions and ground_truth_labels must be lists.")

        if len(predictions) != len(ground_truth_labels):
            raise ValueError("Length of predictions and ground truth labels must be the same.")

        if not predictions: # Handles empty list case
            print("Warning: Empty predictions list provided. Accuracy is 0.")
            return 0.0

        correct_count = 0
        for pred, gt in zip(predictions, ground_truth_labels):
            if pred == gt:
                correct_count += 1
        
        accuracy = correct_count / len(predictions) if predictions else 0
        print(f"\n--- Evaluation Results ---")
        print(f"Total Questions: {len(predictions)}")
        print(f"Correctly Answered: {correct_count}")
        print(f"Accuracy: {accuracy:.4f}")
        return accuracy

if __name__ == "__main__":
    evaluator = Evaluator()
    
    # Example usage
    predictions = [0, 1, 2, 0, 3]
    true_labels = [0, 1, 1, 0, 2] # 0,1,0 are correct

    accuracy = evaluator.evaluate(predictions, true_labels)