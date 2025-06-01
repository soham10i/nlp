class Evaluator:
    def evaluate(self, predictions, ground_truth_labels):
        """
        Calculates accuracy. [cite: 300, 301]
        predictions: List of predicted answer option indices (0-3).
        ground_truth_labels: List of true answer option indices (0-3).
        """
        if len(predictions) != len(ground_truth_labels):
            raise ValueError("Length of predictions and ground truth labels must be the same.")

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