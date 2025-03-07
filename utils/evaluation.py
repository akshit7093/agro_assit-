"""Evaluation framework for assessing AI Assistant output quality."""

import json
from typing import Dict, List, Any, Optional
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu

class EvaluationMetrics:
    """Metrics for evaluating AI Assistant outputs."""
    
    def __init__(self):
        """Initialize the evaluation metrics."""
        self.metrics = {
            "relevance": [],
            "coherence": [],
            "factual_accuracy": [],
            "helpfulness": [],
            "safety": []
        }
        self.benchmark_results = {}
        self.rouge = Rouge()
        logger.info("Initialized Evaluation Metrics")

    def evaluate_nlp_task(self, task: str, reference: Any, hypothesis: Any) -> Dict[str, Any]:
        """Evaluate NLP tasks using appropriate metrics.
        
        Args:
            task (str): Type of NLP task (summarization, sentiment, ner, qa, retrieval)
            reference: Ground truth/reference data
            hypothesis: Model output/prediction
        
        Returns:
            Dict[str, Any]: Evaluation metrics
        """
        if task == "summarization":
            return self._evaluate_summarization(reference, hypothesis)
        elif task == "sentiment":
            return self._evaluate_sentiment(reference, hypothesis)
        elif task == "ner":
            return self._evaluate_ner(reference, hypothesis)
        elif task == "qa":
            return self._evaluate_qa(reference, hypothesis)
        elif task == "retrieval":
            return self._evaluate_retrieval(reference, hypothesis)
        else:
            raise ValueError(f"Unsupported NLP task: {task}")

    def _evaluate_summarization(self, reference: str, hypothesis: str) -> Dict[str, Any]:
        """Evaluate summarization using ROUGE and BLEU scores.
        
        Args:
            reference (str): Reference summary
            hypothesis (str): Generated summary
        
        Returns:
            Dict[str, Any]: Evaluation metrics
        """
        rouge_scores = self.rouge.get_scores(hypothesis, reference)
        bleu_score = sentence_bleu([reference.split()], hypothesis.split())
        return {
            'rouge': rouge_scores[0],
            'bleu': bleu_score
        }

    def _evaluate_sentiment(self, y_true: list, y_pred: list) -> Dict[str, float]:
        """Evaluate sentiment analysis using accuracy and F1-score.
        
        Args:
            y_true (list): True labels
            y_pred (list): Predicted labels
        
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }

    def _evaluate_ner(self, y_true: list, y_pred: list) -> Dict[str, float]:
        """Evaluate NER using precision, recall and F1-score.
        
        Args:
            y_true (list): True labels
            y_pred (list): Predicted labels
        
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        return {
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }

    def _evaluate_qa(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """Evaluate question answering using Exact Match and F1-score.
        
        Args:
            reference (str): Reference answer
            hypothesis (str): Predicted answer
        
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        em_score = float(reference.lower().strip() == hypothesis.lower().strip())
        f1 = f1_score([reference], [hypothesis], average='weighted')
        return {
            'exact_match': em_score,
            'f1_score': f1
        }

    def _evaluate_retrieval(self, relevant: list, retrieved: list, k: int = 10) -> Dict[str, float]:
        """Evaluate retrieval system using Recall@K and MAP.
        
        Args:
            relevant (list): List of relevant documents
            retrieved (list): List of retrieved documents
            k (int): Number of top results to consider
        
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        relevant_set = set(relevant)
        retrieved_set = set(retrieved[:k])

        recall_at_k = len(relevant_set & retrieved_set) / len(relevant_set)
        average_precision = sum(
            [(len(relevant_set & set(retrieved[:i])) / i) 
             for i in range(1, len(retrieved) + 1) 
             if retrieved[i-1] in relevant_set]
        ) / len(relevant_set)

        return {
            'recall_at_k': recall_at_k,
            'mean_average_precision': average_precision
        }

    def add_human_evaluation(self, response_id: str, metrics: Dict[str, float]) -> None:
        """Add human evaluation scores for a response.
        
        Args:
            response_id (str): Unique identifier for the response
            metrics (Dict[str, float]): Evaluation metrics (1-5 scale)
        """
        for metric, score in metrics.items():
            if metric in self.metrics:
                self.metrics[metric].append({"response_id": response_id, "score": score})
        
        logger.info(f"Added human evaluation for response {response_id}")
    
    def calculate_average_scores(self) -> Dict[str, float]:
        """Calculate average scores for each metric.
        
        Returns:
            Dict[str, float]: Average scores
        """
        averages = {}
        for metric, scores in self.metrics.items():
            if scores:
                avg = sum(item["score"] for item in scores) / len(scores)
                averages[metric] = round(avg, 2)
            else:
                averages[metric] = 0.0
        
        return averages
    
    def run_benchmark(self, test_cases: List[Dict[str, Any]], 
                     assistant_func, model_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run benchmark tests against a set of test cases.
        
        Args:
            test_cases (List[Dict[str, Any]]): List of test cases with inputs and expected outputs
            assistant_func: Function to call the assistant
            model_config (Optional[Dict[str, Any]]): Model configuration
            
        Returns:
            Dict[str, Any]: Benchmark results
        """
        results = {
            "total_cases": len(test_cases),
            "passed": 0,
            "failed": 0,
            "details": []
        }
        
        for i, test_case in enumerate(test_cases):
            try:
                # Get assistant response
                response = assistant_func(test_case["input"], model_config)
                
                # Check if response matches expected output
                passed = self._evaluate_response(response, test_case["expected"])
                
                # Record result
                result_detail = {
                    "case_id": i + 1,
                    "input": test_case["input"],
                    "expected": test_case["expected"],
                    "actual": response,
                    "passed": passed
                }
                
                results["details"].append(result_detail)
                if passed:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                    
            except Exception as e:
                logger.error(f"Error in test case {i+1}: {str(e)}")
                results["details"].append({
                    "case_id": i + 1,
                    "input": test_case["input"],
                    "error": str(e),
                    "passed": False
                })
                results["failed"] += 1
        
        results["pass_rate"] = results["passed"] / results["total_cases"] if results["total_cases"] > 0 else 0
        self.benchmark_results = results
        return results
    
    def _evaluate_response(self, actual: Any, expected: Any) -> bool:
        """Evaluate if a response matches the expected output.
        
        This is a simple implementation that can be extended with more sophisticated
        comparison methods like semantic similarity.
        
        Args:
            actual (Any): Actual response
            expected (Any): Expected response
            
        Returns:
            bool: Whether the response is acceptable
        """
        # For string responses, check if expected text is contained in the actual response
        if isinstance(expected, str) and isinstance(actual, str):
            return expected.lower() in actual.lower()
        
        # For dict responses, check if expected keys and values are present
        if isinstance(expected, dict) and isinstance(actual, dict):
            for key, value in expected.items():
                if key not in actual:
                    return False
                if isinstance(value, str) and isinstance(actual[key], str):
                    if value.lower() not in actual[key].lower():
                        return False
                elif actual[key] != value:
                    return False
            return True
        
        # Default comparison
        return actual == expected
    
    def save_results(self, filepath: str) -> None:
        """Save evaluation results to a file.
        
        Args:
            filepath (str): Path to save the results
        """
        data = {
            "metrics": self.metrics,
            "averages": self.calculate_average_scores(),
            "benchmark_results": self.benchmark_results
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved evaluation results to {filepath}")