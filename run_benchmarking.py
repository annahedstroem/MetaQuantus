from .configs import setup_estimators, setup_xai_methods
from .. import metaquantus


# Prepare analyser suite
analyser_suite = {
    "Model Resilience Test":
        metaquantus.ModelPerturbationTest(**{
            "noise_type": "multiplicative",
            "mean": 1.0,
            "std": 0.001,
            "type": "Resilience",
        }
                                          ),
    "Model Adversary Test":
        metaquantus.ModelPerturbationTest(**{
            "noise_type": "multiplicative",
            "mean": 1.0,
            "std": 2.0,
            "type": "Adversary",
        }
                                          ),
    "Input Resilience Test":
        metaquantus.InputPerturbationTest(**{
            "noise": 0.001,
            "type": "Resilience",
        }
                                          ),
    "Input Adversary Test":
        metaquantus.InputPerturbationTest(**{
            "noise": 5.0,
            "type": "Adversary",
        }
                                          ),
}
